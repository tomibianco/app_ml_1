import joblib
import os
import io
from io import StringIO
import pandas as pd
from jwt_manager import create_token
from config import label_mapping, columns_train
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from schemas import User, UserOut, Token, Input, PredictionOutput
from database import engine, Base, SessionLocal
from crud import get_user_by_email, verify_password, create_user_db, create_db_and_tables
from sqlalchemy.orm import Session
from models import Predictions


app = FastAPI()
app.title = "Proyecto Mora Banco"
app.version = "Beta 1.0"

scaler = None
model = None
db_session = None


#initial event of app - db initialization 
@app.on_event("startup")
async def startup():
    create_db_and_tables()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def load_resources():
    """
    Carga el modelo y el escalador cuando la aplicación inicia.
    """
    global scaler, model
    try:
        scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
        model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error al cargar recursos: {e}")


@app.get("/")
def index():
    """
    Ruta de prueba para verificar la API.
    """
    return {"Mensaje": "API de Predicciones de Mora para Clientes Bancarios"}


@app.post("/login", response_model=Token, tags=["login"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Ruta de login para obtener el token.
    """
    user = get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    access_token = create_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/", response_model=UserOut, tags=["login"])
def create_user(user: User, db: Session = Depends(get_db)):
    """
    Ruta para crear un usuario nuevo.
    """
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="El correo ya ha sido registrado.")
    new_user = create_user_db(db=db, user=user)
    return UserOut.from_orm(new_user)


@app.post("/predict_input_csv", tags=["predictions"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Carga CSV, genera predicciones, y las entrega en json.
    """
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="El archivo debe ser de tipo CSV.")
    try:
        content = await file.read()
        csv_content = io.StringIO(content.decode("utf-8"))
        df = pd.read_csv(csv_content)
        df_cleaned = df.dropna(subset=["linea_sf", "deuda_sf", "exp_sf"])
        df_encoder = pd.get_dummies(df_cleaned, columns = ["zona", "nivel_educ", "vivienda"])
        for col in columns_train:
            if col not in df_encoder.columns:
                df_encoder[col] = 0
        df_encoder = df_encoder[columns_train]
        df_scaled = scaler.transform(df_encoder)
        pred = model.predict(df_scaled)
        pred_labels = [label_mapping[p] for p in pred]
        output = []
        for original_row, pred_label in zip(df.to_dict(orient="records"), pred_labels):
            output.append({"input": original_row, "prediction": pred_label})
        return {"predictions": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")


@app.post("/predict_input_download_csv", tags=["predictions"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Carga CSV, genera predicciones, y descarga predicciones en CSV.
    """
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="El archivo debe ser de tipo CSV.")
    try:
        content = await file.read()
        csv_content = io.StringIO(content.decode("utf-8"))
        df = pd.read_csv(csv_content)
        df_cleaned = df.dropna(subset=["linea_sf", "deuda_sf", "exp_sf"])
        df_encoder = pd.get_dummies(df_cleaned, columns = ["zona", "nivel_educ", "vivienda"])
        for col in columns_train:
            if col not in df_encoder.columns:
                df_encoder[col] = 0
        df_encoder = df_encoder[columns_train]
        df_scaled = scaler.transform(df_encoder)
        pred = model.predict(df_scaled)
        pred_labels = [label_mapping[p] for p in pred]
        output = []
        for original_row, pred_label in zip(df.to_dict(orient="records"), pred_labels):
            row_values = list(original_row.values())
            row_values.append(pred_label)
            output.append(row_values)
        columns = list(df.columns) + ['prediction']
        df_output = pd.DataFrame(output, columns=columns)
        db = SessionLocal()
        db_records = [
            Predictions(
                atraso=row["atraso"],
                vivienda=row["vivienda"],
                edad=row["edad"],
                dias_lab=row["dias_lab"],
                exp_sf=row["exp_sf"],
                nivel_ahorro=row["nivel_ahorro"],
                ingreso=row["ingreso"],
                linea_sf=row["linea_sf"],
                deuda_sf=row["deuda_sf"],
                score=row["score"],
                zona=row["zona"],
                clasif_sbs=row["clasif_sbs"],
                nivel_educ=row["nivel_educ"],
                prediction=row["prediction"]
            )
            for _, row in df_output.iterrows()
        ]
        db.add_all(db_records)
        db.commit()
        csv_file = StringIO()
        df_output.to_csv(csv_file, index=False)
        csv_file.seek(0)
        return StreamingResponse(csv_file, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")


@app.post("/predict_input", response_model = PredictionOutput, tags=["predictions"])
def predict_input(data: Input):
    """
    Realiza predicción en tiempo real a partir de input del usuario, y lo entrega en json.
    """
    try:
        df = pd.DataFrame(
            [[
                data.atraso,
                data.vivienda,
                data.edad,
                data.dias_lab,
                data.exp_sf,
                data.nivel_ahorro,
                data.ingreso,
                data.linea_sf,
                data.deuda_sf,
                data.score,
                data.zona,
                data.clasif_sbs,
                data.nivel_educ
            ]],
            columns=[
                "atraso", "vivienda", "edad", "dias_lab", "exp_sf",
                "nivel_ahorro", "ingreso", "linea_sf", "deuda_sf",
                "score", "zona", "clasif_sbs", "nivel_educ"
            ])
        df_encoder = pd.get_dummies(df, columns = ['zona', 'nivel_educ', 'vivienda'])
        for col in columns_train:
            if col not in df_encoder.columns:
                df_encoder[col] = 0
        df_encoder = df_encoder[columns_train]
        df_scaled = scaler.transform(df_encoder)
        pred = model.predict(df_scaled)
        pred_label = label_mapping[pred[0]]
        return {"prediction": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")