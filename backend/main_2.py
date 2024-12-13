import io
import pandas as pd
import mlflow.pyfunc
from jwt_manager import create_token
from config import label_mapping, columns_train
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.security import OAuth2PasswordRequestForm
from schemas import User, UserOut, Token
from database import SessionLocal
from crud import get_user_by_email, verify_password, create_user_db, create_db_and_tables
from sqlalchemy.orm import Session


app = FastAPI()
app.title = "Proyecto Mora Banco"
app.version = "Beta 2.0"


def load_production_model(model_name="Model"):
    return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")


db_session = None
mlflow.set_tracking_uri("sqlite:///backend.db")
model = load_production_model()


@app.on_event("startup")
async def startup():
    create_db_and_tables()
    global model
    model = load_production_model()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

@app.post("/predictions", tags=["predictions_input_csv"])
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
        # df_cleaned = df.drop(["linea_sf", "deuda_sf", "exp_sf"], axis=1)
        df_encoder = pd.get_dummies(df, columns = ["zona", "nivel_educ", "vivienda"])
        for col in columns_train:
            if col not in df_encoder.columns:
                df_encoder[col] = 0
        df_encoder = df_encoder[columns_train]
        pred = model.predict(df_encoder)
        pred_labels = [label_mapping[p] for p in pred]
        output = []
        for original_row, pred_label in zip(df.to_dict(orient="records"), pred_labels):
            output.append({"input": original_row, "prediction": pred_label})
        return {"predictions": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")