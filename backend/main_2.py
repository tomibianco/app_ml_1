model_name = "Model"
stage = "Production"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

predictions = model.predict(data)




import joblib
import os
import io
from io import StringIO
import pandas as pd
import mlflow.pyfunc
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
app.version = "Beta 2.0"

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
        df_cleaned = df.dropna(subset=["linea_sf", "deuda_sf", "exp_sf"])
        df_encoder = pd.get_dummies(df_cleaned, columns = ["zona", "nivel_educ", "vivienda"])
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





# Cargar el modelo en producción desde MLflow
def load_production_model(model_name="my_model"):
    return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

model = load_production_model()

@app.post("/predict")
def predict(data: dict):
    input_data = pd.DataFrame(data["data"], columns=data["columns"])
    predictions = model.predict(input_data)
    return {"predictions": predictions.tolist()}

@app.on_event("startup")
async def update_model():
    global model
    model = load_production_model()







#AGREGAR DESPUES DEL ENTRENAMIENTO DEL MODELO - PARA INFORMAR ACTUALIZACION A API
# Cada vez que el pipeline de Prefect registra un nuevo modelo en Production, tu API debe actualizar el modelo cargado.
# Esto se puede manejar automáticamente.

# A. Configurar Prefect para notificar a la API
# Agrega una tarea al pipeline de Prefect para enviar una solicitud a la API para actualizar el modelo:

import requests

def notify_api(api_url="http://localhost:8000/update_model"):
    response = requests.post(api_url)
    if response.status_code == 200:
        print("Modelo actualizado en la API.")




usar label mapping en config