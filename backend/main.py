import joblib
import os
import io
from io import StringIO
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse


app = FastAPI()
app.title = "Proyecto Mora Banco"
app.version = "Beta 3.0"

scaler = None
model = None
db_session = None

label_mapping = {
    0: "Paga al dia",
    1: "Cliente moroso"}

columns_train = ['atraso',
    'edad',
    'dias_lab',
    'exp_sf',
    'nivel_ahorro',
    'ingreso',
    'linea_sf',
    'deuda_sf',
    'score',
    'clasif_sbs',
    'zona_Amazonas',
    'zona_Ancash',
    'zona_Apurimac',
    'zona_Arequipa',
    'zona_Ayacucho',
    'zona_Cajamarca',
    'zona_Callao',
    'zona_Cuzco',
    'zona_Huanuco',
    'zona_Ica',
    'zona_Junin',
    'zona_La Libertad',
    'zona_Lambayeque',
    'zona_Lima',
    'zona_Loreto',
    'zona_Madre de Dios',
    'zona_Moquegua',
    'zona_Pasco',
    'zona_Piura',
    'zona_Puno',
    'zona_San Martin',
    'zona_Tacna',
    'zona_Tumbes',
    'zona_Ucayali',
    'nivel_educ_SECUNDARIA',
    'nivel_educ_SIN EDUCACION',
    'nivel_educ_TECNICA',
    'nivel_educ_UNIVERSITARIA',
    'vivienda_ALQUILADA',
    'vivienda_FAMILIAR',
    'vivienda_PROPIA']

class Input(BaseModel):
    atraso: int
    vivienda: str
    edad: int
    dias_lab: int
    exp_sf: float
    nivel_ahorro: int
    ingreso: float
    linea_sf: float
    deuda_sf: float
    score: int
    zona: str
    clasif_sbs: int
    nivel_educ: str

class PredictionOutput(BaseModel):
    prediction: str


# @app.on_event("startup")
# def connect_to_database():
#     """
#     Establece la conexión con la base de datos.
#     """
#     global db_session
#     db_session = SessionLocal()


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
    return {"Mensaje": "API de Predicciones con Modelo de Machine Learning"}


@app.post("/predict_input_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Carga CSV, genera predicciones, y las almacena en base de datos.
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


@app.post("/predict_input_download_csv")
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
        csv_file = StringIO()
        df_output.to_csv(csv_file, index=False)
        csv_file.seek(0)
        return StreamingResponse(csv_file, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")


@app.post("/predict_input", response_model = PredictionOutput)
def predict_input(data: Input):
    """
    Realiza predicción en tiempo real a partir de input del usuario.
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