from pydantic import BaseModel, Field


class User(BaseModel):
    email: str
    password: str

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