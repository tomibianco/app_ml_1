from pydantic import BaseModel, Field


class User(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    email: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

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