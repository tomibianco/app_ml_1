from sqlalchemy import Column, Integer, String, Float, JSON
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    atraso = Column(Integer)
    vivienda = Column(String)
    edad = Column(Integer)
    dias_lab = Column(Integer)
    exp_sf = Column(Float)
    nivel_ahorro = Column(Integer)
    ingreso = Column(Float)
    linea_sf = Column(Float)
    deuda_sf = Column(Float)
    score = Column(Integer)
    zona = Column(String)
    clasif_sbs = Column(Integer)
    nivel_educ = Column(Integer)
    prediction = Column(String, nullable=False)