import mlflow.pyfunc
from jwt_manager import create_token
from utils import load_production_model, validate_csv, load_csv, preprocess_dataframe, make_predictions, package_predictions
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.security import OAuth2PasswordRequestForm
from schemas import User, UserOut, Token
from database import SessionLocal
from crud import get_user_by_email, verify_password, create_user_db, create_db_and_tables
from sqlalchemy.orm import Session


app = FastAPI()
app.title = "Proyecto Mora Banco"
app.version = "Beta 2.0"


db_session = None
mlflow.set_tracking_uri("sqlite:///backend.db")
model = load_production_model()


@app.on_event("startup")
async def startup():
    """
    Endpoint que inicia las tablas de bases de datos, y en caso de no existir, las crea.
    """
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


@app.post("/update_model")
async def update_model():
    """
    Endpoint que actualiza el modelo cuando se despliega uno nuevo en producción.
    """
    global model
    model = load_production_model()


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
    validate_csv(file)
    df = load_csv(file)
    columns_to_drop = ["linea_sf", "deuda_sf", "exp_sf"]
    columns_to_encode = ["zona", "nivel_educ", "vivienda"]
    df_encoded = preprocess_dataframe(df, columns_to_drop, columns_to_encode)
    pred_labels = make_predictions(model, df_encoded)
    output = package_predictions(df.to_dict(orient="records"), pred_labels)
    return {"predictions": output}