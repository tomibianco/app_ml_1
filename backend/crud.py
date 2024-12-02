from sqlalchemy.orm import Session
from models import User, Predictions
from passlib.context import CryptContext
from database import engine, Base

# Iniciar el contexto de hash de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Función para obtener el hash de la contraseña
def get_password_hash(password: str):
    return pwd_context.hash(password)

# Función para verificar si la contraseña es correcta
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Crear usuario
def create_user_db(db: Session, user: User):
    db_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Obtener un usuario por correo
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_db_and_tables():
    """
    Crea las tablas en la base de datos.
    """
    Base.metadata.create_all(bind=engine)