from jwt import encode, decode
import token
from passlib.context import CryptContext


# Iniciar el contexto de hash de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_token(data: dict):
    token: str = encode(payload=data, key="secret_key", algorithm="HS256")
    return token

def validate_token(data: dict):
    data: dict = decode(token, key="secret_key", algorithms=["HS256"])
    return data