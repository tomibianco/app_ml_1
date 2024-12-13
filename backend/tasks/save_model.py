import requests
import logging
import joblib
from typing import Any
from prefect import task


model_path = "/home/tomibianco/appml/backend/model"

def notify_api(api_url="http://localhost:8000/update_model"):
    """
    Envía una solicitud HTTP POST para notificar a la API que el modelo en producción ha sido actualizado.
    """
    try:
        response = requests.post(api_url)
        if response.status_code == 200:
            print("Modelo actualizado en la API.")
        else:
            print(f"Error al notificar a la API: {response.status_code}")
    except Exception as e:
        print(f"Excepción al intentar notificar a la API: {e}")


@task
def save_model(model: Any, model_path: str):
    """
    Guarda el modelo generado.
    """
    try:
        logging.info(f"Guardando modelo en {model_path}")
        joblib.dump(model, f"{model_path}/model.pkl")
    except Exception as e:
        logging.error("Ha ocurrido un problema al guardar el modelo", exc_info=True)
        raise e
    
@task
def notify_api_task(api_url="http://localhost:8000/update_model"):
    """
    Notifica a la API que el modelo ha sido actualizado.
    """
    notify_api(api_url)