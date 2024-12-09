import os
import logging
import joblib
from typing import Any
from prefect import task


model_path = os.path.join(os.path.dirname(__file__), "model")

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