import logging
import pandas as pd
import mlflow.sklearn
from prefect import task
from model.model_dev import RandomForest, XGBoost, HyperparameterTuner
from sklearn.base import ClassifierMixin
from tasks.config import model_name


@task
def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ClassifierMixin:
    """
    Args:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Devuelve:
        model: ClassifierMixin
    """
    try:
        model = None
        tuner = None

        if model_name == "randomforest":
            model = RandomForest()
        elif model_name == "xgboost":
            model = XGBoost()
        else:
            raise ValueError("Nombre de modelo no soportado")
        
        tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)
        best_params = tuner.optimize()
        trained_model = model.train(X_train, y_train, **best_params)
        logging.info("Modelo entrenado exitosamente")
        mlflow.sklearn.log_model(trained_model, "model")
        print(f"default artifacts URI: {mlflow.get_artifact_uri()}")
        return trained_model
    except Exception as e:
        logging.error("Error durante el entrenamiento del modelo", exc_info=True)
        raise e