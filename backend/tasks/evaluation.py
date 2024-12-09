import logging
import numpy as np
import pandas as pd
import mlflow
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from model.evaluation import Accuracy, Precision, Recall, F1
from prefect import task


@task
def evaluation(
    model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[float, "accuracy"], 
    Annotated[float, "precision"], 
    Annotated[float, "recall"], 
    Annotated[float, "f1"]
]:
    """
    Args:
        model: ClassifierMixin
        X_test: pd.DataFrame
        y_test: pd.Series
    Devuelve:
        Accuracy: float
        Precision: float
        Recall: float
        F1_Score: float
    """
    try:
        prediction = model.predict(X_test)

        # Usando la clase para predecir Accuracy
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_score(y_test, prediction)

        # Usando la clase para predecir Precision
        precision_class = Precision()
        precision = precision_class.calculate_score(y_test, prediction)

        # Usando la clase para predecir Recall
        recall_class = Recall()
        recall = recall_class.calculate_score(y_test, prediction)

        # Usando la clase para predecir F1_Score
        f1_score_class = F1()
        f1_score = f1_score_class.calculate_score(y_test, prediction)

        logging.info("Cálculo satisfactorio de métricas")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)

        return accuracy, precision, recall, f1_score
    except Exception as e:
        logging.error("Error durante el cálculo de métricas", exc_info=True)
        raise e