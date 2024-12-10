import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluation(ABC):
    """
    Clase abstracta que define la estrategia para evaluar las métricas de rendimiento del modelo.
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray , y_pred: np.ndarray) -> float:
        pass


class Accuracy(Evaluation):
    """
    Estrategia de evaluación de métrica que utiliza Accuracy
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Devuelve:
            accuracy: float
        """
        try:
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("Accuracy:" + str(accuracy))
            return accuracy
        except Exception as e:
            logging.error("Error ocurrido en el cálculo de clase Accuracy", exc_info=True)
            raise e
        
class Precision(Evaluation):
    """
    Estrategia de evaluación de métrica que utiliza Precision
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Devuelve:
            precision: float
        """
        try:
            precision = precision_score(y_true, y_pred)
            logging.info("Precision:" + str(precision))
            return precision
        except Exception as e:
            logging.error("Error ocurrido en el cálculo de clase Precision", exc_info=True)
            raise e
        
class Recall(Evaluation):
    """
    Estrategia de evaluación de métrica que utiliza Recall
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Devuelve:
            recall: float
        """
        try:
            recall = recall_score(y_true, y_pred)
            logging.info("Recall:" + str(recall))
            return recall
        except Exception as e:
            logging.error("Error ocurrido en el cálculo de clase Recall", exc_info=True)
            raise e
        
class F1(Evaluation):
    """
    Estrategia de evaluación de métrica que utiliza F1
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Devuelve:
            f1: float
        """
        try:
            f1 = f1_score(y_true, y_pred)
            logging.info("F1:" + str(f1))
            return f1
        except Exception as e:
            logging.error("Error ocurrido en el cálculo de clase F1", exc_info=True)
            raise e