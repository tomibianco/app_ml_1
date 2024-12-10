import optuna
from abc import ABC, abstractmethod
import mlflow
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class Model(ABC):
    """
    Clase asbtracta de base para todos los modelos a usar.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Entrena el modelo para la data dada.

        Args:
            X_train: Data de entrenamiento
            y_train: Objetivo de entrenamiento
        """
        pass

    @abstractmethod
    def optimize(self, X_train, y_train, X_test, y_test):
        """
        Optimiza los hiperparámetros del modelo.

        Args:
            trial: Objeto de prueba.
            X_train: Datos de entrenamiento.
            y_train: Datos objetivo a predecir.
            X_test: Datos de prueba.
            y_test: Objetivo a predecir de prueba.
        """
        pass

class RandomForest(Model):
    """
    Modelo de Random Forest que implementará la interfaz.
    """
    def train(self, X_train, y_train, **kwargs):
        model = RandomForestClassifier(**kwargs)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        model = self.train(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return model.score(X_test, y_test)

class XGBoost(Model):
    """
    Modelo de XGBoost que implementará la interfaz.
    """
    def train(self, X_train, y_train, **kwargs):
        model = XGBClassifier(**kwargs)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "xgboost")
        return model

    def optimize(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        model = self.train(X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return model.score(X_test, y_test)
    
class HyperparameterTuner:
    """
    Clase para realizar ajustes de hiperparámetros. Utiliza la estrategia del modelo para realizar el ajuste.
    """
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, self.X_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params