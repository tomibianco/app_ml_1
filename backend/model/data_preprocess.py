import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Union 


class DataStrategy(ABC):
    """
    Clase Abstracta para definir estrategia de manejo de data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocess(DataStrategy):
     """
    Clase para preprocesamiento de los datos
    """ 
     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:    
            data = data.drop(
                [
                    "linea_sf",
                    "deuda_sf",
                    "exp_sf"
                ],
                axis=1,
            )
            # Agregar aquí proceso de imputación de valores faltantes.
            logging.info("Data preprocesada correctamente")
            return data
        except Exception as e:
            logging.error("La data no ha sido procesada correctamente", exc_info=True)
            raise e

    
class DataSplit(DataStrategy):
    """
    Clase para dividir data entre train y test
    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            data_cat = pd.get_dummies(data, columns = [
                "zona",
                "nivel_educ",
                "vivienda"
                ])    
            X = data_cat.drop("mora", axis=1)
            y = data_cat["mora"]
            # Agregar aquí en caso de ser necesario, proceso de escalamiento de datos.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("Data dividida correctamente")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("La data no ha sido dividida correctamente", exc_info=True)
            raise e
    
class DataCleaning:
    """
    Clase que limpia, preprocesa la data y la divide en train y test
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Inicializa la clase DataCleaning con una estrategia especifica"""
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Maneja la data en funcion de la estrategia"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error en el manejo de la data", exc_info=True)
            raise e