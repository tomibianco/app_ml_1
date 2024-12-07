import logging
import pandas as pd
from prefect import task
from sqlalchemy import create_engine


class IngestData():
    """
    Clase base para la ingesta de datos. Define una interfaz común.
    """
    def get_data(self) -> pd.DataFrame:
        pass


class IngestDataCSV(IngestData):
    """
    Clase que ingesta los datos desde el CSV fuente, y los devuelve en un Dataframe. 
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.file_path)
            logging.info(f"Datos cargados correctamente desde {self.file_path}")
            return data
        except Exception as e:
            logging.error(f"Error al cargar el archivo CSV: {self.file_path}", exc_info=True)
            raise e


class IngestDataDB(IngestData):
    """
    Clase que ingesta datos desde la Base de Datos a través de un Query, y los devuelve en un Dataframe.
    """
    def __init__(self, connection_string: str, query: str) -> None:
        self.connection_string = connection_string
        self.query = query

    def get_data(self) -> pd.DataFrame:
        try:
            engine = create_engine(self.connection_string)
            data = pd.read_sql(self.query, engine)
            logging.info("Datos cargados correctamente desde la base de datos.")
            return data
        except Exception as e:
            logging.error("Error al cargar datos desde la base de datos.", exc_info=True)
            raise e


@task
def ingest_data(source: str, **kwargs) -> pd.DataFrame:
    """
    Task para la ingesta de datos.
    
    Args:
        source (str): Fuente de datos ("csv" o "db").
        kwargs: Argumentos para la clase correspondiente.
    
    Devuelve:
        data: pd.DataFrame
    """
    try:
        if source == "csv":
            ingestor = IngestDataCSV(kwargs["file_path"])
        elif source == "db":
            ingestor = IngestDataDB(kwargs["connection_string"], kwargs["query"])
        else:
            raise ValueError(f"Fuente no reconocida: {source}")

        data = ingestor.get_data()
        return data
    except Exception as e:
        logging.error("Error en el proceso de ingesta de datos.", exc_info=True)
        raise e