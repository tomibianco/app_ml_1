import logging
import pandas as pd
from prefect import task
from model.data_ingest import IngestDataCSV, IngestDataDB


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