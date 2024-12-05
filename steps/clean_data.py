import logging
import pandas as pd
from prefect import task
from model.data_preprocess import DataPreprocess, DataSplit, DataCleaning
from typing import Annotated, Tuple

@task
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clase de limpieza de datos, que preprocesa y divide en entrenamiento y testeo.

    Args:
        data: pd.DataFrame
    """
    try:
        preprocess_strategy = DataPreprocess()
        data_cleaning = DataPreprocess(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        split_strategy = DataSplit()
        data_cleaning = DataCleaning(preprocessed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data preprocesada y dividida exitosamente")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("", exc_info=True)
        raise e