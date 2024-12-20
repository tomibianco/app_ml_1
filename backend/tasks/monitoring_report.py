import os
import logging
from prefect import task
import pandas as pd
from database import engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# @task
# def extract_sql_data(query: str) -> pd.DataFrame:
#     """
#     Extrae la data de query de SQL, y crea un Dataframe.
#     """
#     try:
#         with engine.connect() as connection:
#             df = pd.read_sql(query, connection)
#         logging.info("Se han cargado los datos del query satisfactoriamente en Dataframe")
#         return df
#     except Exception as e:
#         logging.error("Error al extrar los datos de Query", exc_info=True)
#         raise e

@task
def load_csv_data():
    """
    Requiere cargar 2 CSV, con datos predecidos, y datos reales para comparar.
    """
    try:
        reference_data = pd.read_csv("/home/tomibianco/appml/data/reference_data.csv")
        current_data = pd.read_csv("/home/tomibianco/appml/data/current_data.csv")
        logging.info("Se han cargado los datos satisfactoriamente en Dataframe")
        return reference_data, current_data
    except Exception as e:
        logging.error("Error en el proceso de carga de datos desde CSV", exc_info=True)
        raise e

@task
# def generate_evidently_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
def generate_report(reference_data, current_data):
    """
    Carga ambos datasets, crea reporte de Data Drift y lo guarda.
    """
    try:
        report_dir = "reports"
        report_path = os.path.join(report_dir, "report_data_drift.html")
        os.makedirs(report_dir, exist_ok=True)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report_path = f"reports/report_data_drift.html"
        report.save_html(report_path)
        logging.info(f"Reporte generado: {report_path}")
    except Exception as e:
        logging.error("Error en el proceso de generación de reporte.", exc_info=True)
        raise e