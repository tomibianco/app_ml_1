import os
import logging
from prefect import task
import pandas as pd
from prefect_email import EmailServerCredentials, email_send_message
from database import engine
from sqlalchemy import create_engine
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

@task
def send_email_alert(message: str, attachment_path: str):
    """
    Adjunta reporte y lo envía por correo.
    """
    try:
        email_server_credentials = EmailServerCredentials.load("my-email-block-2")
        with open(attachment_path, "rb") as file:
            attachment_content = file.read()

        # Enviar el email con el archivo adjunto
        email_send_message(
            email_server_credentials=email_server_credentials,
            email_to="contacto@vistalia.cl",
            subject="Nuevo Reporte Evidently",
            msg=message,
            attachments=[{"file_name": "report_data_drift.html", "content": attachment_content}]
        )
        logging.info("Correo enviado exitosamente con el reporte adjunto.")
    except Exception as e:
        logging.error("Error en el proceso de envío de reporte por correo.", exc_info=True)
        raise e