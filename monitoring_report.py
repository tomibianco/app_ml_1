from prefect import flow, task
import pandas as pd
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configuración de la conexión a la base de datos
DB_CONNECTION_STRING = "postgresql://user:password@host:port/database"

# 1. Tarea para extraer datos desde SQL
@task
def extract_sql_data(query: str) -> pd.DataFrame:
    engine = create_engine(DB_CONNECTION_STRING)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

# 2. Tarea para generar el reporte de Evidently
@task
def generate_evidently_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, report_name: str):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report_path = f"reports/{report_name}.html"
    report.save_html(report_path)
    print(f"Reporte generado: {report_path}")

# 3. Flujo principal
@flow(name="SQL_Monitoring_Pipeline")
def monitoring_pipeline():
    # Consultas SQL
    reference_query = "SELECT * FROM training_data LIMIT 1000;"
    current_query = "SELECT * FROM production_data WHERE date >= NOW() - INTERVAL '1 day';"

    # Extracción de datos
    reference_data = extract_sql_data(reference_query)
    current_data = extract_sql_data(current_query)

    # Generar reporte de data drift
    generate_evidently_report(reference_data, current_data, report_name="data_drift_report")

if __name__ == "__main__":
    monitoring_pipeline()




from prefect.blocks.notifications import Email

@task
def send_alert(message: str):
    email_block = Email.load("my-email-block")  # Configura tu bloque de Prefect
    email_block.send(message)




from prefect import task, flow
from prefect.blocks.notifications import Email
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import pandas as pd

# 1. Task para generar el informe Evidently y guardarlo como HTML
@task
def generate_evidently_report():
    # Simulación de datos: ejemplo con drift
    reference_data = pd.DataFrame({"column_a": [1, 2, 3, 4, 5]})
    current_data = pd.DataFrame({"column_a": [1, 2, 6, 7, 8]})

    # Generar el reporte
    report = Report(metrics=[ColumnDriftMetric(column_name="column_a")])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Guardar el reporte en archivo HTML
    report_file = "evidently_report.html"
    report.save_html(report_file)
    print(f"Reporte Evidently guardado en {report_file}")
    return report_file

# 2. Task para enviar un correo con el archivo adjunto
@task
def send_alert_with_attachment(message: str, attachment_path: str):
    # Cargar el bloque de email configurado en Prefect
    email_block = Email.load("my-email-block")
    
    # Leer el archivo Evidently
    with open(attachment_path, "rb") as file:
        attachment_content = file.read()

    # Enviar el email con el archivo adjunto
    email_block.send(
        email_to="destinatario@ejemplo.com",  # Cambia esto por tu correo
        subject="Nuevo reporte Evidently generado",
        msg=message,
        attachments=[{"file_name": "evidently_report.html", "content": attachment_content}]
    )
    print("Correo enviado exitosamente con el reporte adjunto.")

# 3. Definir el flujo principal
@flow
def evidently_report_flow():
    # Generar el reporte
    report_file = generate_evidently_report()
    
    # Enviar alerta con el reporte adjunto
    send_alert_with_attachment(
        message="Se ha generado un nuevo informe Evidently. Revisa el archivo adjunto.",
        attachment_path=report_file
    )

# Ejecutar el flujo
if __name__ == "__main__":
    evidently_report_flow()














from prefect import flow, task
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

@task
def load_data():
    reference_data = pd.read_csv("data/reference_data.csv")
    current_data = pd.read_csv("data/current_data.csv")
    return reference_data, current_data

@task
def generate_data_drift_report(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("reports/data_drift_report.html")
    print("Reporte de Data Drift generado.")

@flow(name="Evidently_Monitoring_Flow")
def monitoring_flow():
    reference_data, current_data = load_data()
    generate_data_drift_report(reference_data, current_data)

if __name__ == "__main__":
    monitoring_flow()
