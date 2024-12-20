import logging
from prefect import flow
from tasks.monitoring_report import load_csv_data, generate_report, send_email_alert


@flow
def monitoring_pipeline():
    """
    Pipeline de generación y envío por correo de Reporte de Data Drift con Evidently.
    """
    try:
        # reference_query = "SELECT * FROM training_data LIMIT 1000;"
        # current_query = "SELECT * FROM production_data WHERE date >= NOW() - INTERVAL '1 day';"
        # reference_data = extract_sql_data(reference_query)
        # current_data = extract_sql_data(current_query)

        reference_data, current_data = load_csv_data()
        generate_report(reference_data, current_data)
        logging.info("Reporte generado satisfactoriamente")
        send_email_alert(
            message="Se ha generado un nuevo informe Evidently. Revisa el archivo adjunto.",
            attachment_path=f"reports/report_data_drift.html"
            )
        logging.info("Correo con reporte enviado satisfactoriamente")
    except Exception as e:
        logging.error("Error durante ejecución de pipeline de monitoreo", exc_info=True)
    

if __name__ == "__main__":
    monitoring_pipeline()

    # monitoring_pipeline.from_source(
    #     source="https://github.com/tomibianco/app_ml_1.git",
    #     entrypoint="/home/tomibianco/appml/backend/monitoring_pipeline.py:monitoring_pipeline"
    # ).deploy(
    #     name="Scheduled Monitoring Pipeline",
    #     work_pool_name="my-managed-pool",
    # )
