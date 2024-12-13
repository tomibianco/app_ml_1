import mlflow
import mlflow.sklearn
from prefect import flow
from mlflow.tracking import MlflowClient
from tasks.ingest_data import ingest_data
from tasks.clean_data import clean_data
from tasks.model_train import model_train
from tasks.evaluation import evaluation
from tasks.save_model import notify_api_task, save_model, model_path


@flow
def train_pipeline(source: str, **kwargs):
    """
    Pipeline de entrenamiento de modelo.
    
    Args:
        source: Indica la fuente de datos ("csv" o "db").
        kwargs: Argumentos adicionales según la fuente de datos.
            - Si source == "csv": kwargs debe incluir "file_path".
            - Si source == "db": kwargs debe incluir "connection_string" y "query".

    Guarda el modelo en local y en Mlflow de ser suficientemente bueno.
    """
    mlflow.set_tracking_uri("sqlite:///backend.db")
    mlflow.set_experiment("Experiment_1")

    with mlflow.start_run():
        data = ingest_data(source=source, **kwargs)
        X_train, X_test, y_train, y_test = clean_data(data)
        model = model_train(X_train, X_test, y_train, y_test)
        accuracy, precision, recall, f1_score = evaluation(model, X_test, y_test)

        # Guardar modelo y desplegar en Mlflow si Accuracy > 0,88
        if accuracy > 0.88:
            save_model(model, model_path)
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "Model")

            # Asignar el modelo a la etapa "Production"
            client = MlflowClient()
            client.transition_model_version_stage(name="Model", version=1, stage="Production")
            notify_api_task(api_url="http://localhost:8000/update_model")
        else:
            raise Exception("Rendimiento del modelo por debajo de métrica necesaria.")


if __name__ == "__main__":
    train_pipeline(
        source= "csv",
        file_path= "/home/tomibianco/appml/data/data.csv"

        # source="db",
        # connection_string="postgresql://usuario:contraseña@localhost:5432/mi_base_de_datos",
        # query="""
        #     SELECT variable_1, variable_2
        #     FROM tabla
        #     WHERE intervalo_temporal
        # """
    )

    train_pipeline.from_source(
        source="https://github.com/tomibianco/app_ml_1.git",
        entrypoint="/home/tomibianco/appml/backend/deployment_pipeline.py:train_pipeline"
    ).deploy(
        name="Scheduled Training Pipeline",
        work_pool_name="my-managed-pool",
    )