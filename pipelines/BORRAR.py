import json
import os
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect_docker import DockerContainer

# Simulación de módulos personalizados
from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from utils import get_data_for_test


# --- Tareas de Prefect ---
@task
def dynamic_importer() -> str:
    """Descarga los últimos datos desde una API simulada."""
    data = get_data_for_test()
    return data


@task
def deployment_trigger(accuracy: float, min_accuracy: float = 0.9) -> bool:
    """Decide si el modelo debe desplegarse basado en la precisión."""
    return accuracy > min_accuracy


@task
def load_prediction_service(pipeline_name: str, step_name: str, running: bool = True):
    """
    Simula la carga de un servicio de predicción. 
    Aquí puedes implementar la lógica para interactuar con MLFlow o cualquier otro servicio.
    """
    # Aquí interactuarías con el MLFlow Model Deployer, simulado para esta implementación.
    if running:
        return f"Servicio {pipeline_name}-{step_name} está corriendo."
    else:
        raise RuntimeError(f"No se encontró un servicio activo para {pipeline_name}-{step_name}")


@task
def predictor(service, data: str) -> np.ndarray:
    """Realiza predicciones usando el servicio desplegado."""
    # Simulación de inicio del servicio
    print(f"Iniciando servicio: {service}")
    
    # Procesamiento de datos
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)
    
    # Simulación de predicción (reemplaza con una API real)
    prediction = np.random.rand(data_array.shape[0])  # Simulación
    print(f"Predicción: {prediction}")
    return prediction


# --- Flujos (Pipelines) con Prefect ---
@flow
def training_pipeline(min_accuracy: float = 0.9):
    """
    Pipeline de entrenamiento, evaluación y despliegue.
    """
    # Ingesta de datos
    df = ingest_data()
    
    # Limpieza de datos
    x_train, x_test, y_train, y_test = clean_data(df)
    
    # Entrenamiento del modelo
    model = train_model(x_train, x_test, y_train, y_test)
    
    # Evaluación del modelo
    mse, rmse = evaluation(model, x_test, y_test)
    
    # Decisión de despliegue
    deploy_decision = deployment_trigger(mse, min_accuracy=min_accuracy).result()
    if deploy_decision:
        print("Modelo aprobado para despliegue.")
        # Aquí puedes añadir la lógica para desplegar en MLFlow u otro servicio.
    else:
        print("Modelo no cumple con los criterios para despliegue.")


@flow
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    """
    Pipeline de inferencia.
    """
    # Importar datos de prueba dinámicamente
    batch_data = dynamic_importer()
    
    # Cargar el servicio de predicción
    model_deployment_service = load_prediction_service(
        pipeline_name=pipeline_name,
        step_name=pipeline_step_name,
        running=True,
    )
    
    # Realizar predicciones
    predictions = predictor(service=model_deployment_service, data=batch_data)
    print(f"Resultados de las predicciones: {predictions}")


# --- Ejecución ---
if __name__ == "__main__":
    # Ejecución del pipeline de entrenamiento
    print("Ejecutando pipeline de entrenamiento...")
    training_pipeline(min_accuracy=0.9)

    # Ejecución del pipeline de inferencia
    print("Ejecutando pipeline de inferencia...")
    inference_pipeline(pipeline_name="training_pipeline", pipeline_step_name="predictor")