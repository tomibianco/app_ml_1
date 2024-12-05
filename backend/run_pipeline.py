from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import model_train


if __name__ == "__main__":
    training = train_pipeline(
        ingest_data=ingest_data,
        clean_data=clean_data,
        model_train=model_train,
        evaluation=evaluation,

        source="csv",
        file_path="./data/data.csv",
        
        # source="db",
        # connection_string="postgresql://usuario:contraseña@localhost:5432/mi_base_de_datos",
        # query="""
        #     SELECT variable_1, variable_2
        #     FROM tabla
        #     WHERE intervalo_temporal
        # """
    )

    training.run()