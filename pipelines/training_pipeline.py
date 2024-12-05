from prefect import flow


@flow
def train_pipeline(ingest_data, clean_data, model_train, evaluation, source: str, **kwargs):
    """
    Pipeline de entrenamiento de modelo.
    
    Args:
        source: Indica la fuente de datos ("csv" o "db").
        kwargs: Argumentos adicionales según la fuente de datos.
            - Si source == "csv": kwargs debe incluir "file_path".
            - Si source == "db": kwargs debe incluir "connection_string" y "query".
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    
    Devuelve:
        accuracy: float
        precision: float
        recall: float
        f1_score: float
    """
    df = ingest_data(source=source, **kwargs)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = model_train(X_train, X_test, y_train, y_test)
    accuracy, precision, recall, f1_score = evaluation(model, X_test, y_test)