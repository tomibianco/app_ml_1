class ModelNameConfig:
    """
    Configuración para elegir el modelo.

    Se puede elegir:
                    xgboost
                    randomforest
    """
    def __init__(self, model_name: str):
        self.model_name = "xgboost"