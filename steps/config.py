class ModelNameConfig:
    """
    Configuración para elegir el modelo.

    Se puede elegir:
                    xgboost
                    randomforest
    """
    def __init__(self, model_name: str = "lightgbm"):
        self.model_name = model_name