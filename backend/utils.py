import io
import mlflow
import pandas as pd
from config import label_mapping, columns_train
from fastapi import HTTPException, UploadFile


def load_production_model(model_name="Model", alias="champion"):
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")

def validate_csv(file: UploadFile):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="El archivo debe ser de tipo CSV.")

def load_csv(file: UploadFile):
    try:
        content = file.file.read()
        csv_content = io.StringIO(content.decode("utf-8"))
        df = pd.read_csv(csv_content)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {str(e)}")

def preprocess_dataframe(df, columns_to_drop, columns_to_encode):
    try:
        df_cleaned = df.drop(columns=columns_to_drop)
        df_encoded = pd.get_dummies(df_cleaned, columns=columns_to_encode)
        for col in columns_train:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[columns_train]
        return df_encoded
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al preprocesar el DataFrame: {str(e)}")

def make_predictions(model, df_encoded):
    try:
        predictions = model.predict(df_encoded)
        pred_labels = [label_mapping[pred] for pred in predictions]
        return pred_labels
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar predicciones: {str(e)}")

def package_predictions(original_data, predictions):
    output = []
    for original_row, pred_label in zip(original_data, predictions):
        output.append({"input": original_row, "prediction": pred_label})
    return output