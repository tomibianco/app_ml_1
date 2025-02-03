import logging
import pandas as pd
from langchain_ollama import ChatOllama

# Configuración del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info("CSV cargado correctamente.")
        return df
    except Exception as e:
        logging.error(f"Error al cargar el CSV: {e}")
        raise

def preprocess_data(df):
    # Limpieza y validación de datos
    df_clean = df.dropna()
    logging.info("Datos preprocesados correctamente.")
    return df_clean

def build_prompt(df, additional_query):
    data_str = df.to_string(index=False)
    prompt = f"""
Tengo los siguientes datos de ventas:

{data_str}

{additional_query}
"""
    return prompt

def main():
    filepath = "/home/tomibianco/appml/pruebas-llm/libro.csv"
    query_text = "Dime cuantos trabajadores tienen más de 5 ventas"
    
    df = load_data(filepath)
    df = preprocess_data(df)
    
    prompt = build_prompt(df, query_text)
    
    llm = ChatOllama(model="deepseek-r1:8b")
    
    try:
        response = llm.invoke(prompt)
        logging.info("Respuesta obtenida del modelo.")
        print(response)
    except Exception as e:
        logging.error(f"Error durante la invocación del modelo: {e}")

if __name__ == "__main__":
    main()