from langchain_ollama import ChatOllama
import pandas as pd

# Cargar el CSV en un DataFrame
df = pd.read_csv("/home/tomibianco/appml/pruebas-llm/libro.csv")

# Cargar el modelo de DeepSeek desde Ollama
llm = ChatOllama(model="deepseek-r1:8b")

# Convertimos el dataframe en un string estructurado para el modelo
data_str = df.to_string(index=False)

# Definimos la consulta
query = f"""
Tengo los siguientes datos de ventas:

{data_str}

Dime cuantos trabajadores tienen más de 5 ventas
"""

# Ejecutamos la consulta en DeepSeek
response = llm.invoke(query)

# Mostramos la respuesta
return response