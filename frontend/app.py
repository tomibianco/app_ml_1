import streamlit as st
import requests
import io

# URL de la API FastAPI
API_URL = 'http://localhost:8000/predict'  # Cambia esta URL por la de tu backend

# Título de la aplicación
st.title("Predicción de Modelo ML con FastAPI")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Mostrar botón para procesar el archivo
    if st.button('Generar Predicción'):
        # Enviar el archivo al backend
        response = requests.post(
            API_URL,
            files={"file": ("uploaded_file.csv", uploaded_file.getvalue(), 'text/csv')}
        )

        # Manejar la respuesta del backend
        if response.status_code == 200:
            result_json = response.json()

            # Mostrar los resultados en JSON
            st.subheader("Resultado de la Predicción (JSON):")
            st.json(result_json.get('predictions', {}))

            # Permitir descarga del CSV si está disponible
            csv_file = result_json.get('csv_file', None)
            if csv_file:
                st.subheader("Descargar archivo CSV con predicciones:")
                st.download_button(
                    label="Descargar CSV",
                    data=csv_file,
                    file_name="predicciones.csv",
                    mime="text/csv"
                )
        else:
            st.error(f"Error en la API: {response.status_code}, {response.text}")
