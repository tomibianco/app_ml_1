import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Predictor CSV", layout="centered")

# Título de la aplicación
st.title("Predictor de CSV con Machine Learning")

# Subir archivo CSV
st.header("Sube tu archivo CSV")
uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

# Botón para enviar el archivo al backend
if uploaded_file:
    st.info("El archivo será procesado para generar predicciones.")
    
    # Botón de predicción
    if st.button("Procesar y Obtener Predicciones"):
        with st.spinner("Procesando el archivo..."):
            try:
                # Leer el contenido del archivo como binario
                file_bytes = uploaded_file.read()
                
                # Preparar los datos para enviar al backend
                files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
                
                # Enviar la solicitud POST al backend
                response = requests.post(
                    "http://localhost:8000/predict_input_download_csv",  # Cambia por tu URL si corresponde
                    files=files
                )

                if response.status_code == 200:
                    # Guardar y descargar CSV
                    st.success("Predicciones generadas exitosamente.")
                    
                    # Botón para descargar el archivo procesado
                    st.download_button(
                        label="Descargar CSV con Predicciones",
                        data=response.content,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                else:
                    # Mostrar mensaje de error si ocurre algún problema
                    st.error(f"Error: {response.json().get('detail', 'Error desconocido')}")

            except Exception as e:
                st.error(f"Error al conectarse con el backend: {str(e)}")