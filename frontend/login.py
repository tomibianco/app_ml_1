import streamlit as st
from streamlit_option_menu import option_menu
import requests
import os

# URL del backend (modifica según tu servidor)
API_URL = "http://127.0.0.1:8000/login"

# Ruta a la carpeta de imágenes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "static", "logo.png")

# Configurar página de Streamlit (título y ancho)
st.set_page_config(page_title="Login - Aplicación ML", layout="centered")

# Inicializar estado de sesión
if "login_status" not in st.session_state:
    st.session_state.login_status = False
if "username" not in st.session_state:
    st.session_state.username = None

# Función para pantalla de login
def login_screen():
    """Pantalla de inicio de sesión"""
    st.image(LOGO_PATH, width=120, use_container_width=True)
    st.markdown("#### **Inicia Sesión**")
    st.text_input("Usuario", key="login_user", placeholder="Ingresa tu usuario")
    st.text_input("Contraseña", type="password", key="login_password", placeholder="Ingresa tu contraseña")

    if st.button("Iniciar sesión", use_container_width=True):
        # Obtener datos del formulario
        username = st.session_state.login_user
        password = st.session_state.login_password
        if not username or not password:
            st.error("Por favor, completa ambos campos.")
            return

        # Enviar solicitud al backend
        try:
            response = requests.post(API_URL, json={"username": username, "password": password})
            if response.status_code == 200:
                st.session_state.login_status = True
                st.session_state.username = username
                st.success("Inicio de sesión exitoso. Redirigiendo...")
            else:
                st.error("Usuario o contraseña incorrectos.")
        except Exception as e:
            st.error(f"Error de conexión con el servidor: {e}")

# Función para contenido después del login
def main_app():
    """Contenido principal de la aplicación"""
    st.title(f"Bienvenid@, {st.session_state.username} 👋")
    st.markdown("#### **Dashboard Principal**")
    with st.sidebar:
        menu = option_menu(
            menu_title="Menú Principal",
            options=["Inicio", "Predicciones", "Configuración", "Cerrar sesión"],
            icons=["house", "bar-chart-line", "gear", "box-arrow-left"],
            menu_icon="list",
            default_index=0,
        )

    # Navegación del menú
    if menu == "Inicio":
        st.subheader("📊 Resumen general")
        st.write("Aquí puedes ver métricas clave y visualizaciones importantes.")
    elif menu == "Predicciones":
        st.subheader("🤖 Predicciones de ML")
        st.write("Sube datos o revisa resultados de predicciones.")
    elif menu == "Configuración":
        st.subheader("⚙️ Configuración")
        st.write("Personaliza las opciones de tu aplicación.")
    elif menu == "Cerrar sesión":
        st.session_state.login_status = False
        st.experimental_rerun()

# Lógica de navegación
if not st.session_state.login_status:
    login_screen()
else:
    main_app()