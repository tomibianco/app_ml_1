# Aplicación para ...

Este proyecto es una aplicación de Machine Learning para predecir fallas de maquinaria en la minería de litio. Utiliza un modelo entrenado con XGBoost y una API desarrollada en FastAPI, empaquetada con Docker, y desplegable en Amazon Elastic Beanstalk.

## Tabla de Contenidos
- [Introducción](#introducción)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [API Endpoints](#api-endpoints)
- [Despliegue en Elastic Beanstalk](#despliegue-en-elastic-beanstalk)

---

## Introducción

Este proyecto tiene como objetivo minimizar el tiempo de inactividad y optimizar el mantenimiento de maquinaria minera mediante la predicción de fallas. La aplicación está diseñada para recibir datos en tiempo real de diversas máquinas, procesarlos y retornar una predicción sobre posibles fallas futuras. Esto permite a las empresas tomar decisiones informadas y programar mantenimientos preventivos.

## Tecnologías Utilizadas

- **Python**: Lenguaje principal de programación.
- **XGBoost**: Biblioteca para la construcción del modelo de predicción.
- **FastAPI**: Framework para el desarrollo de la API RESTful.
- **Docker**: Contenerización de la aplicación para facilitar su despliegue y portabilidad.
- **Elastic Beanstalk (AWS)**: Plataforma de despliegue para ejecutar aplicaciones y servicios.
- **Git**: Control de versiones.

## Estructura del Proyecto

├── app/
│   ├── main.py                 # Archivo principal de la API
│   ├── models/
│   │   └── model.pkl           # Modelo XGBoost pre-entrenado
│   ├── schemas.py              # Esquemas de datos para las solicitudes y respuestas
│   └── utils.py                # Funciones de utilidad para preprocesamiento
├── Dockerfile                  # Dockerfile para contenerización de la aplicación
├── requirements.txt            # Lista de dependencias del proyecto
└── README.md                   # Documentación del proyecto

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Docker** (para crear el contenedor)
- **Git** (para clonar el repositorio)
- **AWS CLI** (para el despliegue en Elastic Beanstalk, si aplica)

## Instalación

### 1. Clonar el Repositorio

```bash
git clone <URL-DEL-REPOSITORIO>
cd <NOMBRE-DEL-PROYECTO>
```

### 2. Crear un Entorno Virtual e Instalar Dependencias

Si deseas ejecutar la aplicación sin Docker, crea un entorno virtual e instala las dependencias:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Construir y Ejecutar con Docker

Para ejecutar la aplicación dentro de un contenedor Docker, primero asegúrate de que Docker esté instalado y ejecutándose en tu máquina.

```bash
docker build -t prediccion-fallas-maquinaria .
docker run -d -p 8000:8000 prediccion-fallas-maquinaria
Esto desplegará la API en http://localhost:8000.
```

## #Uso

### Realizar una Predicción

Para realizar una predicción, en /docs, enviar una solicitud POST al endpoint /predict con los datos de la maquinaria en formato JSON. La API responderá con una predicción de falla basada en el modelo.

## API Endpoints

POST /predict
Descripción: Predice la probabilidad de falla para una máquina.

Solicitud: ...

Respuesta: ...

## Despliegue en Elastic Beanstalk

### Paso 1: Configuración Inicial

1. Instala la CLI de Elastic Beanstalk:

```bash
pip install awsebcli
```

2. Configura las credenciales de AWS:

```bash
aws configure
```

### Paso 2: Crear un Entorno y Desplegar

Dentro del directorio de tu proyecto, ejecuta los siguientes comandos:

```bash
eb init -p docker prediccion-fallas-maquinaria
eb create prediccion-fallas-env
eb open
```

Estos comandos crearán y abrirán un entorno en Elastic Beanstalk donde tu aplicación estará desplegada.