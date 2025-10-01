from setuptools import setup, find_packages

setup(
    name="cardioPredict",                # Nombre del paquete/proyecto
    version="0.1.0",                   # Versión inicial
    description="Utilidades para el proyecto (ej: data_utils).",
    author="Federico Trujillo",
    packages=find_packages(where="src"),   # Busca los paquetes dentro de /src
    package_dir={"": "src"},               # Define que src/ es la carpeta raíz de los paquetes
)
