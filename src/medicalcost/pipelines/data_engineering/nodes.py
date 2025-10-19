from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def download_and_load_raw_data(parameters: dict) -> pd.DataFrame:
    """
    Descarga un conjunto de datos desde Kaggle, lo carga en un DataFrame de pandas y limpia los archivos temporales.

    Este nodo realiza las siguientes acciones:
    1. Se conecta a la API de Kaggle usando las credenciales del usuario.
    2. Descarga y descomprime el dataset especificado en los parámetros.
    3. Busca el primer archivo .csv dentro de la carpeta de descarga.
    4. Carga el archivo CSV en un DataFrame de pandas.
    5. Elimina la carpeta temporal de descarga para no dejar archivos residuales.

    Args:
        parameters: Un diccionario que contiene:
            - dataset_handle: El identificador del dataset en Kaggle (ej: 'usuario/nombre-dataset').
            - temp_download_path: La ruta a una carpeta temporal para la descarga.

    Returns:
        Un DataFrame de pandas con los datos crudos del archivo CSV. Kedro se encargará
        de guardar este DataFrame en la ubicación definida en el catálogo.
    """
    # Extrae los parámetros necesarios del archivo `parameters.yml`.
    dataset_handle = parameters["dataset_handle"]
    temp_download_path = Path(parameters["temp_download_path"])

    # Se utiliza un bloque `try...finally` para asegurar que la limpieza
    # del directorio temporal se ejecute siempre, incluso si ocurre un error.
    try:
        # Paso 1: Crear el directorio temporal donde se descargarán los archivos.
        log.info(f"Creando directorio temporal en: {temp_download_path}")
        temp_download_path.mkdir(parents=True, exist_ok=True)

        # Paso 2: Autenticarse en la API de Kaggle.
        # Esto requiere que el archivo `kaggle.json` esté en la ubicación correcta.
        log.info("Autenticando con la API de Kaggle...")
        api = KaggleApi()
        api.authenticate()

        # Paso 3: Descargar los archivos del dataset y descomprimirlos en la carpeta temporal.
        log.info(f"Descargando y descomprimiendo '{dataset_handle}'...")
        api.dataset_download_files(dataset_handle, path=temp_download_path, unzip=True)
        log.info("Dataset descargado y descomprimido.")

        # Paso 4: Buscar el archivo CSV dentro de la carpeta descomprimida.
        csv_files = list(temp_download_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                "No se encontró ningún archivo CSV en el dataset descargado."
            )
        csv_file_path = csv_files[
            0
        ]  # Asumimos que el primer CSV encontrado es el correcto.

        # Paso 5: Leer el archivo CSV y cargarlo en un DataFrame de pandas.
        log.info(f"Leyendo el archivo CSV desde: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        log.info("DataFrame creado. Kedro procederá a guardarlo.")
        return df

    finally:
        # Paso final: Limpiar la carpeta temporal para no dejar archivos residuales.
        if temp_download_path.exists():
            log.info(f"Limpiando directorio temporal: {temp_download_path}")
            shutil.rmtree(temp_download_path)
