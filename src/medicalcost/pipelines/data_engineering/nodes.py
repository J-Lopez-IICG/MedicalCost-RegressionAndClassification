from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def download_and_load_raw_data(parameters: dict) -> pd.DataFrame:
    """
    Descarga el dataset desde Kaggle usando la API oficial, lo descomprime,
    lee el archivo CSV y lo devuelve como un DataFrame de pandas,
    limpiando los archivos temporales.

    Args:
        parameters: Diccionario con los parámetros del pipeline.
    """
    dataset_handle = parameters["dataset_handle"]
    temp_download_path = Path(parameters["temp_download_path"])

    try:
        # 1. Crear carpeta temporal y autenticar
        log.info(f"Creando directorio temporal en: {temp_download_path}")
        temp_download_path.mkdir(parents=True, exist_ok=True)

        log.info("Autenticando con la API de Kaggle...")
        api = KaggleApi()
        api.authenticate()

        # 2. Descargar y descomprimir el dataset
        log.info(f"Descargando y descomprimiendo '{dataset_handle}'...")
        api.dataset_download_files(dataset_handle, path=temp_download_path, unzip=True)
        log.info("Dataset descargado y descomprimido.")

        # 3. Encontrar el archivo CSV descomprimido
        csv_files = list(temp_download_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                "No se encontró ningún archivo CSV en el dataset descargado."
            )
        csv_file_path = csv_files[0]  # Asumimos que solo hay un CSV

        # 4. Leer el CSV y devolver el DataFrame
        log.info(f"Leyendo el archivo CSV desde: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        log.info("DataFrame creado. Kedro procederá a guardarlo.")
        return df

    finally:
        # 5. Limpiar la carpeta temporal
        if temp_download_path.exists():
            log.info(f"Limpiando directorio temporal: {temp_download_path}")
            shutil.rmtree(temp_download_path)
