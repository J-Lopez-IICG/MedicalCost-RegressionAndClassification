import pandas as pd
import logging

log = logging.getLogger(__name__)


def clean_and_validate_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Limpia y valida el DataFrame de costos médicos.

    Esta función realiza las siguientes operaciones en secuencia:
    1. Elimina filas con valores nulos/faltantes.
    2. Elimina filas duplicadas.
    3. Asegura que los tipos de datos de las columnas categóricas sean correctos.

    Args:
        df: DataFrame de entrada con los datos de costos médicos crudos.

    Returns:
        Un único DataFrame limpio y validado.
    """
    log.info(f"Iniciando limpieza. Forma inicial del DataFrame: {df.shape}")
    df_clean = df.copy()

    # --- 1. Eliminar filas con datos faltantes ---
    initial_rows = len(df_clean)
    df_clean.dropna(inplace=True)
    rows_after_na = len(df_clean)
    if initial_rows > rows_after_na:
        log.info(
            f"Se eliminaron {initial_rows - rows_after_na} filas con valores nulos."
        )
    else:
        log.info("No se encontraron valores nulos en el dataset.")

    # --- 2. Eliminar filas duplicadas ---
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    rows_after_duplicates = len(df_clean)
    if initial_rows > rows_after_duplicates:
        log.info(
            f"Se eliminaron {initial_rows - rows_after_duplicates} filas duplicadas."
        )
    else:
        log.info("No se encontraron filas duplicadas.")

    # --- 3. Asegurar tipos de datos ---
    for col in ["sex", "smoker", "region"]:
        df_clean[col] = df_clean[col].astype("category")
    log.info("Tipos de datos de columnas categóricas asegurados.")

    log.info(f"Limpieza finalizada. Forma final del DataFrame: {df_clean.shape}")
    return df_clean
