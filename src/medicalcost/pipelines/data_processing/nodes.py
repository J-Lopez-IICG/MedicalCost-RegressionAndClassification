import pandas as pd
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)


def clean_and_validate_data(
    df: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Limpia y valida el DataFrame de tweets.

    Esta función realiza las siguientes operaciones:
    1. Elimina filas donde 'TweetContent' es nulo.
    2. Mapea sentimientos (ej. 'Irrelevant' a 'Neutral') según los parámetros.
    3. Elimina filas duplicadas basadas en un subconjunto de columnas.
    4. Filtra el DataFrame para mantener solo los sentimientos válidos.

    Args:
        df: DataFrame de entrada con los datos de tweets.
        parameters: Diccionario de parámetros que contiene:
            - sentiment_mapping: Un mapa para reemplazar valores de sentimiento.
            - subset_for_duplicates: Columnas para identificar duplicados.
            - valid_sentiments: Una lista de sentimientos a conservar.

    Returns:
        Un DataFrame limpio y validado.
    """
    log.info(f"Iniciando limpieza. Forma inicial del DataFrame: {df.shape}")

    # 1. Eliminar filas con 'TweetContent' nulo
    initial_rows = len(df)
    df.dropna(subset=["TweetContent"], inplace=True)
    rows_after_na = len(df)
    log.info(
        f"Se eliminaron {initial_rows - rows_after_na} filas con 'TweetContent' nulo."
    )

    # 2. Mapear 'Irrelevant' a 'Neutral' usando el parámetro
    sentiment_map = parameters.get("sentiment_mapping", {})
    if sentiment_map:
        df["Sentiment"] = df["Sentiment"].replace(sentiment_map)
        log.info(f"Sentimientos mapeados usando la configuración: {sentiment_map}")

    # 3. Eliminar duplicados
    initial_rows = len(df)
    subset_cols = parameters["subset_for_duplicates"]
    df.drop_duplicates(subset=subset_cols, inplace=True)
    rows_after_duplicates = len(df)
    log.info(f"Se eliminaron {initial_rows - rows_after_duplicates} filas duplicadas.")

    # 4. Filtrar por sentimientos válidos
    initial_rows = len(df)
    valid_sentiments = parameters["valid_sentiments"]
    df = df[df["Sentiment"].isin(valid_sentiments)].copy()
    rows_after_filter = len(df)
    log.info(
        f"Se eliminaron {initial_rows - rows_after_filter} filas con sentimientos no válidos."
    )

    log.info(f"Limpieza finalizada. Forma final del DataFrame: {df.shape}")

    return df
