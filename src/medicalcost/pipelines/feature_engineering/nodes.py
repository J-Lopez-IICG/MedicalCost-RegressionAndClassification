import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


def _remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Función auxiliar para eliminar outliers de una columna usando el método IQR.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    initial_rows = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_rows = initial_rows - len(df_filtered)
    if removed_rows > 0:
        log.info(f"Se eliminaron {removed_rows} outliers de la columna '{column}'.")
    return df_filtered


def prepare_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelado: elimina outliers, crea la variable
    objetivo para clasificación y genera variables dummy.

    Args:
        df: DataFrame limpio proveniente del pipeline `data_processing`.

    Returns:
        DataFrame preparado para ser consumido por los pipelines de modelado.
    """
    log.info(f"Iniciando preparación de características. Forma inicial: {df.shape}")
    df_prepared = df.copy()

    # 1. Eliminar outliers de 'bmi' y 'charges'
    df_prepared = _remove_outliers(df_prepared, "bmi")
    df_prepared = _remove_outliers(df_prepared, "charges")
    log.info(f"Forma después de eliminar outliers: {df_prepared.shape}")

    # 2. Calcular la mediana de 'charges' (después de quitar outliers) y crear 'cost_category'
    charges_median = df_prepared["charges"].median()
    log.info(f"La mediana de 'charges' calculada es: {charges_median:.2f}")
    df_prepared["cost_category"] = np.where(
        df_prepared["charges"] > charges_median, "Alto", "Bajo"
    )
    df_prepared["cost_category"] = df_prepared["cost_category"].astype("category")
    log.info("Columna 'cost_category' creada.")

    # 3. Crear variables dummy para las columnas categóricas
    log.info("Creando variables dummy para 'sex', 'smoker' y 'region'.")
    df_prepared = pd.get_dummies(
        df_prepared, columns=["sex", "smoker", "region"], drop_first=True
    )
    log.info(
        f"Preparación de características finalizada. Forma final: {df_prepared.shape}"
    )

    return df_prepared
