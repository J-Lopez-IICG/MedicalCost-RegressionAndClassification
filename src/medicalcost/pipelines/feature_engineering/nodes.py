import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


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

    # 1. No se eliminan outliers, ya que los valores extremos (ej. fumadores con
    # alto IMC) contienen información predictiva crucial para los modelos.

    # 2. Calcular la mediana de 'charges' y crear la variable objetivo 'cost_category'
    charges_median = df_prepared["charges"].median()
    log.info(f"La mediana de 'charges' calculada es: {charges_median:.2f}")
    cost_category_str = np.where(
        df_prepared["charges"] > charges_median, "Alto", "Bajo"
    )
    df_prepared["cost_category"] = pd.Series(cost_category_str).map(
        {"Bajo": 0, "Alto": 1}
    )
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
