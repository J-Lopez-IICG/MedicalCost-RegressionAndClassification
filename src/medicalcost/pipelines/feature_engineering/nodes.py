import pandas as pd
import logging

log = logging.getLogger(__name__)


def create_dummy_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas categóricas en variables dummy usando one-hot encoding.

    Args:
        df: DataFrame procesado.

    Returns:
        DataFrame con las variables dummy añadidas.
    """
    log.info("Creando variables dummy para 'sex', 'smoker', y 'region'.")
    df_featured = pd.get_dummies(
        df, columns=["sex", "smoker", "region"], drop_first=True
    )
    return df_featured
