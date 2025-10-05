import pandas as pd
import logging

log = logging.getLogger(__name__)


def clean_and_validate_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos eliminando filas con valores nulos y asegura
    que cada columna tenga el tipo de dato correcto.

    Args:
        raw_data: DataFrame con los datos crudos.

    Returns:
        DataFrame limpio y validado.
    """
    # Eliminar filas con valores nulos
    log.info(f"Tamaño original del dataset: {raw_data.shape[0]} filas.")
    cleaned_data = raw_data.dropna().copy()
    log.info(
        f"Tamaño del dataset después de eliminar nulos: {cleaned_data.shape[0]} filas."
    )

    if cleaned_data.shape[0] < raw_data.shape[0]:
        log.warning(
            f"Se eliminaron {raw_data.shape[0] - cleaned_data.shape[0]} filas con valores nulos."
        )

    # Definir los tipos de datos esperados
    expected_types = {
        "age": "int64",
        "sex": "category",
        "bmi": "float64",
        "children": "int64",
        "smoker": "category",
        "region": "category",
        "charges": "float64",
    }

    # Filtrar el diccionario para solo incluir columnas que existen en el DataFrame
    # y advertir sobre las que no se encuentren.
    actual_types_to_convert = {}
    for col, dtype in expected_types.items():
        if col in cleaned_data.columns:
            actual_types_to_convert[col] = dtype
        else:
            log.warning(
                f"Columna '{col}' para conversión de tipo no encontrada en el dataset."
            )

    # Convertir todos los tipos de datos de una sola vez
    log.info("Convirtiendo tipos de datos de las columnas...")
    cleaned_data = cleaned_data.astype(actual_types_to_convert)
    log.info("Tipos de datos validados correctamente.")

    return cleaned_data
