import pandas as pd
import logging
import pandera as pa
from pandera.errors import SchemaErrors

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

    # Manejo de outliers identificados en el EDA (capping)
    # Se definen umbrales razonables para acotar los valores extremos.
    bmi_cap = 55
    charges_cap = 52000
    cleaned_data["bmi"] = cleaned_data["bmi"].clip(upper=bmi_cap)
    cleaned_data["charges"] = cleaned_data["charges"].clip(upper=charges_cap)
    log.info(
        f"Outliers acotados: 'bmi' limitado a {bmi_cap} y 'charges' a {charges_cap}."
    )

    # Definir el esquema de validación con Pandera
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(int, pa.Check.between(min_value=18, max_value=100)),
            "sex": pa.Column(str, pa.Check.isin(["male", "female"])),
            "bmi": pa.Column(float, pa.Check.between(min_value=10, max_value=bmi_cap)),
            "children": pa.Column(int, pa.Check.ge(0)),
            "smoker": pa.Column(str, pa.Check.isin(["yes", "no"])),
            "region": pa.Column(str),
            "charges": pa.Column(float, nullable=False),
        }
    )

    try:
        log.info("Validando datos con el esquema de Pandera...")
        cleaned_data = schema.validate(cleaned_data, lazy=True)
        log.info("Validación de datos con Pandera exitosa.")
    except SchemaErrors as err:
        log.error("Falló la validación de datos con Pandera. Revisar los errores.")
        log.error(err.failure_cases)  # Muestra los datos que fallaron
        raise  # Vuelve a lanzar la excepción para detener el pipeline

    # Convertir a tipos 'category' después de la validación para optimizar memoria
    cleaned_data = cleaned_data.astype(
        {"sex": "category", "smoker": "category", "region": "category"}
    )
    return cleaned_data
