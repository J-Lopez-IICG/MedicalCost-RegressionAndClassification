from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_and_validate_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de procesamiento de datos.

    Este pipeline toma los datos crudos y realiza la limpieza básica
    (elimina nulos, duplicados y ajusta tipos de datos).

    Returns:
        Un objeto Pipeline que define el flujo de trabajo de procesamiento de datos.
    """
    return pipeline(
        [
            node(
                func=clean_and_validate_data,
                # Toma los datos crudos como entrada.
                inputs="raw_medical_cost_data",
                # Genera un único dataset intermedio y limpio.
                outputs="processed_medical_data",
                name="clean_and_validate_data_node",
            )
        ]
    )
