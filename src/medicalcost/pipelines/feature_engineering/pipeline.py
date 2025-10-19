from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_features_and_target


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de ingeniería de características.

    Este pipeline toma los datos limpios, elimina outliers, crea la variable
    objetivo para clasificación y genera variables dummy para el modelado.

    Returns:
        Un objeto Pipeline que define el flujo de trabajo de ingeniería de características.
    """
    return pipeline(
        [
            node(
                func=prepare_features_and_target,
                # Toma los datos limpios del pipeline anterior.
                inputs="processed_medical_data",
                # Genera el dataset primario listo para los modelos.
                outputs="primary_medical_data",
                name="prepare_features_node",
            )
        ]
    )
