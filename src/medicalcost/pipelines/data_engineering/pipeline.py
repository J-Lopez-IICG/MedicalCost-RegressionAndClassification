from kedro.pipeline import Pipeline, node, pipeline
from .nodes import download_and_load_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de ingeniería de datos.

    Este pipeline contiene un único nodo que se encarga de descargar los datos
    crudos desde Kaggle y cargarlos en un DataFrame.

    Returns:
        Un objeto Pipeline que define el flujo de trabajo de ingeniería de datos.
    """
    return pipeline(
        [
            node(
                func=download_and_load_raw_data,
                # 'inputs' toma los parámetros definidos en 'conf/base/parameters.yml'
                # bajo la sección 'data_engineering'.
                inputs="params:data_engineering",
                # 'outputs' es el nombre del dataset que se guardará. Kedro lo mapea
                # al archivo definido en 'conf/base/catalog.yml'.
                outputs="raw_medical_cost_data",
                name="download_and_load_raw_data_node",
            )
        ]
    )
