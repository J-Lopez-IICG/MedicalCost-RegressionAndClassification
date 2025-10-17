from kedro.pipeline import Pipeline, node, pipeline
from .nodes import download_and_load_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_and_load_raw_data,
                inputs="params:data_engineering",
                outputs="raw_medical_data_csv",
                name="download_and_load_raw_data_node",
            )
        ]
    )
