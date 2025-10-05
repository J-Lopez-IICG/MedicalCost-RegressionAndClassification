from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_and_validate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_and_validate_data,
                inputs="raw_medical_data_csv",
                outputs="processed_medical_data_excel",
                name="clean_and_validate_data_node",
            )
        ]
    )
