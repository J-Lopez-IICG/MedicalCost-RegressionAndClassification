from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_dummy_variables


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_dummy_variables,
                inputs="processed_medical_data_excel",
                outputs="featured_medical_data",
                name="create_dummy_variables_node",
            )
        ]
    )
