from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs={
                    "primary_medical_data": "primary_medical_data",
                    "parameters": "params:model_regression",
                },
                outputs=["reg_model", "reg_X_test", "reg_y_test", "y_pred", "X"],
                name="train_linear_regression_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["reg_model", "reg_X_test", "reg_y_test", "y_pred", "X"],
                outputs=[
                    "r2_score_output",
                    "model_coefficients",
                    "model_evaluation_output",
                ],
                name="evaluate_model_node",
            ),
        ]
    )
