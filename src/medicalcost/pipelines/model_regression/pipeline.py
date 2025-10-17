from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs="raw_medical_data_csv",
                outputs="model_input_data",
                name="preprocess_medical_data_node",
            ),
            node(
                func=train_model,
                inputs="model_input_data",
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
