from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de regresión.

    Este pipeline toma los datos preparados, entrena un modelo de regresión lineal
    para predecir los costos del seguro, y luego lo evalúa para generar
    métricas de rendimiento.

    Returns:
        Un objeto Pipeline que define el flujo de trabajo de regresión.
    """
    return pipeline(
        [
            node(
                # Divide los datos y entrena el modelo de regresión lineal.
                func=train_model,
                inputs={
                    "primary_medical_data": "primary_medical_data",
                    "parameters": "params:model_regression",
                },
                outputs=["reg_model", "reg_X_test", "reg_y_test", "y_pred", "X"],
                name="train_linear_regression_model_node",
            ),
            node(
                # Evalúa el modelo calculando R-cuadrado y extrayendo coeficientes.
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
