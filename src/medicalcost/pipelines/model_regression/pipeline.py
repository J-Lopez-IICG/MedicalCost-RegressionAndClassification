from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_data,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    predict,
    evaluate_model,
    plot_feature_correlation_heatmap,
    plot_r2_comparison,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de regresión para entrenar y comparar múltiples modelos.

    Este pipeline toma los datos preparados, entrena modelos de Regresión Lineal,
    Random Forest y XGBoost, y luego los evalúa para generar métricas de rendimiento.

    Returns:
        Un objeto Pipeline que define el flujo de trabajo de regresión.
    """
    data_preparation_pipeline = pipeline(
        [
            node(
                # Genera un mapa de calor de correlación para las características finales.
                func=plot_feature_correlation_heatmap,
                inputs="primary_medical_data",
                outputs="regression_feature_correlation_heatmap",
                name="plot_regression_feature_correlation_node",
            ),
            node(
                # Divide los datos en conjuntos de entrenamiento y prueba.
                func=split_data,
                inputs={
                    "primary_medical_data": "primary_medical_data",
                    "parameters": "params:model_regression",
                },
                outputs=["X_train", "X_test", "y_train", "y_test", "X_reference"],
                name="split_data_node",
            ),
        ]
    )

    # --- Pipeline para Regresión Lineal ---
    linear_regression_pipeline = pipeline(
        [
            node(
                func=train_linear_regression,
                inputs=["X_train", "y_train"],
                outputs="reg_model_lr",
                name="reg_train_linear_regression_node",
            ),
            node(
                func=predict,
                inputs=["reg_model_lr", "X_test"],
                outputs="reg_y_pred_lr",
                name="reg_predict_lr_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "reg_model_lr",
                    "X_train",
                    "y_train",
                    "y_test",
                    "reg_y_pred_lr",
                    "X_reference",
                ],
                outputs=[
                    "r2_score_lr",
                    "metrics_lr",
                    "evaluation_output_lr",
                ],
                name="reg_evaluate_lr_node",
            ),
        ]
    )

    # --- Pipeline para Random Forest ---
    random_forest_pipeline = pipeline(
        [
            node(
                func=train_random_forest,
                inputs=["X_train", "y_train", "params:model_regression"],
                outputs="reg_model_rf",
                name="reg_train_random_forest_node",
            ),
            node(
                func=predict,
                inputs=["reg_model_rf", "X_test"],
                outputs="reg_y_pred_rf",
                name="reg_predict_rf_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "reg_model_rf",
                    "X_train",
                    "y_train",
                    "y_test",
                    "reg_y_pred_rf",
                    "X_reference",
                ],
                outputs=["r2_score_rf", "metrics_rf", "evaluation_output_rf"],
                name="reg_evaluate_rf_node",
            ),
        ]
    )

    # --- Pipeline para XGBoost ---
    xgboost_pipeline = pipeline(
        [
            node(
                func=train_xgboost,
                inputs=["X_train", "y_train", "params:model_regression"],
                outputs="reg_model_xgb",
                name="reg_train_xgboost_node",
            ),
            node(
                func=predict,
                inputs=["reg_model_xgb", "X_test"],
                outputs="reg_y_pred_xgb",
                name="reg_predict_xgb_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "reg_model_xgb",
                    "X_train",
                    "y_train",
                    "y_test",
                    "reg_y_pred_xgb",
                    "X_reference",
                ],
                outputs=[
                    "r2_score_xgb",
                    "metrics_xgb",
                    "evaluation_output_xgb",
                ],
                name="reg_evaluate_xgb_node",
            ),
        ]
    )

    return (
        data_preparation_pipeline
        + linear_regression_pipeline
        + random_forest_pipeline
        + xgboost_pipeline
        + pipeline(
            [
                node(
                    func=plot_r2_comparison,
                    inputs=["r2_score_lr", "r2_score_rf", "r2_score_xgb"],
                    outputs="plot_r2_comparison",
                    name="plot_r2_comparison_node",
                )
            ]
        )
    )
