from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_classification_data,
    train_and_evaluate_logistic_regression,
    train_and_evaluate_random_forest,
    train_and_evaluate_xgboost,
    train_and_evaluate_svc,
    create_classification_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_classification_data,
                inputs={
                    "featured_classification_data": "featured_classification_data",
                    "parameters": "params:model_classification",
                },
                outputs=[
                    "cls_X_train",
                    "cls_X_test",
                    "cls_y_train",
                    "cls_y_test",
                ],
                name="split_classification_data_node",
            ),
            node(
                func=train_and_evaluate_logistic_regression,
                inputs=[
                    "cls_X_train",
                    "cls_X_test",
                    "cls_y_train",
                    "cls_y_test",
                ],
                outputs=[
                    "log_reg_model",
                    "classification_accuracy_log_reg",
                    "classification_report_log_reg",
                    "log_reg_feature_importance",
                    "plot_log_reg_feature_importance",
                ],
                name="train_and_evaluate_logistic_regression_node",
            ),
            node(
                func=train_and_evaluate_random_forest,
                inputs={
                    "cls_X_train": "cls_X_train",
                    "cls_X_test": "cls_X_test",
                    "cls_y_train": "cls_y_train",
                    "cls_y_test": "cls_y_test",
                    "parameters": "params:model_classification.random_forest",
                },
                outputs=[
                    "random_forest_model",
                    "classification_accuracy_rf",
                    "classification_report_rf",
                    "plot_rf_grid_search_heatmap",
                ],
                name="train_and_evaluate_random_forest_node",
            ),
            node(
                func=train_and_evaluate_xgboost,
                inputs={
                    "cls_X_train": "cls_X_train",
                    "cls_X_test": "cls_X_test",
                    "cls_y_train": "cls_y_train",
                    "cls_y_test": "cls_y_test",
                    "parameters": "params:model_classification.xgboost",
                },
                outputs=[
                    "xgboost_model",
                    "classification_accuracy_xgb",
                    "classification_report_xgb",
                    "plot_xgb_grid_search_heatmap",
                ],
                name="train_and_evaluate_xgboost_node",
            ),
            node(
                func=train_and_evaluate_svc,
                inputs={
                    "cls_X_train": "cls_X_train",
                    "cls_X_test": "cls_X_test",
                    "cls_y_train": "cls_y_train",
                    "cls_y_test": "cls_y_test",
                    "parameters": "params:model_classification.svc",
                },
                outputs=[
                    "svc_model",
                    "classification_accuracy_svc",
                    "classification_report_svc",
                    "plot_svc_grid_search_heatmap",
                ],
                name="train_and_evaluate_svc_node",
            ),
            node(
                func=create_classification_summary,
                inputs=[
                    "classification_accuracy_log_reg",
                    "classification_accuracy_rf",
                    "classification_accuracy_xgb",
                    "classification_accuracy_svc",
                ],
                outputs="classification_summary_output",
                name="create_classification_summary_node",
            ),
        ]
    )
