from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_classification_data,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_svc,
    predict,
    evaluate_classifier,
    extract_and_plot_log_reg_importance,
    plot_grid_search_heatmap,
    create_classification_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de clasificación para entrenar y comparar múltiples modelos."""
    data_preparation_pipeline = pipeline(
        [
            node(
                func=split_classification_data,
                inputs={
                    "featured_classification_data": "primary_medical_data",
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
        ]
    )

    logistic_regression_pipeline = pipeline(
        [
            node(
                func=train_logistic_regression,
                inputs=[
                    "cls_X_train",
                    "cls_y_train",
                    "params:model_classification.logistic_regression",
                ],
                outputs="log_reg_model",
                name="cls_train_logistic_regression_node",
            ),
            node(
                func=predict,
                inputs=["log_reg_model", "cls_X_test"],
                outputs="cls_y_pred_lr",
                name="cls_predict_lr_node",
            ),
            node(
                func=evaluate_classifier,
                inputs=["log_reg_model", "cls_y_test", "cls_y_pred_lr"],
                outputs=[
                    "classification_accuracy_log_reg",
                    "classification_report_log_reg",
                ],
                name="cls_evaluate_lr_node",
            ),
            node(
                func=extract_and_plot_log_reg_importance,
                inputs=["log_reg_model", "cls_X_train"],
                outputs=[
                    "log_reg_feature_importance",
                    "plot_log_reg_feature_importance",
                ],
                name="cls_plot_log_reg_importance_node",
            ),
        ]
    )

    random_forest_pipeline = pipeline(
        [
            node(
                func=train_random_forest,
                inputs=[
                    "cls_X_train",
                    "cls_y_train",
                    "params:model_classification.random_forest",
                ],
                outputs="random_forest_model",
                name="cls_train_random_forest_node",
            ),
            node(
                func=predict,
                inputs=["random_forest_model", "cls_X_test"],
                outputs="cls_y_pred_rf",
                name="cls_predict_rf_node",
            ),
            node(
                func=evaluate_classifier,
                inputs=["random_forest_model", "cls_y_test", "cls_y_pred_rf"],
                outputs=["classification_accuracy_rf", "classification_report_rf"],
                name="cls_evaluate_rf_node",
            ),
            node(
                func=plot_grid_search_heatmap,
                inputs={
                    "grid_search_model": "random_forest_model",
                    "x_param": "params:model_classification.random_forest.heatmap_params.x",
                    "y_param": "params:model_classification.random_forest.heatmap_params.y",
                    "model_name": "params:model_classification.random_forest.heatmap_params.name",
                },
                outputs="plot_rf_grid_search_heatmap",
                name="cls_plot_rf_heatmap_node",
            ),
        ]
    )

    xgboost_pipeline = pipeline(
        [
            node(
                func=train_xgboost,
                inputs=[
                    "cls_X_train",
                    "cls_y_train",
                    "params:model_classification.xgboost",
                ],
                outputs="xgboost_model",
                name="cls_train_xgboost_node",
            ),
            node(
                func=predict,
                inputs=["xgboost_model", "cls_X_test"],
                outputs="cls_y_pred_xgb",
                name="cls_predict_xgb_node",
            ),
            node(
                func=evaluate_classifier,
                inputs=["xgboost_model", "cls_y_test", "cls_y_pred_xgb"],
                outputs=["classification_accuracy_xgb", "classification_report_xgb"],
                name="cls_evaluate_xgb_node",
            ),
            node(
                func=plot_grid_search_heatmap,
                inputs={
                    "grid_search_model": "xgboost_model",
                    "x_param": "params:model_classification.xgboost.heatmap_params.x",
                    "y_param": "params:model_classification.xgboost.heatmap_params.y",
                    "model_name": "params:model_classification.xgboost.heatmap_params.name",
                },
                outputs="plot_xgb_grid_search_heatmap",
                name="cls_plot_xgb_heatmap_node",
            ),
        ]
    )

    svc_pipeline = pipeline(
        [
            node(
                func=train_svc,
                inputs=[
                    "cls_X_train",
                    "cls_y_train",
                    "params:model_classification.svc",
                ],
                outputs="svc_model",
                name="cls_train_svc_node",
            ),
            node(
                func=predict,
                inputs=["svc_model", "cls_X_test"],
                outputs="cls_y_pred_svc",
                name="cls_predict_svc_node",
            ),
            node(
                func=evaluate_classifier,
                inputs=["svc_model", "cls_y_test", "cls_y_pred_svc"],
                outputs=["classification_accuracy_svc", "classification_report_svc"],
                name="cls_evaluate_svc_node",
            ),
            node(
                func=plot_grid_search_heatmap,
                inputs={
                    "grid_search_model": "svc_model",
                    "x_param": "params:model_classification.svc.heatmap_params.x",
                    "y_param": "params:model_classification.svc.heatmap_params.y",
                    "model_name": "params:model_classification.svc.heatmap_params.name",
                },
                outputs="plot_svc_grid_search_heatmap",
                name="cls_plot_svc_heatmap_node",
            ),
        ]
    )

    summary_pipeline = pipeline(
        [
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

    return (
        data_preparation_pipeline
        + logistic_regression_pipeline
        + random_forest_pipeline
        + xgboost_pipeline
        + svc_pipeline
        + summary_pipeline
    )
