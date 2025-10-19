"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from medicalcost.pipelines import (
    data_engineering,
    data_processing,
    exploratory_data,
    feature_engineering,
    model_regression,
    # model_classification,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    data_engineering_pipeline = data_engineering.create_pipeline()
    data_processing_pipeline = data_processing.create_pipeline()
    exploratory_data_pipeline = exploratory_data.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    model_regression_pipeline = model_regression.create_pipeline()
    # model_classification_pipeline = model_classification.create_pipeline()

    pipelines["__default__"] = (
        data_engineering_pipeline
        + data_processing_pipeline
        + exploratory_data_pipeline
        + feature_engineering_pipeline
        + model_regression_pipeline
        # + model_classification_pipeline
    )
    return pipelines
