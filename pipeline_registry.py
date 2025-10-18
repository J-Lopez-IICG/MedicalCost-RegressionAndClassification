"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from medicalcost.pipelines import (
    data_engineering,
    exploratory_data,
    feature_engineering,
    model_classification,
    model_regression,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    data_engineering_pipeline = data_engineering.create_pipeline()
    exploratory_data_pipeline = exploratory_data.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    model_classification_pipeline = model_classification.create_pipeline()
    model_regression_pipeline = model_regression.create_pipeline()

    pipelines["__default__"] = (
        data_engineering_pipeline  # First, load raw data
        + exploratory_data_pipeline  # Then, explore the raw data
        + feature_engineering_pipeline  # Then, create dummy variables
        + model_classification_pipeline  # Now, run classification models on featured data
        + model_regression_pipeline
    )
    return pipelines
