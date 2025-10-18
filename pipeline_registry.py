"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from medicalcost.pipelines import (
    data_engineering,
    data_processing,
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
    data_processing_pipeline = data_processing.create_pipeline()
    exploratory_data_pipeline = exploratory_data.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    model_classification_pipeline = model_classification.create_pipeline()
    model_regression_pipeline = model_regression.create_pipeline()

    pipelines["__default__"] = (
        data_engineering_pipeline  # 1. Load raw data
        + data_processing_pipeline  # 2. Clean the data
        + exploratory_data_pipeline  # 3. Explore the CLEANED data
        + feature_engineering_pipeline  # 4. Create dummy variables from CLEANED data
        + model_classification_pipeline  # Now, run classification models on featured data
        + model_regression_pipeline
    )
    return pipelines
