"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from medicalcost.pipelines import model_regression
from medicalcost.pipelines import model_classification


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["model_regression"] = model_regression.create_pipeline()
    pipelines["model_classification"] = model_classification.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    return pipelines
