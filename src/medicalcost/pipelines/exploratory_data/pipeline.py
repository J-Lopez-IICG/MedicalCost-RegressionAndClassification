from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_univariate_regressions,
    plot_interactions_and_correlations,
    plot_numerical_distributions,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=plot_univariate_regressions,
                inputs="raw_medical_data_csv",
                outputs=[
                    "plot_age_vs_charges",
                    "plot_bmi_vs_charges",
                    "plot_smoker_vs_charges",
                    "univariate_regression_output",
                ],
                name="plot_univariate_regressions_node",
            ),
            node(
                func=plot_interactions_and_correlations,
                inputs="raw_medical_data_csv",
                outputs=[
                    "plot_bmi_smoker_interaction",
                    "plot_correlation_heatmap",
                ],
                name="plot_interactions_and_correlations_node",
            ),
            node(
                func=plot_numerical_distributions,
                inputs="raw_medical_data_csv",
                outputs=[
                    "plot_age_histogram",
                    "plot_bmi_histogram",
                    "plot_charges_histogram",
                    "plot_children_barplot",
                ],
                name="plot_numerical_distributions_node",
            ),
        ]
    )
