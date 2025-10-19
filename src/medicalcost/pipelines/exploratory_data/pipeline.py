from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_univariate_regressions,
    plot_interactions_and_correlations,
    plot_numerical_distributions,
    plot_numerical_boxplots,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de análisis exploratorio de datos (EDA).

    Este pipeline toma los datos limpios (después del pipeline de data_processing)
    y genera una serie de visualizaciones para entender las distribuciones,
    correlaciones e interacciones entre las variables.

    Returns:
        Un objeto Pipeline que define el flujo de trabajo de EDA.
    """
    return pipeline(
        [
            node(
                # Genera gráficos de regresión simple para 'age' y 'bmi' vs 'charges'.
                func=plot_univariate_regressions,
                inputs="processed_medical_data",
                outputs=[
                    "plot_age_vs_charges",
                    "plot_bmi_vs_charges",
                    "plot_smoker_vs_charges",
                    "univariate_regression_output",
                ],
                name="plot_univariate_regressions_node",
            ),
            node(
                # Visualiza la interacción entre 'bmi' y 'smoker', y la correlación general.
                func=plot_interactions_and_correlations,
                inputs="processed_medical_data",
                outputs=[
                    "plot_bmi_smoker_interaction",
                    "plot_correlation_heatmap",
                ],
                name="plot_interactions_and_correlations_node",
            ),
            node(
                # Crea histogramas para las variables numéricas continuas.
                func=plot_numerical_distributions,
                inputs="processed_medical_data",
                outputs=[
                    "plot_age_histogram",
                    "plot_bmi_histogram",
                    "plot_charges_histogram",
                    "plot_children_barplot",
                ],
                name="plot_numerical_distributions_node",
            ),
            node(
                # Genera diagramas de caja para detectar outliers en variables numéricas.
                func=plot_numerical_boxplots,
                inputs="processed_medical_data",
                outputs=[
                    "plot_age_boxplot",
                    "plot_bmi_boxplot",
                    "plot_charges_boxplot",
                ],
                name="plot_numerical_boxplots_node",
            ),
        ]
    )
