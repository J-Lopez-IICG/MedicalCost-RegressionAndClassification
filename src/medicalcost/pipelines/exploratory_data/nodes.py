import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib  # noqa: I201

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_smoker_vs_charges_distribution(
    df_cleaned: pd.DataFrame,
) -> Figure:
    """Genera un gráfico de caja para visualizar la distribución de costos entre fumadores y no fumadores.

    Args:
        df_cleaned: Los datos limpios del seguro médico.

    Returns:
        fig_smoker_vs_charges (Figure): Gráfico de caja de fumador vs. costos.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Gráfico de caja para fumador vs. costos
    fig_smoker_vs_charges, ax_smoker = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x="smoker",
        y="charges",
        data=df_cleaned,
        ax=ax_smoker,
        palette="viridis",
        hue="smoker",
    )
    ax_smoker.set_title(
        "Distribución de Costos: Fumadores vs. No Fumadores", fontsize=14, weight="bold"
    )
    ax_smoker.set_xlabel("¿Es Fumador?")
    ax_smoker.set_ylabel("Costo del Seguro (Charges)")
    fig_smoker_vs_charges.tight_layout()
    plt.close(fig_smoker_vs_charges)

    return fig_smoker_vs_charges


def plot_interactions_and_correlations(
    df_cleaned: pd.DataFrame,
) -> tuple[Figure, Figure]:
    """Genera gráficos de interacción y correlación.

    Args:
        df_cleaned: Los datos limpios del seguro médico.

    Returns:
        A tuple containing:
            - fig_bmi_smoker_interaction (Figure): Plot of BMI, smoker interaction.
            - fig_correlation_heatmap (Figure): Plot of correlation heatmap.
    """
    # Gráfico de dispersión para la interacción entre IMC y fumador
    fig_bmi_smoker_interaction, ax_interaction = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="bmi",
        y="charges",
        hue="smoker",
        data=df_cleaned,
        alpha=0.7,
        ax=ax_interaction,
    )
    ax_interaction.set_title("Interacción entre IMC, ser Fumador y Costos del Seguro")
    ax_interaction.set_xlabel("Índice de Masa Corporal (BMI)")
    ax_interaction.set_ylabel("Costo del Seguro (Charges)")
    ax_interaction.grid(True)
    fig_bmi_smoker_interaction.tight_layout()
    plt.close(fig_bmi_smoker_interaction)

    # Mapa de calor para la matriz de correlación
    numeric_cols = df_cleaned.select_dtypes(include=np.number)
    fig_correlation_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heatmap
    )
    ax_heatmap.set_title("Matriz de Correlación de Variables Numéricas")
    fig_correlation_heatmap.tight_layout()
    plt.close(fig_correlation_heatmap)

    return fig_bmi_smoker_interaction, fig_correlation_heatmap


def plot_numerical_distributions(
    df_cleaned: pd.DataFrame,
) -> tuple[Figure, Figure, Figure, Figure]:
    """Genera gráficos de distribución para las columnas numéricas.

    Args:
        df_cleaned: Los datos limpios del seguro médico.

    Returns:
        A tuple containing:
            - fig_age_hist (Figure): Histogram for age.
            - fig_bmi_hist (Figure): Histogram for BMI.
            - fig_charges_hist (Figure): Histogram for charges.
            - fig_children_bar (Figure): Bar plot for children.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Histograma para la edad
    fig_age_hist, ax_age = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_cleaned, x="age", kde=True, ax=ax_age, color="#ff7f0e")
    ax_age.set_title("Distribución de la Edad", fontsize=14, weight="bold")
    ax_age.set_xlabel("Edad")
    ax_age.set_ylabel("Frecuencia")
    fig_age_hist.tight_layout()
    plt.close(fig_age_hist)

    # Histograma para el IMC
    fig_bmi_hist, ax_bmi = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_cleaned, x="bmi", kde=True, ax=ax_bmi, color="#d62728")
    ax_bmi.set_title(
        "Distribución del Índice de Masa Corporal (BMI)", fontsize=14, weight="bold"
    )
    ax_bmi.set_xlabel("BMI")
    ax_bmi.set_ylabel("Frecuencia")
    fig_bmi_hist.tight_layout()
    plt.close(fig_bmi_hist)

    # Histograma para los costos
    fig_charges_hist, ax_charges = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_cleaned, x="charges", kde=True, ax=ax_charges, color="#9467bd")
    ax_charges.set_title(
        "Distribución de los Costos del Seguro (Charges)", fontsize=14, weight="bold"
    )
    ax_charges.set_xlabel("Charges")
    ax_charges.set_ylabel("Frecuencia")
    fig_charges_hist.tight_layout()
    plt.close(fig_charges_hist)

    # Gráfico de barras para el número de hijos
    fig_children_bar, ax_children = plt.subplots(figsize=(8, 6))
    sns.countplot(
        x="children", data=df_cleaned, ax=ax_children, palette="viridis", hue="children"
    )
    ax_children.set_title(
        "Distribución del Número de Hijos", fontsize=14, weight="bold"
    )
    ax_children.set_xlabel("Número de Hijos")
    ax_children.set_ylabel("Conteo")
    fig_children_bar.tight_layout()
    plt.close(fig_children_bar)

    return (
        fig_age_hist,
        fig_bmi_hist,
        fig_charges_hist,
        fig_children_bar,
    )


def plot_numerical_boxplots(df_cleaned: pd.DataFrame) -> tuple[Figure, Figure, Figure]:
    """Genera diagramas de caja para identificar outliers en columnas numéricas.

    Args:
        df_cleaned: Los datos limpios del seguro médico.

    Returns:
        A tuple containing:
            - fig_age_boxplot (Figure): Box plot for age.
            - fig_bmi_boxplot (Figure): Box plot for BMI.
            - fig_charges_boxplot (Figure): Box plot for charges.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Diagrama de caja para la edad
    fig_age_boxplot, ax_age = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df_cleaned["age"], ax=ax_age, color="#1f77b4")
    ax_age.set_title("Box Plot de Edad", fontsize=14, weight="bold")
    ax_age.set_ylabel("Edad")
    fig_age_boxplot.tight_layout()
    plt.close(fig_age_boxplot)

    # Diagrama de caja para el IMC
    fig_bmi_boxplot, ax_bmi = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df_cleaned["bmi"], ax=ax_bmi, color="#2ca02c")
    ax_bmi.set_title(
        "Box Plot del Índice de Masa Corporal (BMI)", fontsize=14, weight="bold"
    )
    ax_bmi.set_ylabel("BMI")
    fig_bmi_boxplot.tight_layout()
    plt.close(fig_bmi_boxplot)

    # Diagrama de caja para los costos
    fig_charges_boxplot, ax_charges = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df_cleaned["charges"], ax=ax_charges, color="#ff7f0e")
    ax_charges.set_title(
        "Box Plot de los Costos del Seguro (Charges)", fontsize=14, weight="bold"
    )
    ax_charges.set_ylabel("Charges")
    fig_charges_boxplot.tight_layout()
    plt.close(fig_charges_boxplot)

    return fig_age_boxplot, fig_bmi_boxplot, fig_charges_boxplot


def plot_univariate_regressions(
    df_cleaned: pd.DataFrame, parameters: dict
) -> list[Figure]:
    """Genera gráficos de regresión univariada separados para las columnas especificadas.

    Args:
        df_cleaned: Los datos limpios del seguro médico.
        parameters: Diccionario que contiene `univariate_plot_columns`.

    Returns:
        Una lista de figuras de Matplotlib, una para cada columna.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    columns_to_plot = parameters["univariate_plot_columns"]
    figs = []

    # Iterar sobre las columnas para crear una figura separada para cada una
    for col in columns_to_plot:
        X_col = df_cleaned[[col]]
        y_col = df_cleaned["charges"]

        model = LinearRegression()
        model.fit(X_col, y_col)
        r2_val = r2_score(y_col, model.predict(X_col))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            x=col,
            y="charges",
            data=df_cleaned,
            ax=ax,
            seed=42,
            line_kws={"color": "red"},
        )
        ax.set_title(
            f"Regresión: Costos vs. {col.capitalize()} (R² = {r2_val:.2f})",
            fontsize=14,
            weight="bold",
        )
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Costo del Seguro (Charges)")
        fig.tight_layout()
        figs.append(fig)
        plt.close(fig)

    return figs
