import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_univariate_regressions(
    df_raw: pd.DataFrame,
) -> tuple[Figure, Figure, Figure, str]:
    """Generates univariate regression plots and a text summary.

    Args:
        df_raw: The raw medical insurance data.

    Returns:
        A tuple containing:
            - fig_age_vs_charges (Figure): Plot of age vs. charges.
            - fig_bmi_vs_charges (Figure): Plot of BMI vs. charges.
            - fig_smoker_vs_charges (Figure): Plot of smoker vs. charges.
            - univariate_output (str): A formatted string with univariate regression interpretations.
    """
    # Plot age vs. charges
    X_age = df_raw[["age"]]
    y_age = df_raw["charges"]
    model_age = LinearRegression()
    plt.style.use("seaborn-v0_8-whitegrid")
    model_age.fit(X_age, y_age)

    fig_age_vs_charges, ax_age = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x="age",
        y="charges",
        data=df_raw,
        line_kws={"color": "#1f77b4"},  # Changed color to a standard blue
        scatter_kws={"alpha": 0.5},
        ax=ax_age,
    )
    r2_age = r2_score(y_age, model_age.predict(X_age))
    ax_age.set_title(
        f"Regresión Lineal: Costos vs. Edad (R² = {r2_age:.2f})",
        fontsize=14,
        weight="bold",
    )
    ax_age.set_xlabel("Edad")
    ax_age.set_ylabel("Costo del Seguro (Charges)")
    ax_age.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig_age_vs_charges.tight_layout()
    plt.close(fig_age_vs_charges)  # Added to close figure

    # Plot BMI vs. charges
    X_bmi = df_raw[["bmi"]]
    y_bmi = df_raw["charges"]
    model_bmi = LinearRegression()
    model_bmi.fit(X_bmi, y_bmi)

    fig_bmi_vs_charges, ax_bmi = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x="bmi",
        y="charges",
        data=df_raw,
        line_kws={"color": "#2ca02c"},  # Changed color to a standard green
        scatter_kws={"alpha": 0.5},
        ax=ax_bmi,
    )
    r2_bmi = r2_score(y_bmi, model_bmi.predict(X_bmi))
    ax_bmi.set_title(
        f"Regresión Lineal: Costos vs. IMC (R² = {r2_bmi:.2f})",
        fontsize=14,
        weight="bold",
    )
    ax_bmi.set_xlabel("Índice de Masa Corporal (BMI)")
    ax_bmi.set_ylabel("Costo del Seguro (Charges)")
    ax_bmi.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig_bmi_vs_charges.tight_layout()
    plt.close(fig_bmi_vs_charges)  # Added to close figure

    # Plot smoker vs. charges
    fig_smoker_vs_charges, ax_smoker = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x="smoker", y="charges", data=df_raw, ax=ax_smoker, palette="viridis"
    )  # Added palette
    ax_smoker.set_title(
        "Distribución de Costos: Fumadores vs. No Fumadores", fontsize=14, weight="bold"
    )
    ax_smoker.set_xlabel("¿Es Fumador?")
    ax_smoker.set_ylabel("Costo del Seguro (Charges)")
    fig_smoker_vs_charges.tight_layout()
    plt.close(fig_smoker_vs_charges)  # Added to close figure

    univariate_output = f"Ecuación (Edad): charges = {model_age.coef_[0]:.2f} * age + {model_age.intercept_:.2f}\n\n"
    univariate_output += f"Ecuación (IMC): charges = {model_bmi.coef_[0]:.2f} * bmi + {model_bmi.intercept_:.2f}\n\n"
    univariate_output += "Interpretación (Edad): Se observa una clara tendencia positiva: a mayor edad, mayor es el costo. Sin embargo, los datos parecen agruparse en tres 'bandas' distintas. Esto sugiere que hay otro factor muy importante que no estamos considerando.\n\n"
    univariate_output += "Interpretación (IMC): La relación positiva también existe, pero es más débil y los datos están mucho más dispersos. Al igual que con la edad, parece haber una división en los datos que este modelo simple no puede explicar.\n\n"
    univariate_output += "Interpretación (Fumador): ¡Este es el hallazgo clave! La diferencia en costos entre fumadores y no fumadores es masiva. Ser fumador no solo eleva el costo promedio, sino que también aumenta la variabilidad. Esto explica las 'bandas' que vimos en los gráficos anteriores."

    return (
        fig_age_vs_charges,
        fig_bmi_vs_charges,
        fig_smoker_vs_charges,
        univariate_output,
    )


def plot_interactions_and_correlations(df_raw: pd.DataFrame) -> tuple[Figure, Figure]:
    """Generates interaction and correlation plots.

    Args:
        df_raw: The raw medical insurance data.

    Returns:
        A tuple containing:
            - fig_bmi_smoker_interaction (Figure): Plot of BMI, smoker interaction.
            - fig_correlation_heatmap (Figure): Plot of correlation heatmap.
    """
    # Plot BMI, smoker interaction
    fig_bmi_smoker_interaction, ax_interaction = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="bmi", y="charges", hue="smoker", data=df_raw, alpha=0.7, ax=ax_interaction
    )
    ax_interaction.set_title("Interacción entre IMC, ser Fumador y Costos del Seguro")
    ax_interaction.set_xlabel("Índice de Masa Corporal (BMI)")
    ax_interaction.set_ylabel("Costo del Seguro (Charges)")
    ax_interaction.grid(True)
    fig_bmi_smoker_interaction.tight_layout()  # Ensure tight layout for this plot too
    plt.close(fig_bmi_smoker_interaction)  # Added to close figure

    # Plot correlation heatmap
    numeric_cols = df_raw.select_dtypes(include=np.number)
    fig_correlation_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heatmap
    )
    ax_heatmap.set_title("Matriz de Correlación de Variables Numéricas")
    fig_correlation_heatmap.tight_layout()  # Ensure tight layout for this plot too
    plt.close(fig_correlation_heatmap)  # Added to close figure

    return fig_bmi_smoker_interaction, fig_correlation_heatmap


def plot_numerical_distributions(
    df_raw: pd.DataFrame,
) -> tuple[Figure, Figure, Figure, Figure]:
    """Generates distribution plots for numerical columns.

    Args:
        df_raw: The raw medical insurance data.

    Returns:
        A tuple containing:
            - fig_age_hist (Figure): Histogram for age.
            - fig_bmi_hist (Figure): Histogram for BMI.
            - fig_charges_hist (Figure): Histogram for charges.
            - fig_children_bar (Figure): Bar plot for children.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Histograms for continuous numerical columns
    fig_age_hist, ax_age = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df_raw, x="age", kde=True, ax=ax_age, color="#ff7f0e"
    )  # Changed color to a standard orange
    ax_age.set_title("Distribución de la Edad", fontsize=14, weight="bold")
    ax_age.set_xlabel("Edad")
    ax_age.set_ylabel("Frecuencia")
    fig_age_hist.tight_layout()
    plt.close(fig_age_hist)  # Added to close figure

    fig_bmi_hist, ax_bmi = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df_raw, x="bmi", kde=True, ax=ax_bmi, color="#d62728"
    )  # Changed color to a standard red
    ax_bmi.set_title(
        "Distribución del Índice de Masa Corporal (BMI)", fontsize=14, weight="bold"
    )
    ax_bmi.set_xlabel("BMI")
    ax_bmi.set_ylabel("Frecuencia")
    fig_bmi_hist.tight_layout()
    plt.close(fig_bmi_hist)  # Added to close figure

    fig_charges_hist, ax_charges = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df_raw, x="charges", kde=True, ax=ax_charges, color="#9467bd"
    )  # Changed color to a standard purple
    ax_charges.set_title(
        "Distribución de los Costos del Seguro (Charges)", fontsize=14, weight="bold"
    )
    ax_charges.set_xlabel("Charges")
    ax_charges.set_ylabel("Frecuencia")
    fig_charges_hist.tight_layout()
    plt.close(fig_charges_hist)  # Added to close figure

    # Bar plot for discrete numerical column (children)
    fig_children_bar, ax_children = plt.subplots(figsize=(8, 6))
    sns.countplot(x="children", data=df_raw, ax=ax_children, palette="viridis")
    ax_children.set_title(
        "Distribución del Número de Hijos", fontsize=14, weight="bold"
    )
    ax_children.set_xlabel("Número de Hijos")
    ax_children.set_ylabel("Conteo")
    fig_children_bar.tight_layout()
    plt.close(fig_children_bar)  # Added to close figure

    return (
        fig_age_hist,
        fig_bmi_hist,
        fig_charges_hist,
        fig_children_bar,
    )
