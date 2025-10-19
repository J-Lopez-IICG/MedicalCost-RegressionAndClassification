import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib  # noqa: I201

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_univariate_regressions(
    df_cleaned: pd.DataFrame,
) -> tuple[Figure, Figure, Figure, str]:
    """Genera gráficos de regresión univariada y un resumen de texto.

    Args:
        df_cleaned: Los datos limpios del seguro médico para la regresión.

    Returns:
        A tuple containing:
            - fig_age_vs_charges (Figure): Plot of age vs. charges.
            - fig_bmi_vs_charges (Figure): Plot of BMI vs. charges.
            - fig_smoker_vs_charges (Figure): Plot of smoker vs. charges.
            - univariate_output (str): A formatted string with univariate regression interpretations.
    """
    # Gráfico de regresión para edad vs. costos
    X_age = df_cleaned[["age"]]
    y_age = df_cleaned["charges"]
    model_age = LinearRegression()
    plt.style.use("seaborn-v0_8-whitegrid")
    model_age.fit(X_age, y_age)

    fig_age_vs_charges, ax_age = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x="age",
        y="charges",
        data=df_cleaned,
        line_kws={"color": "#1f77b4"},
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
    plt.close(fig_age_vs_charges)

    # Gráfico de regresión para IMC vs. costos
    X_bmi = df_cleaned[["bmi"]]
    y_bmi = df_cleaned["charges"]
    model_bmi = LinearRegression()
    model_bmi.fit(X_bmi, y_bmi)

    fig_bmi_vs_charges, ax_bmi = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x="bmi",
        y="charges",
        data=df_cleaned,
        line_kws={"color": "#2ca02c"},
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
    plt.close(fig_bmi_vs_charges)

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

    # Genera un resumen de texto con las interpretaciones de los gráficos
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
