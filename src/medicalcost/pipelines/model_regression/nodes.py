import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # Import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json


def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw data by creating dummy variables for categorical features.

    Args:
        df_raw: The raw medical insurance data.

    Returns:
        The preprocessed DataFrame with dummy variables.
    """
    df_model = df_raw.copy()
    df_model = pd.get_dummies(
        df_model, columns=["sex", "smoker", "region"], drop_first=True
    )
    return df_model


def train_model(
    df_model: pd.DataFrame,
) -> tuple[LinearRegression, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Trains a linear regression model and splits data into training and testing sets.

    Args:
        df_model: The preprocessed DataFrame.

    Returns:
        A tuple containing:
            - The trained LinearRegression model.
            - X_test (DataFrame): Testing features.
            - y_test (Series): Actual testing targets.
            - y_pred (Series): Predicted testing targets.
            - X (DataFrame): All features used for training.
    """
    X = df_model.drop("charges", axis=1)
    y = df_model["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_pred = multi_model.predict(X_test)

    return multi_model, X_test, y_test, pd.Series(y_pred, index=X_test.index), X


def evaluate_model(
    multi_model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    X: pd.DataFrame,
) -> tuple[float, pd.DataFrame, str]:
    """Evaluates the model and generates R-squared score, coefficients, and a text summary.

    Args:
        multi_model: The trained LinearRegression model.
        X_test: Testing features.
        y_test: Actual testing targets.
        y_pred: Predicted testing targets.
        X: All features used for training.

    Returns:
        A tuple containing:
            - r2 (float): The R-squared score.
            - coeffs (DataFrame): DataFrame of model coefficients.
            - evaluation_output (str): A formatted string with model evaluation results.
    """
    r2 = r2_score(y_test, y_pred)
    coeffs = pd.DataFrame(multi_model.coef_, X.columns, columns=["Coeficiente"])

    evaluation_output = f"Precisión del Modelo (R-cuadrado): {r2:.4f}\n\n"
    evaluation_output += "Impacto de cada variable en el costo:\n"
    evaluation_output += coeffs.to_string()

    return r2, coeffs, evaluation_output


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
    model_age.fit(X_age, y_age)

    fig_age_vs_charges, ax_age = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x="age",
        y="charges",
        data=df_raw,
        line_kws={"color": "red"},
        scatter_kws={"alpha": 0.5},
        ax=ax_age,
    )
    ax_age.set_title("Regresión Lineal: Costos del Seguro vs. Edad")
    ax_age.set_xlabel("Edad")
    ax_age.set_ylabel("Costo del Seguro (Charges)")
    ax_age.grid(True)

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
        line_kws={"color": "green"},
        scatter_kws={"alpha": 0.5},
        ax=ax_bmi,
    )
    ax_bmi.set_title("Regresión Lineal: Costos del Seguro vs. IMC (BMI)")
    ax_bmi.set_xlabel("Índice de Masa Corporal (BMI)")
    ax_bmi.set_ylabel("Costo del Seguro (Charges)")
    ax_bmi.grid(True)

    # Plot smoker vs. charges
    fig_smoker_vs_charges, ax_smoker = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="smoker", y="charges", data=df_raw, ax=ax_smoker)
    ax_smoker.set_title("Distribución de Costos para Fumadores vs. No Fumadores")
    ax_smoker.set_xlabel("¿Es Fumador?")
    ax_smoker.set_ylabel("Costo del Seguro (Charges)")
    ax_smoker.grid(True, axis="y", linestyle="--", alpha=0.7)

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

    # Plot correlation heatmap
    numeric_cols = df_raw.select_dtypes(include=np.number)
    fig_correlation_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heatmap
    )
    ax_heatmap.set_title("Matriz de Correlación de Variables Numéricas")

    return fig_bmi_smoker_interaction, fig_correlation_heatmap
