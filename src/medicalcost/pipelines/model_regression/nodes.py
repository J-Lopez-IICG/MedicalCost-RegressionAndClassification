import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import logging

log = logging.getLogger(__name__)


def create_univariate_regression_plots(
    df_cleaned: pd.DataFrame,
) -> tuple[Figure, Figure, str]:
    """Genera gráficos de regresión univariada y un resumen de texto.

    Args:
        df_cleaned: Los datos limpios del seguro médico para la regresión.

    Returns:
        A tuple containing:
            - fig_age_vs_charges (Figure): Plot of age vs. charges.
            - fig_bmi_vs_charges (Figure): Plot of BMI vs. charges.
            - univariate_output (str): A formatted string with univariate regression interpretations.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Gráfico de regresión para edad vs. costos
    X_age = df_cleaned[["age"]]
    y_age = df_cleaned["charges"]
    model_age = LinearRegression()
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

    # Genera un resumen de texto con las interpretaciones de los gráficos
    univariate_output = f"Ecuación (Edad): charges = {model_age.coef_[0]:.2f} * age + {model_age.intercept_:.2f}\n\n"
    univariate_output += f"Ecuación (IMC): charges = {model_bmi.coef_[0]:.2f} * bmi + {model_bmi.intercept_:.2f}\n\n"
    univariate_output += "Interpretación (Edad): Se observa una clara tendencia positiva: a mayor edad, mayor es el costo. Sin embargo, los datos parecen agruparse en tres 'bandas' distintas. Esto sugiere que hay otro factor muy importante que no estamos considerando.\n\n"
    univariate_output += "Interpretación (IMC): La relación positiva también existe, pero es más débil y los datos están mucho más dispersos. Al igual que con la edad, parece haber una división en los datos que este modelo simple no puede explicar.\n\n"
    univariate_output += "Interpretación (Fumador): ¡Este es el hallazgo clave! La diferencia en costos entre fumadores y no fumadores es masiva. Ser fumador no solo eleva el costo promedio, sino que también aumenta la variabilidad. Esto explica las 'bandas' que vimos en los gráficos anteriores."

    return (
        fig_age_vs_charges,
        fig_bmi_vs_charges,
        univariate_output,
    )


def train_model(
    primary_medical_data: pd.DataFrame,
    parameters: dict,
) -> tuple[LinearRegression, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Divide los datos y entrena un modelo de regresión lineal.

    Args:
        primary_medical_data: El DataFrame preprocesado del pipeline de `feature_engineering`.
        parameters: Diccionario de parámetros con `test_size` y `random_state`.

    Returns:
        A tuple containing:
        - El modelo de regresión lineal entrenado.
        - reg_X_test (DataFrame): Características del conjunto de prueba.
        - reg_y_test (Series): Variable objetivo real del conjunto de prueba.
        - y_pred (Series): Predicciones sobre el conjunto de prueba.
        - X (DataFrame): Todas las características, para referencia de columnas.
    """
    # Se elimina 'cost_category' porque es para clasificación, no para regresión.
    X = primary_medical_data.drop(["charges", "cost_category"], axis=1)
    y = primary_medical_data["charges"]
    X_train, reg_X_test, y_train, reg_y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )

    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_pred = multi_model.predict(reg_X_test)

    return (
        multi_model,
        reg_X_test,
        reg_y_test,
        pd.Series(y_pred, index=reg_X_test.index),
        X,
    )


def evaluate_model(
    multi_model: LinearRegression,
    reg_X_test: pd.DataFrame,
    reg_y_test: pd.Series,
    y_pred: pd.Series,
    X: pd.DataFrame,
) -> tuple[float, pd.DataFrame, str]:
    """Evalúa el modelo y genera métricas de rendimiento.

    Args:
        multi_model: El modelo de regresión lineal entrenado.
        reg_X_test: Características del conjunto de prueba.
        reg_y_test: Variable objetivo real del conjunto de prueba.
        y_pred: Predicciones del modelo.
        X: Todas las características, para referencia de columnas.

    Returns:
        A tuple containing:
        - r2 (float): La puntuación R-cuadrado del modelo.
        - coeffs (DataFrame): DataFrame con los coeficientes del modelo.
        - evaluation_output (str): Un texto formateado con el resumen de la evaluación.
    """
    r2 = r2_score(reg_y_test, y_pred)
    coeffs = pd.DataFrame(multi_model.coef_, X.columns, columns=["Coeficiente"])

    evaluation_output = f"Precisión del Modelo (R-cuadrado): {r2:.4f}\n\n"
    evaluation_output += "Impacto de cada variable en el costo:\n"
    evaluation_output += coeffs.to_string()

    return r2, coeffs, evaluation_output
