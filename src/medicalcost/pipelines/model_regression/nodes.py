import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import logging

log = logging.getLogger(__name__)


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
