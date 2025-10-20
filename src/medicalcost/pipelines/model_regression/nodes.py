import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)


def plot_feature_correlation_heatmap(featured_data: pd.DataFrame) -> Figure:
    """Genera un mapa de calor de la correlación de las características finales.

    Args:
        featured_data: El DataFrame con todas las características listas para el modelo.

    Returns:
        Una figura de Matplotlib con el mapa de calor.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 12))
    corr_matrix = featured_data.drop("cost_category", axis=1).corr()
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
    )
    ax.set_title("Matriz de Correlación de Características para Modelado", fontsize=16)
    fig.tight_layout()
    plt.close(fig)
    return fig


def split_data(
    primary_medical_data: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        primary_medical_data: El DataFrame preprocesado del pipeline de `feature_engineering`.
        parameters: Diccionario de parámetros con `test_size` y `random_state`.

    Returns:
        A tuple containing:
        - X_train: Características de entrenamiento.
        - X_test: Características de prueba.
        - y_train: Variable objetivo de entrenamiento.
        - y_test: Variable objetivo de prueba.
        - X: Todas las características para referencia.
    """
    # Se elimina 'cost_category' porque es para clasificación, no para regresión.
    X = primary_medical_data.drop(["charges", "cost_category"], axis=1)
    y = primary_medical_data["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )

    return X_train, X_test, y_train, y_test, X


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Entrena un modelo de Regresión Lineal."""
    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    return multi_model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> RandomForestRegressor:
    """Entrena un modelo de RandomForest Regressor."""
    regressor = RandomForestRegressor(
        n_estimators=100, random_state=parameters["random_state"], n_jobs=-1
    )
    regressor.fit(X_train, y_train)
    return regressor


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> xgb.XGBRegressor:
    """Entrena un modelo XGBoost Regressor."""
    regressor = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        random_state=parameters["random_state"],
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)
    return regressor


def predict(model, X_test: pd.DataFrame) -> pd.Series:
    """Realiza predicciones sobre el conjunto de prueba."""
    y_pred = model.predict(X_test)
    return pd.Series(y_pred, index=X_test.index)


def evaluate_model(
    model,
    y_test: pd.Series,
    y_pred: pd.Series,
    X: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, str]:
    """Evalúa un modelo de regresión y genera métricas de rendimiento.

    Args:
        model: El modelo de regresión entrenado.
        y_test: Variable objetivo real del conjunto de prueba.
        y_pred: Predicciones del modelo.
        X: Todas las características, para referencia de columnas.

    Returns:
        A tuple containing:
        - score (dict): Diccionario con la puntuación R-cuadrado.
        - metrics (DataFrame): DataFrame con coeficientes o importancia de características.
        - evaluation_output (str): Un texto formateado con el resumen de la evaluación.
    """
    r2 = r2_score(y_test, y_pred)
    evaluation_output = f"Precisión del Modelo (R-cuadrado): {r2:.4f}\n\n"

    # Comprobar si el modelo tiene coeficientes (lineal) o importancia de características (árbol)
    if hasattr(model, "coef_"):
        metrics = pd.DataFrame(model.coef_, X.columns, columns=["Coeficiente"])
        evaluation_output += "Impacto de cada variable en el costo (Coeficientes):\n"
    elif hasattr(model, "feature_importances_"):
        metrics = pd.DataFrame(
            model.feature_importances_, X.columns, columns=["Importancia"]
        )
        evaluation_output += "Importancia de cada variable en la predicción:\n"

    metrics = metrics.sort_values(by=metrics.columns[0], ascending=False)
    evaluation_output += metrics.to_string()

    return {"r2_score": r2}, metrics, evaluation_output


# --- Nota sobre Modelos de Regresión Regularizada (Ridge, Lasso) ---
#
# No se implementaron modelos como Ridge o Lasso porque el análisis de correlación
# (feature_correlation_heatmap.png) no mostró problemas graves de multicolinealidad
# entre las características. El enfoque se centró en comparar un modelo lineal simple
# con modelos de ensamblaje (Random Forest, XGBoost) capaces de capturar interacciones
# complejas, lo cual resultó ser más beneficioso para mejorar la precisión.
