import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # Import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import logging

log = logging.getLogger(__name__)


def train_model(
    df_model: pd.DataFrame,
) -> tuple[LinearRegression, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Trains a linear regression model and splits data into training and testing sets.

    Args:
        df_model: The preprocessed DataFrame.

    Returns:
        A tuple containing:
            - The trained LinearRegression model.
            - reg_X_test (DataFrame): Testing features.
            - reg_y_test (Series): Actual testing targets.
            - y_pred (Series): Predicted testing targets.
            - X (DataFrame): All features used for training.
    """
    X = df_model.drop("charges", axis=1)
    y = df_model["charges"]
    X_train, reg_X_test, y_train, reg_y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    """Evaluates the model and generates R-squared score, coefficients, and a text summary.

    Args:
        multi_model: The trained LinearRegression model.
        reg_X_test: Testing features.
        reg_y_test: Actual testing targets.
        y_pred: Predicted testing targets.
        X: All features used for training.

    Returns:
        A tuple containing:
            - r2 (float): The R-squared score.
            - coeffs (DataFrame): DataFrame of model coefficients.
            - evaluation_output (str): A formatted string with model evaluation results.
    """
    r2 = r2_score(reg_y_test, y_pred)
    coeffs = pd.DataFrame(multi_model.coef_, X.columns, columns=["Coeficiente"])

    evaluation_output = f"Precisión del Modelo (R-cuadrado): {r2:.4f}\n\n"
    evaluation_output += "Impacto de cada variable en el costo:\n"
    evaluation_output += coeffs.to_string()

    return r2, coeffs, evaluation_output
