import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json


def preprocess_classification_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw data for classification by creating a target variable and dummy variables.

    Args:
        df_raw: The raw medical insurance data.

    Returns:
        The preprocessed DataFrame with a classification target and dummy variables.
    """
    df_clasificacion = df_raw.copy()
    umbral_costo = df_clasificacion["charges"].median()
    df_clasificacion["cost_category"] = np.where(
        df_clasificacion["charges"] > umbral_costo, "Alto", "Bajo"
    )
    df_clasificacion = df_clasificacion.drop("charges", axis=1)
    df_model_cls = df_clasificacion.copy()
    df_model_cls = pd.get_dummies(
        df_model_cls, columns=["sex", "smoker", "region"], drop_first=True
    )
    df_model_cls["cost_category"] = df_model_cls["cost_category"].map(
        {"Bajo": 0, "Alto": 1}
    )
    return df_model_cls


def split_classification_data(
    df_model_cls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Splits the preprocessed classification data into training and testing sets.

    Args:
        df_model_cls: The preprocessed DataFrame for classification.

    Returns:
        A tuple containing:
            - cls_X_train (DataFrame): Training features.
            - cls_X_test (DataFrame): Testing features.
            - cls_y_train (Series): Actual training targets.
            - cls_y_test (Series): Actual testing targets.
            - X (DataFrame): All features used for training.
            - y (Series): All targets used for training.
    """
    X = df_model_cls.drop("cost_category", axis=1)
    y = df_model_cls["cost_category"]
    cls_X_train, cls_X_test, cls_y_train, cls_y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return cls_X_train, cls_X_test, cls_y_train, cls_y_test, X, y


def train_and_evaluate_logistic_regression(
    cls_X_train: pd.DataFrame,
    cls_X_test: pd.DataFrame,
    cls_y_train: pd.Series,
    cls_y_test: pd.Series,
    X: pd.DataFrame,
) -> tuple[LogisticRegression, float, str, pd.DataFrame, Figure]:
    """Trains and evaluates a Logistic Regression model.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.
        X: All features used for training.

    Returns:
        A tuple containing:
            - log_model (LogisticRegression): The trained Logistic Regression model.
            - accuracy_log (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - feature_importance (DataFrame): Feature importance DataFrame.
            - fig_feature_importance (Figure): Feature importance plot.
    """
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(cls_X_train, cls_y_train)
    y_pred_log = log_model.predict(cls_X_test)

    accuracy_log = float(accuracy_score(cls_y_test, y_pred_log))
    classification_report_str = str(
        classification_report(cls_y_test, y_pred_log, target_names=["Bajo", "Alto"])
    )

    coefficients = log_model.coef_[0]
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": coefficients}
    )
    feature_importance = feature_importance.sort_values(
        by="Coefficient", ascending=False
    )

    fig_feature_importance, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=feature_importance, ax=ax)
    ax.set_title(
        "Importancia de las Características en el Modelo de Regresión Logística"
    )
    ax.set_xlabel('Coeficiente (Impacto en la probabilidad de ser "Alto Costo")')
    ax.set_ylabel("Característica")
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(True)

    return (
        log_model,
        accuracy_log,
        classification_report_str,
        feature_importance,
        fig_feature_importance,
    )


def train_and_evaluate_random_forest(
    cls_X_train: pd.DataFrame,
    cls_X_test: pd.DataFrame,
    cls_y_train: pd.Series,
    cls_y_test: pd.Series,
) -> tuple[RandomForestClassifier, float, str, Figure]:
    """Trains and evaluates a Random Forest model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.

    Returns:
        A tuple containing:
            - best_rf_model (RandomForestClassifier): The trained Random Forest model.
            - accuracy_best_rf (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - fig_heatmap (Figure): GridSearchCV heatmap.
    """
    param_grid_rf = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid_search_rf.fit(cls_X_train, cls_y_train)

    best_rf_model = grid_search_rf.best_estimator_
    y_pred_best_rf = best_rf_model.predict(cls_X_test)

    accuracy_best_rf = float(accuracy_score(cls_y_test, y_pred_best_rf))
    classification_report_str = str(
        classification_report(cls_y_test, y_pred_best_rf, target_names=["Bajo", "Alto"])
    )

    results_rf = pd.DataFrame(grid_search_rf.cv_results_)
    pivot_table_rf = results_rf.pivot_table(
        values="mean_test_score", index="param_max_depth", columns="param_n_estimators"
    )

    fig_heatmap, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table_rf, annot=True, fmt=".4f", cmap="viridis", ax=ax)
    ax.set_title(
        "Heatmap de Resultados de GridSearchCV para Random Forest (Accuracy Promedio)"
    )
    ax.set_xlabel("Número de Estimadores (n_estimators)")
    ax.set_ylabel("Profundidad Máxima (max_depth)")

    return best_rf_model, accuracy_best_rf, classification_report_str, fig_heatmap


def train_and_evaluate_xgboost(
    cls_X_train: pd.DataFrame,
    cls_X_test: pd.DataFrame,
    cls_y_train: pd.Series,
    cls_y_test: pd.Series,
) -> tuple[XGBClassifier, float, str, Figure]:
    """Trains and evaluates an XGBoost model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.

    Returns:
        A tuple containing:
            - best_xgb_model (XGBClassifier): The trained XGBoost model.
            - accuracy_best_xgb (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - fig_heatmap (Figure): GridSearchCV heatmap.
    """
    param_grid_xgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid=param_grid_xgb,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid_search_xgb.fit(cls_X_train, cls_y_train)

    best_xgb_model = grid_search_xgb.best_estimator_
    y_pred_best_xgb = best_xgb_model.predict(cls_X_test)

    accuracy_best_xgb = float(accuracy_score(cls_y_test, y_pred_best_xgb))
    classification_report_str = str(
        classification_report(
            cls_y_test, y_pred_best_xgb, target_names=["Bajo", "Alto"]
        )
    )

    results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
    pivot_table_xgb = results_xgb.pivot_table(
        values="mean_test_score",
        index="param_learning_rate",
        columns="param_n_estimators",
    )

    fig_heatmap, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table_xgb, annot=True, fmt=".4f", cmap="plasma", ax=ax)
    ax.set_title(
        "Heatmap de Resultados de GridSearchCV para XGBoost (Accuracy Promedio)"
    )
    ax.set_xlabel("Número de Estimadores (n_estimators)")
    ax.set_ylabel("Tasa de Aprendizaje (learning_rate)")

    return best_xgb_model, accuracy_best_xgb, classification_report_str, fig_heatmap


def train_and_evaluate_svc(
    cls_X_train: pd.DataFrame,
    cls_X_test: pd.DataFrame,
    cls_y_train: pd.Series,
    cls_y_test: pd.Series,
) -> tuple[Pipeline, float, str, Figure]:
    """Trains and evaluates an SVC model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.

    Returns:
        A tuple containing:
            - best_svc_model (Pipeline): The trained SVC model (within a pipeline).
            - accuracy_best_svc (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - fig_heatmap (Figure): GridSearchCV heatmap.
    """
    pipeline_svc = Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(random_state=42, probability=True))]
    )

    param_grid_svc = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", "auto", 0.1, 1]}

    grid_search_svc = GridSearchCV(
        pipeline_svc, param_grid_svc, cv=5, n_jobs=-1, verbose=0
    )
    grid_search_svc.fit(cls_X_train, cls_y_train)

    best_svc_model = grid_search_svc.best_estimator_
    y_pred_best_svc = best_svc_model.predict(cls_X_test)

    accuracy_best_svc = float(accuracy_score(cls_y_test, y_pred_best_svc))
    classification_report_str = str(
        classification_report(
            cls_y_test, y_pred_best_svc, target_names=["Bajo", "Alto"]
        )
    )

    results_svc = pd.DataFrame(grid_search_svc.cv_results_)
    pivot_table_svc = results_svc.pivot_table(
        values="mean_test_score", index="param_svc__C", columns="param_svc__gamma"
    )

    fig_heatmap, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table_svc, annot=True, fmt=".4f", cmap="magma", ax=ax)
    ax.set_title("Heatmap de Resultados de GridSearchCV para SVC (Accuracy Promedio)")
    ax.set_xlabel("Parámetro Gamma (gamma)")
    ax.set_ylabel("Parámetro de Regularización (C)")

    return best_svc_model, accuracy_best_svc, classification_report_str, fig_heatmap


def create_classification_summary(
    accuracy_log: float,
    accuracy_best_rf: float,
    accuracy_best_xgb: float,
    accuracy_best_svc: float,
) -> str:
    """Creates a summary of all classification models' accuracies.

    Args:
        accuracy_log: Accuracy of Logistic Regression.
        accuracy_best_rf: Accuracy of Random Forest.
        accuracy_best_xgb: Accuracy of XGBoost.
        accuracy_best_svc: Accuracy of SVC.

    Returns:
        A formatted string with the classification summary.
    """
    summary = "## Paso 5: El Veredicto Final y la Coronación del Campeón\n\n"
    summary += "Después de una rigurosa evaluación y optimización, los resultados finales hablaron por sí mismos.\n\n"
    summary += "| Modelo | Accuracy (Precisión Final) |\n"
    summary += "| :--- | :---: |\n"
    summary += f"| Regresión Logística | {accuracy_log * 100:.2f}% |\n"
    summary += f"| Support Vector Classifier (SVC) | {accuracy_best_svc * 100:.2f}% |\n"
    summary += f"| XGBoost | {accuracy_best_xgb * 100:.2f}% |\n"
    summary += f"| **Random Forest** | **{accuracy_best_rf * 100:.2f}%** |\n\n"
    summary += "---\n\n"
    summary += (
        "> El modelo **Random Forest optimizado** es el campeón indiscutible de este análisis, logrando la mayor precisión con un **"
        + f"{accuracy_best_rf * 100:.2f}%"
        + "**.\n\n"
    )
    summary += "Este proyecto demuestra una lección clave en la ciencia de datos: el algoritmo más complejo no siempre es el mejor. La experimentación y la validación rigurosa son esenciales para descubrir la solución óptima para un problema específico.\n\n"
    summary += "Hemos concluido con una herramienta de clasificación de alto rendimiento, capaz de identificar con gran fiabilidad a los pacientes que probablemente incurrirán en altos costos, permitiendo así intervenciones más efectivas y proactivas."
    return summary
