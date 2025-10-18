import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json


def feature_engineering_for_classification(
    featured_medical_data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepares the data for classification by selecting features.
    This node now passes the data through, as feature engineering (dummies)
    is already done in a previous pipeline. It mainly ensures the 'charges'
    column is present for the next step.

    Args:
        featured_medical_data: The featured medical insurance data (with dummy variables).

    Returns:
        The DataFrame ready for splitting.
    """
    return featured_medical_data.copy()


def create_target_variable(
    cls_y_train: pd.Series, cls_y_test: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Creates the binary target variable based on the median of the training set
    to prevent data leakage.

    Args:
        cls_y_train: The 'charges' Series for the training set.
        cls_y_test: The 'charges' Series for the test set.

    Returns:
        A tuple containing:
            - cls_y_train_binary (Series): Binary target for training.
            - cls_y_test_binary (Series): Binary target for testing.
    """
    # Calculate threshold ONLY from the training data
    cost_threshold = cls_y_train.median()

    # Apply threshold to both train and test sets
    cls_y_train_binary = (cls_y_train > cost_threshold).astype(int)
    cls_y_test_binary = (cls_y_test > cost_threshold).astype(int)

    return cls_y_train_binary, cls_y_test_binary


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
            - cls_y_train (Series): Actual training targets (charges).
            - cls_y_test (Series): Actual testing targets (charges).
            - X (DataFrame): All features used for training.
            - y (Series): All targets used for training (charges).
    """
    X = df_model_cls.drop("charges", axis=1)
    y = df_model_cls["charges"]
    # Stratify is not needed here as we are splitting based on X and continuous y
    # It will be applied on the binary target later if needed, but train_test_split
    # is robust enough for this.
    cls_X_train, cls_X_test, cls_y_train, cls_y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return cls_X_train, cls_X_test, cls_y_train, cls_y_test, X, y


def train_and_evaluate_logistic_regression(
    cls_X_train: pd.DataFrame,
    cls_X_test: pd.DataFrame,
    cls_y_train_binary: pd.Series,
    cls_y_test_binary: pd.Series,
    X: pd.DataFrame,
) -> tuple[LogisticRegression, float, str, pd.DataFrame, Figure]:
    """Trains and evaluates a Logistic Regression model.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train_binary: Actual training targets.
        cls_y_test_binary: Actual testing targets.
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
    log_model.fit(cls_X_train, cls_y_train_binary)
    y_pred_log = log_model.predict(cls_X_test)

    accuracy_log = float(accuracy_score(cls_y_test_binary, y_pred_log))
    classification_report_str = str(
        classification_report(
            cls_y_test_binary, y_pred_log, target_names=["Bajo", "Alto"]
        )
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
    cls_y_train_binary: pd.Series,
    cls_y_test_binary: pd.Series,
) -> tuple[RandomForestClassifier, float, str, Figure]:
    """Trains and evaluates a Random Forest model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train_binary: Actual training targets.
        cls_y_test_binary: Actual testing targets.

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

    # Definir la estrategia de validación cruzada para que sea reproducible
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
    )
    grid_search_rf.fit(cls_X_train, cls_y_train_binary)

    best_rf_model = grid_search_rf.best_estimator_
    y_pred_best_rf = best_rf_model.predict(cls_X_test)

    accuracy_best_rf = float(accuracy_score(cls_y_test_binary, y_pred_best_rf))
    classification_report_str = str(
        classification_report(
            cls_y_test_binary, y_pred_best_rf, target_names=["Bajo", "Alto"]
        )
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
    cls_y_train_binary: pd.Series,
    cls_y_test_binary: pd.Series,
) -> tuple[XGBClassifier, float, str, Figure]:
    """Trains and evaluates an XGBoost model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train_binary: Actual training targets.
        cls_y_test_binary: Actual testing targets.

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

    # Definir la estrategia de validación cruzada para que sea reproducible
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid=param_grid_xgb,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
    )
    grid_search_xgb.fit(cls_X_train, cls_y_train_binary)

    best_xgb_model = grid_search_xgb.best_estimator_
    y_pred_best_xgb = best_xgb_model.predict(cls_X_test)

    accuracy_best_xgb = float(accuracy_score(cls_y_test_binary, y_pred_best_xgb))
    classification_report_str = str(
        classification_report(
            cls_y_test_binary, y_pred_best_xgb, target_names=["Bajo", "Alto"]
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
    cls_y_train_binary: pd.Series,
    cls_y_test_binary: pd.Series,
) -> tuple[Pipeline, float, str, Figure]:
    """Trains and evaluates an SVC model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train_binary: Actual training targets.
        cls_y_test_binary: Actual testing targets.

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

    # Definir la estrategia de validación cruzada para que sea reproducible
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search_svc = GridSearchCV(
        pipeline_svc, param_grid_svc, cv=cv_strategy, n_jobs=-1, verbose=0
    )
    grid_search_svc.fit(cls_X_train, cls_y_train_binary)

    best_svc_model = grid_search_svc.best_estimator_
    y_pred_best_svc = best_svc_model.predict(cls_X_test)

    accuracy_best_svc = float(accuracy_score(cls_y_test_binary, y_pred_best_svc))
    classification_report_str = str(
        classification_report(
            cls_y_test_binary, y_pred_best_svc, target_names=["Bajo", "Alto"]
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
    # --- INICIO DE LA CORRECCIÓN ---

    # 1. Almacenar los resultados en un diccionario para facilitar la comparación
    results = {
        "Regresión Logística": accuracy_log,
        "Support Vector Classifier (SVC)": accuracy_best_svc,
        "XGBoost": accuracy_best_xgb,
        "Random Forest": accuracy_best_rf,
    }

    # 2. Encontrar el modelo con la mayor precisión
    best_model_name = max(results, key=lambda k: results[k])
    best_model_accuracy = results[best_model_name]

    # 3. Construir la tabla de resultados dinámicamente
    table_header = "| Modelo | Accuracy (Precisión Final) |\n| :--- | :---: |\n"
    table_rows = ""
    for name, acc in results.items():
        table_rows += f"| {name} | {acc * 100:.2f}% |\n"

    # 4. Construir el texto del veredicto con el ganador real
    champion_line = f"> El modelo **{best_model_name} optimizado** es el campeón indiscutible de este análisis, logrando la mayor precisión con un **{best_model_accuracy * 100:.2f}%**."

    # 5. Ensamblar el reporte completo
    summary = f"""## Paso 5: El Veredicto Final y la Coronación del Campeón

Después de una rigurosa evaluación y optimización, los resultados finales hablaron por sí mismos.

{table_header}{table_rows}
---

{champion_line}

Este proyecto demuestra una lección clave en la ciencia de datos: la experimentación y la validación rigurosa son esenciales para descubrir la solución óptima para un problema específico.

Hemos concluido con una herramienta de clasificación de alto rendimiento, capaz de identificar con gran fiabilidad a los pacientes que probablemente incurrirán en altos costos, permitiendo así intervenciones más efectivas y proactivas."""

    return summary.strip()
