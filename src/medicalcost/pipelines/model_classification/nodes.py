import pandas as pd
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


def split_classification_data(
    featured_classification_data: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the preprocessed classification data into training and testing sets.

    Args:
        featured_classification_data: The preprocessed DataFrame for classification.
        parameters: Parameters dictionary containing test_size and random_state.

    Returns:
        A tuple containing:
            - cls_X_train (DataFrame): Training features.
            - cls_X_test (DataFrame): Testing features.
            - cls_y_train (Series): Training target.
            - cls_y_test (Series): Testing target.
    """
    # 'cost_category' is now the target, 'charges' is dropped as it's redundant.
    X = featured_classification_data.drop(["charges", "cost_category"], axis=1)
    y = featured_classification_data["cost_category"]

    cls_X_train, cls_X_test, cls_y_train, cls_y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )
    return cls_X_train, cls_X_test, cls_y_train, cls_y_test


def train_and_evaluate_logistic_regression(
    cls_X_train: pd.DataFrame,
    cls_X_test: pd.DataFrame,
    cls_y_train: pd.Series,
    cls_y_test: pd.Series,
) -> tuple[LogisticRegression, float, str, pd.DataFrame, Figure]:
    """Trains and evaluates a Logistic Regression model.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.

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
        {"Feature": cls_X_train.columns, "Coefficient": coefficients}
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
    parameters: dict,
) -> tuple[RandomForestClassifier, float, str, Figure]:
    """Trains and evaluates a Random Forest model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.
        parameters: Dictionary with model parameters (e.g., param_grid).

    Returns:
        A tuple containing:
            - best_rf_model (RandomForestClassifier): The trained Random Forest model.
            - accuracy_best_rf (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - fig_heatmap (Figure): GridSearchCV heatmap.
    """
    # Definir la estrategia de validación cruzada para que sea reproducible
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=parameters["param_grid"],
        cv=cv_strategy,
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
    parameters: dict,
) -> tuple[XGBClassifier, float, str, Figure]:
    """Trains and evaluates an XGBoost model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.
        parameters: Dictionary with model parameters (e.g., param_grid).

    Returns:
        A tuple containing:
            - best_xgb_model (XGBClassifier): The trained XGBoost model.
            - accuracy_best_xgb (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - fig_heatmap (Figure): GridSearchCV heatmap.
    """
    # Definir la estrategia de validación cruzada para que sea reproducible
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        ),
        param_grid=parameters["param_grid"],
        cv=cv_strategy,
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
    parameters: dict,
) -> tuple[Pipeline, float, str, Figure]:
    """Trains and evaluates an SVC model using GridSearchCV.

    Args:
        cls_X_train: Training features.
        cls_X_test: Testing features.
        cls_y_train: Actual training targets.
        cls_y_test: Actual testing targets.
        parameters: Dictionary with model parameters (e.g., param_grid).

    Returns:
        A tuple containing:
            - best_svc_model (Pipeline): The trained SVC model (within a pipeline).
            - accuracy_best_svc (float): Accuracy score.
            - classification_report_str (str): Classification report.
            - fig_heatmap (Figure): GridSearchCV heatmap.
    """
    pipeline_svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=42, probability=True)),
        ]  # probability=True is needed for some metrics
    )

    # Definir la estrategia de validación cruzada para que sea reproducible
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search_svc = GridSearchCV(
        pipeline_svc, parameters["param_grid"], cv=cv_strategy, n_jobs=-1, verbose=0
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
