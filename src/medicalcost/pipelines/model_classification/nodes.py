import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def split_classification_data(
    featured_classification_data: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos de clasificación en conjuntos de entrenamiento y prueba.

    Args:
        featured_classification_data: DataFrame preprocesado para clasificación.
        parameters: Diccionario de parámetros con `test_size` y `random_state`.

    Returns:
        Una tupla que contiene:
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


def train_logistic_regression(
    cls_X_train: pd.DataFrame, cls_y_train: pd.Series, parameters: dict
) -> LogisticRegression:
    """Entrena un modelo de Regresión Logística."""
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(cls_X_train, cls_y_train)
    return log_model


def train_random_forest(
    cls_X_train: pd.DataFrame, cls_y_train: pd.Series, parameters: dict
) -> GridSearchCV:
    """Entrena y optimiza un modelo Random Forest usando GridSearchCV."""
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=parameters["param_grid"],
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
    )
    grid_search_rf.fit(cls_X_train, cls_y_train)
    return grid_search_rf


def train_xgboost(
    cls_X_train: pd.DataFrame, cls_y_train: pd.Series, parameters: dict
) -> GridSearchCV:
    """Entrena y optimiza un modelo XGBoost usando GridSearchCV."""
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid=parameters["param_grid"],
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
    )
    grid_search_xgb.fit(cls_X_train, cls_y_train)
    return grid_search_xgb


def train_svc(
    cls_X_train: pd.DataFrame, cls_y_train: pd.Series, parameters: dict
) -> GridSearchCV:
    """Entrena y optimiza un modelo SVC usando GridSearchCV."""
    pipeline_svc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=42, probability=True)),
        ]
    )
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_svc = GridSearchCV(
        pipeline_svc, parameters["param_grid"], cv=cv_strategy, n_jobs=-1, verbose=1
    )
    grid_search_svc.fit(cls_X_train, cls_y_train)
    return grid_search_svc


def predict(model, cls_X_test: pd.DataFrame) -> pd.Series:
    """Realiza predicciones sobre el conjunto de prueba."""
    y_pred = model.predict(cls_X_test)
    return pd.Series(y_pred, index=cls_X_test.index)


def evaluate_classifier(
    model, cls_y_test: pd.Series, y_pred: pd.Series
) -> tuple[dict, str]:
    """Evalúa un modelo de clasificación."""
    accuracy = accuracy_score(cls_y_test, y_pred)
    report = classification_report(cls_y_test, y_pred, target_names=["Bajo", "Alto"])
    return {"accuracy": accuracy}, str(report)


def extract_and_plot_log_reg_importance(
    model: LogisticRegression, X_train: pd.DataFrame
) -> tuple[pd.DataFrame, Figure]:
    """Extrae y grafica la importancia de características de una Regresión Logística."""
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Coefficient": coefficients}
    )
    feature_importance = feature_importance.sort_values(
        by="Coefficient", ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    # Dibujar las barras con cierta transparencia
    sns.barplot(x="Coefficient", y="Feature", data=feature_importance, ax=ax, alpha=0.6)

    # Añadir una línea que conecte los puntos máximos de las barras
    ax.plot(
        feature_importance["Coefficient"],
        feature_importance["Feature"],
        marker="o",
        linestyle="-",
        color="darkred",
    )

    ax.set_title("Impacto de Características en la Clasificación de Costo")
    ax.set_xlabel('Coeficiente (Impacto en la probabilidad de ser "Alto Costo")')
    ax.set_ylabel("Característica")
    ax.axvline(0, color="black", lw=0.5)
    fig.tight_layout()
    plt.close(fig)
    return feature_importance, fig


def plot_roc_curves_comparison(
    log_reg_model,
    random_forest_model,
    xgboost_model,
    svc_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Figure:
    """
    Genera y compara las curvas ROC para múltiples modelos de clasificación.

    Args:
        log_reg_model: Modelo de Regresión Logística entrenado.
        random_forest_model: Modelo de Random Forest entrenado.
        xgboost_model: Modelo de XGBoost entrenado.
        svc_model: Modelo de SVC entrenado.
        X_test: Características del conjunto de prueba.
        y_test: Variable objetivo real del conjunto de prueba.

    Returns:
        Una figura de Matplotlib con las curvas ROC superpuestas.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    models = {
        "Regresión Logística": log_reg_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model,
        "SVC": svc_model,
    }

    for name, model in models.items():
        # Para GridSearchCV, primero obtenemos el mejor estimador
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_

        # Obtener las probabilidades de la clase positiva
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calcular la curva ROC y el área bajo la curva (AUC)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Dibujar la curva
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # Dibujar la línea de referencia (azar)
    ax.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.50)")

    ax.set_title("Comparación de Curvas ROC para Modelos de Clasificación", fontsize=16)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_grid_search_heatmap(
    grid_search_model: GridSearchCV, x_param: str, y_param: str, model_name: str
) -> Figure:
    """Crea un heatmap a partir de los resultados de GridSearchCV."""
    results = pd.DataFrame(grid_search_model.cv_results_)
    pivot_table = results.pivot_table(
        values="mean_test_score", index=f"param_{y_param}", columns=f"param_{x_param}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis", ax=ax)
    ax.set_title(f"GridSearchCV Heatmap - {model_name}")
    ax.set_xlabel(x_param.replace("svc__", ""))
    ax.set_ylabel(y_param.replace("svc__", ""))
    fig.tight_layout()
    plt.close(fig)
    return fig


def create_classification_summary(
    accuracy_log: dict, accuracy_rf: dict, accuracy_xgb: dict, accuracy_svc: dict
) -> str:
    """Crea un resumen en texto comparando la precisión de los modelos."""
    results = {
        "Regresión Logística": accuracy_log["accuracy"],
        "Support Vector Classifier (SVC)": accuracy_svc["accuracy"],
        "XGBoost": accuracy_xgb["accuracy"],
        "Random Forest": accuracy_rf["accuracy"],
    }

    # Ordenar los resultados por precisión de mayor a menor
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

    best_model_name = sorted_results[0][0]
    best_model_accuracy = sorted_results[0][1]

    table_header = "| Modelo | Accuracy (Precisión Final) |\n| :--- | :---: |\n"
    table_rows = ""
    for name, acc in sorted_results:
        if name == best_model_name:
            table_rows += f"| **{name}** | **{acc * 100:.2f}%** |\n"
        else:
            table_rows += f"| {name} | {acc * 100:.2f}% |\n"

    champion_line = f"> El modelo **{best_model_name} optimizado** es el campeón indiscutible de este análisis, logrando la mayor precisión."

    summary = f"## Resumen de Rendimiento de Modelos de Clasificación\n\n{table_header}{table_rows}\n---\n\n{champion_line}"

    return summary.strip()
