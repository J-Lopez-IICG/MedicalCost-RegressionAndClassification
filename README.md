<h1 align="center">🏥 Medical Cost Prediction 🏥</h1>

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## 🎯 Visión General

Este proyecto Kedro implementa un pipeline de ciencia de datos de extremo a extremo para predecir los costos de seguros médicos y clasificar a los pacientes en categorías de costo. La solución utiliza el conjunto de datos "Medical Insurance Cost Dataset", disponible en [Kaggle: Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset), que contiene información demográfica y de salud de individuos. El pipeline ingiere estos datos crudos, los procesa para garantizar su calidad, entrena y evalúa modelos de regresión para predecir costos exactos, y modelos de clasificación para predecir si un paciente incurrirá en costos "Altos" o "Bajos".

El objetivo es demostrar un flujo de trabajo de Machine Learning estructurado y reproducible, donde cada paso, desde la limpieza de datos hasta la generación de reportes, está encapsulado en un pipeline modular y robusto.

---

## 🎯 Hipótesis

La hipótesis central es que las características demográficas y de salud de un individuo no solo permiten predecir sus costos médicos, sino también clasificarlo en un grupo de riesgo con alta precisión. Para validar esto, se plantearon las siguientes sub-hipótesis:

1.  **Hipótesis de Regresión (Predicción de Costo):**
    *   **¿Es posible predecir el costo exacto del seguro (`charges`)?** Se postula que un modelo de regresión podrá explicar una porción significativa de la varianza en los costos (R² > 0.75).
    *   **¿Cuál es el factor más influyente?** Se hipotetiza que ser fumador (`smoker`) será, por un amplio margen, el predictor más determinante del costo, superando a la edad y al IMC.
    *   **¿Existen efectos de interacción?** Se espera encontrar una fuerte interacción entre ser fumador y el IMC, donde el impacto del IMC en los costos se magnifica exponencialmente en individuos fumadores.

2.  **Hipótesis de Clasificación (Categorización de Riesgo):**
    *   **¿Se puede clasificar a los pacientes en categorías de costo 'Alto' o 'Bajo' con alta precisión?** Se anticipa que los modelos de clasificación alcanzarán una precisión superior al 90%.
    *   **¿Qué tipo de modelo será más efectivo?** Dada la complejidad y las interacciones no lineales (como la de `smoker` y `bmi`), se hipotetiza que los modelos de ensamblaje (Random Forest, XGBoost) superarán en rendimiento a los modelos lineales (Regresión Logística) y a otros clasificadores como SVC.

---
## 🏗️ Estructura del Proyecto

El proyecto está organizado en una serie de pipelines modulares, cada uno con una responsabilidad específica, garantizando un flujo de trabajo claro y reproducible.

```mermaid
graph TD
    A[1. data_engineering] --> B[2. data_processing];
    B --> C[3. exploratory_data];
    B --> D[4. feature_engineering];
    D --> E[5. model_regression];
    D --> F[6. model_classification];
```

```text
src/medicalcost/pipelines/
├── data_engineering/     # 1. 📥 Descarga y carga de datos crudos desde Kaggle.
├── data_processing/      # 2. 🧼 Limpieza y validación de datos (nulos, duplicados).
├── exploratory_data/     # 3. 🗺️ Generación de gráficos para el Análisis Exploratorio de Datos (EDA).
├── feature_engineering/  # 4. 🛠️ Creación de características para modelado (One-Hot Encoding, etc.).
├── model_regression/     # 5. 📈 Entrenamiento y evaluación de modelos de Regresión (Lineal, RF, XGBoost).
└── model_classification/ # 6. 📊 Entrenamiento y evaluación de modelos de Clasificación (Log-Reg, SVC, RF, XGBoost).
```
</div>

---

## ⚙️ Flujo de Preparación de Datos

El preprocesamiento de datos es un pilar fundamental de este proyecto, automatizado a través de una secuencia de pipelines de Kedro para garantizar la consistencia y reproducibilidad. El flujo es el siguiente:

1.  **Ingesta de Datos (`data_engineering`)**:
    *   El pipeline se conecta a la API de Kaggle para descargar y cargar el dataset crudo, asegurando que siempre se trabaje con la fuente de datos original.

2.  **Limpieza y Validación (`data_processing`)**:
    *   **Manejo de Nulos y Duplicados**: Se eliminan sistemáticamente todas las filas que contienen valores nulos o que son duplicados exactos, garantizando la integridad del dataset.
    *   **Conversión de Tipos**: Las columnas `sex`, `smoker` y `region` se convierten al tipo de dato `category` para optimizar el uso de memoria y prepararlas para la codificación.

3.  **Ingeniería de Características (`feature_engineering`)**:
    *   **Creación de Variable Objetivo (Clasificación)**: Se crea la columna `cost_category` para los modelos de clasificación. Un paciente se etiqueta como `1` (Alto costo) si sus `charges` superan la mediana del dataset, y `0` (Bajo costo) en caso contrario.
    *   **Codificación One-Hot**: Las variables categóricas (`sex`, `smoker`, `region`) se transforman en formato numérico usando One-Hot Encoding con `drop_first=True` para evitar multicolinealidad.
    *   **Manejo de Outliers**: Se toma la decisión explícita de **no eliminar outliers**. Los valores extremos, especialmente en `charges` para fumadores con alto IMC, son considerados información predictiva crucial y no ruido.

---

## 💡 Resultados: Una Historia en Tres Actos

El pipeline generó una serie de reportes y visualizaciones que, en conjunto, cuentan la historia de los datos y el rendimiento de los modelos.

### Acto 1: Exploración de los Datos

El análisis exploratorio inicial (EDA) reveló patrones clave que sentaron las bases para las hipótesis del proyecto.

1.  **Ser Fumador es el Factor Decisivo**: El primer hallazgo contundente fue la abismal diferencia en costos entre fumadores y no fumadores. Los fumadores no solo pagan más, sino que la variabilidad de sus costos es inmensa.

    <img src="data/08_reporting/exploratory/smoker_vs_charges.png" alt="Smoker vs Charges" width="600"/>

2.  **La Interacción Exponencial entre IMC y Fumar**: El análisis de interacciones demostró que, si bien un IMC alto aumenta los costos para todos, este efecto se magnifica exponencialmente en individuos fumadores. Esto sugiere que los modelos no lineales serían más efectivos.

    <img src="data/08_reporting/exploratory/bmi_smoker_interaction.png" alt="BMI Smoker Interaction" width="700"/>

### Acto 2: Predicción del Costo Exacto (Regresión)

El objetivo aquí era responder: **¿Podemos predecir el costo exacto del seguro?** Se compararon tres modelos, y los resultados confirmaron que los modelos de ensamblaje superaron con creces al modelo lineal.

<img src="data/08_reporting/regression/r2_comparison_plot.png" alt="R2 Comparison" width="700"/>

El modelo **XGBoost Regressor** se coronó como el campeón, explicando un **90.25%** de la varianza en los costos del seguro en el conjunto de prueba.

La importancia de las características del modelo ganador confirmó la hipótesis inicial de forma rotunda:

| Característica    | Importancia |
| :---------------- | :---------- |
| **smoker_yes**    | **0.8307**  |
| bmi               | 0.0991      |
| age               | 0.0440      |
| children          | 0.0108      |
| sex_male          | 0.0047      |
| region_southwest  | 0.0047      |
| region_northwest  | 0.0036      |
| region_southeast  | 0.0023      |

> ✅ **Conclusión de Regresión**: Es posible predecir los costos con alta precisión (R² > 0.90), y ser fumador (`smoker_yes`) es, por un margen abrumador, el factor más determinante.

### Acto 3: Clasificación del Riesgo de Costo (Clasificación)

Finalmente, se buscó responder: **¿Podemos clasificar a los pacientes en categorías de 'Alto' o 'Bajo' costo?** Los resultados fueron excelentes, superando el 90% de precisión anticipado.

A continuación se muestra el resumen de rendimiento de los modelos:

| Modelo                          | Accuracy (Precisión Final) |
| :------------------------------ | :------------------------: |
| **XGBoost**                     |         **94.78%**         |
| Random Forest                   |           94.78%           |
| Support Vector Classifier (SVC) |           92.91%           |
| Regresión Logística             |           90.67%           |

Los modelos **XGBoost** y **Random Forest** demostraron un rendimiento prácticamente idéntico y superior, validando la hipótesis de que los modelos de ensamblaje serían los más efectivos.

La comparación de las curvas ROC confirma visualmente el rendimiento superior de los modelos de ensamblaje, con áreas bajo la curva (AUC) de 0.99 para RF y 0.95 para XGBoost, indicando una capacidad de discriminación casi perfecta.

<img src="data/08_reporting/classification/roc_curves_comparison.png" alt="ROC Curves" width="700"/>

> ✅ **Conclusión de Clasificación**: Es posible clasificar a los pacientes por riesgo de costo con una precisión extremadamente alta (≈95%), y los modelos de ensamblaje son la mejor herramienta para esta tarea.

---

## � Configuración de Kaggle

Para poder ejecutar este pipeline, es necesario configurar las credenciales de la API de Kaggle.

1.  **Crea un token de API en Kaggle:**
    *   Ve a tu perfil de Kaggle y entra en la sección "Settings": [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
    *   En la sección "API", haz clic en **"Create New Token"**. Se descargará un archivo llamado `kaggle.json`.

2.  **Coloca el archivo de credenciales:**
    *   Mueve el archivo `kaggle.json` a la carpeta `.kaggle` dentro de tu directorio de usuario.
        *   En Windows: `C:\Users\<Tu-Usuario>\.kaggle\kaggle.json`
        *   En Linux/macOS: `~/.kaggle/kaggle.json`

Una vez completados estos pasos, el pipeline podrá autenticarse con Kaggle para descargar los datos necesarios.

---

## 🚀 Instalación y Ejecución

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local. Se requiere Python 3.11.9.

### 1. Clonar el Repositorio

Primero, clona este repositorio.

```bash
git clone https://github.com/J-Lopez-IICG/MedicalCostKedro.git
cd MedicalCostKedro-WorkInProgress
```

### 2. Crear y Activar un Entorno Virtual

Es una práctica recomendada utilizar un entorno virtual para aislar las dependencias del proyecto.

```bash
# Crear el entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar en macOS/Linux
# source venv/bin/activate
```

### 3. Instalar Dependencias

Una vez que el entorno virtual esté activado, instala todas las librerías necesarias.

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el Pipeline

Con las dependencias instaladas, puedes ejecutar el pipeline completo con un solo comando.

```sh
kedro run
```

Esto ejecutará todos los nodos en secuencia, generando los datos limpios, los modelos entrenados y las gráficas de resultados en la carpeta `data/`.

---

## 📓 Desarrollo con Notebooks

La carpeta `notebooks` contiene los Jupyter Notebooks utilizados durante la fase de exploración y desarrollo.

Para trabajar con ellos de forma interactiva dentro del contexto de Kedro, ejecuta:

```bash
kedro jupyter lab
# o también
kedro jupyter notebook
```

> **Nota**: Al usar estos comandos, Kedro inicia el notebook con las variables `context`, `session`, `catalog` y `pipelines` ya cargadas, facilitando la interacción con los datos y funciones del proyecto.
