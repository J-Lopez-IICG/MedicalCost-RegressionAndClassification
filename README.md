# Medical Cost Prediction

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Visión General

Este proyecto Kedro implementa un pipeline de ciencia de datos de extremo a extremo para predecir los costos de seguros médicos y clasificar a los pacientes en categorías de costo. La solución utiliza el conjunto de datos "Medical Insurance Cost Dataset", disponible en [Kaggle: Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset), que contiene información demográfica y de salud de individuos. El pipeline ingiere estos datos crudos, los procesa para garantizar su calidad, entrena y evalúa modelos de regresión para predecir costos exactos, y modelos de clasificación para predecir si un paciente incurrirá en costos "Altos" o "Bajos". Finalmente, evalúa y visualiza el rendimiento de todos los modelos.

El objetivo es demostrar un flujo de trabajo de Machine Learning estructurado y reproducible, donde cada paso, desde la limpieza de datos hasta la generación de reportes, está encapsulado en un pipeline modular y robusto.

---

## Hipótesis

La hipótesis central de este análisis es que **factores demográficos y de salud pueden ser utilizados para predecir los costos de seguros médicos y clasificar a los pacientes por riesgo de costo**. Se espera que, mediante algoritmos de regresión y clasificación, se puedan construir modelos capaces de estimar los costos y distinguir con alta precisión entre pacientes de "alto" y "bajo" costo, basándose en características como la edad, el IMC, el hábito de fumar, etc.

---
## Estructura del Proyecto

El proyecto se organiza en los siguientes pipelines principales:

*   **`data_processing`**: Se enfoca en la limpieza y preparación inicial de los datos. Toma los datos crudos, elimina filas con valores nulos y duplicados, y realiza la codificación de variables categóricas (One-Hot Encoding) para preparar los datos para ambos tipos de modelado (regresión y clasificación). Guarda los datos procesados listos para el modelado.
*   **`model_regression`**: Contiene la lógica para el modelado de regresión. Este pipeline:
    *   Divide los datos limpios en conjuntos de entrenamiento y prueba.
    *   Entrena un modelo de Regresión Lineal Múltiple para predecir los costos de seguros.
    *   Evalúa el modelo utilizando métricas como R-cuadrado.
    *   Genera gráficos de regresión univariada, interacción y correlación.
    *   Guarda el modelo entrenado y los resultados de la evaluación.
*   **`model_classification`**: Contiene la lógica para el modelado de clasificación. Este pipeline:
    *   Transforma el problema de regresión en uno de clasificación, creando una variable objetivo binaria ('Alto'/'Bajo' costo) basada en la mediana de los cargos.
    *   Divide los datos limpios en conjuntos de entrenamiento y prueba (estratificado).
    *   Entrena y ajusta hiperparámetros para múltiples modelos de clasificación (Regresión Logística, Random Forest, XGBoost, SVC), utilizando **GridSearchCV con validación cruzada K-Fold** para una evaluación robusta.
    *   Evalúa cada modelo utilizando métricas como Accuracy y Classification Report.
    *   Compara los modelos y selecciona el de mejor rendimiento.
    *   Guarda los modelos entrenados y los resultados de la evaluación.

---

## Resultados y Visualización

### Desafíos Técnicos Solucionados en la Preparación de Datos

*   **Manejo de Datos Nulos y Duplicados**: El pipeline identifica y elimina sistemáticamente registros con valores faltantes y filas duplicadas para asegurar la calidad del dataset.
*   **Codificación de Variables Categóricas**: Se transformaron variables como `sex`, `smoker` y `region` en formatos numéricos (One-Hot Encoding) para que los modelos pudieran procesarlas.
*   **Creación de Variable Objetivo Binaria para Clasificación**: Se transformó la variable `charges` en una variable binaria `cost_category` ('Alto'/'Bajo') para el problema de clasificación, utilizando la mediana como umbral.
*   **Automatización del Flujo de Datos**: Al encapsular todo el proceso de limpieza y preparación en pipelines de Kedro, se garantiza que los datos para el modelado sean completamente automatizados y reproducibles.

---

## Resultados del Modelo (Pipeline Kedro)

Esta sección presenta las conclusiones detalladas y los artefactos generados por los pipelines de Kedro, que encapsulan el proceso de modelado predictivo para la regresión de costos y la clasificación de categorías de costo. Hemos realizado un análisis comparativo exhaustivo de múltiples modelos, incluyendo un riguroso ajuste de hiperparámetros para los modelos de clasificación mediante **validación cruzada K-Fold (Stratified K-Fold)**.

Los pipelines de modelado y `reporting` se encargan de consolidar y visualizar estos resultados, proporcionando una visión clara del rendimiento de cada modelo y los factores clave que influyen en los costos de seguros médicos.

### Modelos de Regresión

El modelo de Regresión Lineal Múltiple entrenado en el pipeline `model_regression` obtuvo una precisión (R-cuadrado) de aproximadamente **0.7836**.

**Detalles del Modelo de Regresión Lineal:**

R-cuadrado del modelo de regresión: 0.7836
Coeficientes del modelo de regresión:
Intercept: -12050.84
age: 257.33
bmi: 337.31
children: 474.59
sex_male: -131.31
smoker_yes: 23600.54
region_northwest: -352.97
region_southeast: -1033.67
region_southwest: -959.69

**Impacto de cada variable en el costo (Regresión Lineal):**
*   **`smoker_yes`**: Es, por un margen enorme, el factor más determinante, aumentando el costo en más de $23,600.
*   **`age`** y **`bmi`**: Son los siguientes factores más importantes, aumentando el costo en ~$257 y ~$337 por cada unidad, respectivamente.
*   **`children`**: También tiene un impacto positivo notable.
*   **`sex_male`** y la **`region`**: Tienen un impacto mucho menor y, en algunos casos, negativo en comparación con la categoría de referencia.

### Modelos de Clasificación: Random Forest es el Modelo con Mejor Rendimiento

Tras el ajuste de hiperparámetros, el modelo **Random Forest** demostró ser el más efectivo para la clasificación de costos, obteniendo la mayor precisión:

| Modelo | Accuracy (Precisión Final) |
| :--- | :---: |
| Regresión Logística | 90.00% |
| Support Vector Classifier (SVC) | 92.54% |
| XGBoost | 92.91% |
| **Random Forest** | **94.03%** |

**Resultados Detallados de los Modelos de Clasificación:**

Resultados de los modelos de clasificación:
Regresión Logística - Accuracy: 0.9000
Support Vector Classifier (SVC) - Accuracy: 0.9254
XGBoost - Accuracy: 0.9291
Random Forest - Accuracy: 0.9403

Reporte de Clasificación (Random Forest):
              precision    recall  f1-score   support

        Bajo       0.94      0.94      0.94       662
        Alto       0.94      0.94      0.94       671

    accuracy                           0.94      1333
   macro avg       0.94      0.94      0.94      1333
weighted avg       0.94      0.94      0.94      1333

Este resultado sugiere que, para el problema de clasificación de costos médicos en este dataset, el Random Forest es el modelo más robusto y con mayor capacidad para distinguir entre pacientes de "alto" y "bajo" costo.

### Visualización Detallada del Rendimiento y Análisis Exploratorio

Los pipelines generan diversas visualizaciones para entender el comportamiento de los datos y el rendimiento de los modelos.

**Gráficos de Regresión Univariada y Correlación:**
*   **Regresión Lineal: Costos del Seguro vs. Edad**: Muestra una tendencia positiva, con datos agrupados en "bandas" (explicadas por el hábito de fumar).
    ![Regresión Lineal: Costos del Seguro vs. Edad](data/08_reporting/age_vs_charges.png)
*   **Regresión Lineal: Costos del Seguro vs. IMC (BMI)**: Relación positiva más débil y dispersa.
    ![Regresión Lineal: Costos del Seguro vs. IMC (BMI)](data/08_reporting/bmi_vs_charges.png)
*   **Distribución de Costos para Fumadores vs. No Fumadores**: Revela una diferencia masiva en costos, siendo el hábito de fumar un factor clave.
    ![Distribución de Costos para Fumadores vs. No Fumadores](data/08_reporting/smoker_vs_charges.png)
*   **Interacción entre IMC, ser Fumador y Costos del Seguro**: Muestra cómo el IMC impacta drásticamente los costos para fumadores, un claro efecto de interacción.
    ![Interacción entre IMC, ser Fumador y Costos del Seguro](data/08_reporting/bmi_smoker_interaction.png)
*   **Matriz de Correlación de Variables Numéricas**: Confirma las correlaciones entre `age`, `bmi` y `charges`.
    ![Matriz de Correlación de Variables Numéricas](data/08_reporting/correlation_heatmap.png)

**Gráficos de Clasificación y Ajuste de Hiperparámetros:**
*   **Importancia de las Características en el Modelo de Regresión Logística**: Muestra el impacto de cada variable en la clasificación.
    ![Importancia de las Características en el Modelo de Regresión Logística](data/08_reporting/log_reg_feature_importance.png)
*   **Heatmap de Resultados de GridSearchCV para Random Forest**: Visualiza el impacto de los hiperparámetros en la precisión del modelo Random Forest.
    ![Heatmap de Resultados de GridSearchCV para Random Forest (Accuracy Promedio)](data/08_reporting/rf_grid_search_heatmap.png)
*   **Heatmap de Resultados de GridSearchCV para XGBoost**: Visualiza el impacto de los hiperparámetros en la precisión del modelo XGBoost.
    ![Heatmap de Resultados de GridSearchCV para XGBoost (Accuracy Promedio)](data/08_reporting/xgb_grid_search_heatmap.png)
*   **Heatmap de Resultados de GridSearchCV para SVC**: Visualiza el impacto de los hiperparámetros en la precisión del modelo SVC.
    ![Heatmap de Resultados de GridSearchCV para SVC (Accuracy Promedio)](data/08_reporting/svc_grid_search_heatmap.png)

### La Importancia del Ajuste de Hiperparámetros

El proceso de ajuste de hiperparámetros mediante GridSearchCV, utilizando **validación cruzada K-Fold**, fue fundamental para optimizar el rendimiento de todos los modelos de clasificación. Se observaron mejoras significativas en la precisión para la mayoría de los modelos.

### Robustez de los Modelos de Ensamble

Como era de esperar, Random Forest y XGBoost, al ser algoritmos de ensamble, superaron consistentemente a la Regresión Logística y SVC en este problema. Los modelos de ensamble son inherentemente más robustos, reducen el sobreajuste y mejoran la capacidad de generalización.

### No hay un "Ganador" Universal

Aunque Random Forest demostró ser el mejor en este caso, esta observación resalta la importancia de:

*   **Evaluar múltiples algoritmos:** No hay un "mejor" algoritmo universal; el rendimiento óptimo depende de las características específicas del dataset.
*   **Ajuste exhaustivo:** Cada algoritmo requiere un ajuste cuidadoso de sus hiperparámetros para maximizar su potencial en un problema dado.

### Importancia de las Características

Todos los modelos proporcionaron información valiosa sobre la **importancia de las características**. Esta información es crucial para comprender qué factores (edad, IMC, fumador, etc.) son los más influyentes en los costos de seguros médicos. Estos conocimientos pueden ser utilizados para:

*   Refinar el modelo en futuras iteraciones.
*   Obtener insights sobre los factores de riesgo en seguros de salud.

---

**En resumen:** El **Random Forest** es el modelo recomendado para la clasificación de costos médicos en este proyecto, debido a su consistente y superior rendimiento en términos de precisión después de un riguroso ajuste de hiperparámetros. El análisis de regresión también confirmó la importancia crítica de factores como el hábito de fumar, la edad y el IMC en la determinación de los costos.

---

## Configuración

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

## Instalación y Ejecución

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Clonar el Repositorio

Primero, clona este repositorio en tu máquina.

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

```bash
kedro run
```

Esto ejecutará todos los nodos en secuencia, generando los datos limpios, los modelos entrenados y las gráficas de resultados en la carpeta `data/`.

---

## Desarrollo con Notebooks

La carpeta `notebooks` contiene los Jupyter Notebooks utilizados durante la fase de exploración y desarrollo.

Para trabajar con ellos de forma interactiva dentro del contexto de Kedro, ejecuta:

```bash
kedro jupyter lab
# o también
kedro jupyter notebook
```

> **Nota**: Al usar estos comandos, Kedro inicia el notebook con las variables `context`, `session`, `catalog` y `pipelines` ya cargadas, facilitando la interacción con los datos y funciones del proyecto.

## Reglas y Directrices

*   No elimines ninguna línea del archivo `.gitignore`.
*   No subas datos al repositorio (la carpeta `data/` está ignorada por defecto).
*   No subas credenciales o configuraciones locales. Mantenlas en la carpeta `conf/local/`.
