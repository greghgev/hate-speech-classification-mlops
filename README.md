# Proyecto de Clasificación Binaria: Del EDA al Despliegue

Este repositorio documenta el ciclo de vida completo de un modelo de Machine Learning para **clasificación binaria de mensajes de odio**. El enfoque principal del proyecto es demostrar rigor analítico en el tratamiento de los datos, criterio en la evaluación de modelos y la aplicación de buenas prácticas de ingeniería orientadas a su futuro paso a producción.

## Fases del Proyecto y Decisiones Clave

### 1. Análisis Exploratorio de Datos (EDA) Exhaustivo
Se realizó un EDA profundo, estructurado y con sentido. En lugar de aplicar transformaciones a ciegas, se analizó el comportamiento de las variables, sus distribuciones y las relaciones inter-clase para guiar el preprocesamiento de forma lógica y matemáticamente fundamentada.

### 2. Modelado y Comparativa de Algoritmos
Se estableció un modelo lineal (Regresión Logística) como *baseline* inicial para compararlo contra un modelo basado en árboles (LightGBM). Ambos enfoques arrojaron métricas muy buenas.

### 3. Optimización Bayesiana
Para ajustar el modelo final, se implementó una Optimización Bayesiana utilizando Optuna (estimador TPE).

### 4. Explicabilidad Algorítmica (SHAP)
Se sometió el modelo a una auditoría con SHAP. Esto sirvió para comprobar de forma visual que no había *data leakage* (ninguna variable estaba "haciendo trampa") y entender de manera clara cómo el algoritmo tomaba sus decisiones basándose en las características más importantes.

### 5. Empaquetado y MLOps
El proyecto tiene mentalidad de despliegue. Se ha serializado con `joblib` el **Pipeline completo** (modelo + fases de preprocesamiento), dejándolo empaquetado y listo para recibir datos crudos en un entorno de producción sin generar errores.


## Estructura del Repositorio

* `notebooks`:
    * `01_Eda_Feature_Engineering.ipynb`: Análisis Exploratorio de Datos (EDA) exhaustivo y decisiones de ingeniería de características.
    * `02_Seleccion_Modelos.ipynb`: Aislamiento de datos, construcción del Pipeline modular y comparativa de modelos base (Regresión Logística vs LightGBM).
    * `03_Ajuste_Explicabilidad_Serializacion.ipynb`: Optimización bayesiana con Optuna, auditoría de explicabilidad con SHAP y empaquetado del artefacto con Joblib.
* `src/`: Módulos de Python (`.py`). Contiene la lógica encapsulada (ej. funciones de carga y evaluación) para mantener el notebook limpio.
* `modelos_exportados/`: Contiene el artefacto final (`pipeline_produccion.joblib`) listo para inferencia.

*Nota: Los datasets originales no se incluyen en el repositorio por buenas prácticas de seguridad y control de peso.*

## Stack Tecnológico
* **Manipulación y Análisis de Datos:** Pandas, NumPy.
* **Preprocesamiento y Machine Learning:** Scikit-Learn (Pipelines, Transformers personalizados, Métricas de Evaluación), LightGBM (Gradient Boosting).
* **Optimización de Hiperparámetros:** Optuna.
* **Explicabilidad Algorítmica y Visualización:** SHAP, Matplotlib, Seaborn.
* **MLOps y Serialización:** Joblib.
