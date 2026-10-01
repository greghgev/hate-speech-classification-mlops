# Binary Classification Project: From EDA to Deployment

This repository documents the full lifecycle of a Machine Learning model for the **binary classification of hate messages**. The main focus of the project is to demonstrate analytical rigor in data handling, sound judgment in model evaluation and the application of engineering best practices geared towards its future move to production.

## Project Phases and Key Decisions

### 1. Exhaustive Exploratory Data Analysis (EDA)
An in-depth, structured and purposeful EDA was carried out. Instead of applying transformations blindly, the behavior of the variables, their distributions and the inter-class relationships were analyzed to guide preprocessing in a logical and mathematically grounded way.

### 2. Modeling and Algorithm Comparison
A linear model (Logistic Regression) was established as the initial *baseline* to compare it against a tree-based model (LightGBM). Both approaches yielded very good metrics.

### 3. Bayesian Optimization
To tune the final model, Bayesian Optimization was implemented using Optuna (TPE estimator).

### 4. Algorithmic Explainability (SHAP)
The model was put through a SHAP audit. This served to visually confirm that there was no *data leakage* (no variable was "cheating") and to clearly understand how the algorithm made its decisions based on the most important features.

### 5. Packaging and MLOps
The project has a deployment mindset. The **complete Pipeline** (model + preprocessing stages) has been serialized with `joblib`, leaving it packaged and ready to receive raw data in a production environment without raising errors.


## Repository Structure

* `notebooks`:
    * `01_Eda_Feature_Engineering.ipynb`: Exhaustive Exploratory Data Analysis (EDA) and feature engineering decisions.
    * `02_Seleccion_Modelos.ipynb`: Data isolation, construction of the modular Pipeline and comparison of base models (Logistic Regression vs LightGBM).
    * `03_Ajuste_Explicabilidad_Serializacion.ipynb`: Bayesian optimization with Optuna, explainability audit with SHAP and packaging of the artifact with Joblib.
* `src/`: Python modules (`.py`). Contains the encapsulated logic (e.g. loading and evaluation functions) to keep the notebook clean.
* `modelos_exportados/`: Contains the final artifact (`pipeline_lightgbm_produccion.joblib`) ready for inference.
* *Note: The original datasets are not included in the repository, following security best practices and to keep its size under control.*

## Technology Stack
* **Data Manipulation and Analysis:** Pandas, NumPy.
* **Preprocessing and Machine Learning:** Scikit-Learn (Pipelines, custom Transformers, Evaluation Metrics), LightGBM (Gradient Boosting).
* **Hyperparameter Optimization:** Optuna.
* **Algorithmic Explainability and Visualization:** SHAP, Matplotlib, Seaborn.
* **MLOps and Serialization:** Joblib.
