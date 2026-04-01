# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_validate

def auditar_modelo(pipeline, X, y, cv, metricas, nombre_modelo="Modelo"):
    """
    Ejecuta validación cruzada, calcula métricas y audita el sobreajuste.
    Diseñado para ser agnóstico al estimador (Lineal o Árboles).
    """
    print(f"Entrenando {nombre_modelo} con Validación Cruzada...")
    
    # Ejecutamos cross_validate forzando return_train_score=True para la auditoría
    cv_results = cross_validate(
        pipeline, 
        X, 
        y, 
        cv=cv, 
        scoring=metricas, 
        n_jobs=-1, 
        return_train_score=True
    )
    
    # Extraemos las métricas medias necesarias para los prints
    auc_val = cv_results['test_roc_auc'].mean()
    auc_train = cv_results['train_roc_auc'].mean()
    acc_val = cv_results['test_accuracy'].mean()
    acc_train = cv_results['train_accuracy'].mean()
    
    # --- INICIO DE TU BLOQUE DE IMPRESIÓN ---
    print("\n-------------------------------")
    print(f"MÉTRICAS DE VALIDACIÓN CRUZADA: {nombre_modelo.upper()}")
    print("-------------------------------")

    print(f"- ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f} (+/- {cv_results['test_roc_auc'].std():.4f})")
    print(f"- Accuracy : {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
    print(f"- F1-Macro : {cv_results['test_f1_macro'].mean():.4f} (+/- {cv_results['test_f1_macro'].std():.4f})")

    print("\n--------------------------------------")
    print("AUDITORÍA DE SOBREAJUSTE (TRAIN vs VAL)")
    print("--------------------------------------")
    print(f"- Accuracy Gap : Train ({acc_train:.4f}) vs Val ({acc_val:.4f}) -> Diferencia: {(acc_train - acc_val):.4f}")
    print(f"- ROC-AUC Gap  : Train ({auc_train:.4f}) vs Val ({auc_val:.4f}) -> Diferencia: {(auc_train - auc_val):.4f}")
    # --- FIN DE TU BLOQUE DE IMPRESIÓN ---
    
    return cv_results


import optuna
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import warnings

def optimizar_lightgbm(X, y, preprocesador, cv, n_trials=30):
    """
    Orquesta la optimización bayesiana de hiperparámetros aislando la lógica del notebook.
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Closure: Definimos el objective DENTRO de la función para que herede X, y, preprocesador
    def objective(trial):
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 700, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        pipeline_trial = Pipeline(steps=[
            ('preprocesador', preprocesador), 
            ('modelo', LGBMClassifier(**param_grid))
        ])
        
        cv_results = cross_validate(
            pipeline_trial, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        return cv_results['test_score'].mean()

    print("Iniciando Optimización Bayesiana con Optuna (TPE)...")

    # MLOps: Silenciamos los logs por iteración para mantener la consola limpia
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # MLOps: Congelamos el motor estocástico para reproducibilidad estricta
    sampler_determinista = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(direction='maximize', study_name="LGBM_Tuning", sampler=sampler_determinista)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone

def graficar_curva_aprendizaje_profesional(X, y, preprocesador, params, test_size=0.2, random_state=42):
    """
    Visualización de Diagnóstico Avanzado para Convergencia de Modelos (Nivel Senior).
    Calcula automáticamente el punto óptimo de Early Stopping, sombrea el Generalization Gap,
    y anota métricas clave en el lienzo para toma de decisiones MLOps inmediata.
    """
    print("Iniciando entorno de diagnóstico...")
    
    # 1. Configuración Estética (Senior Level Aesthetics)
    # Usamos un tema limpio y fuentes modernas para legibilidad en reportes
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Roboto', 'DejaVu Sans']
    
    # 2. Aislamiento y Preparación de Datos (Cero Data Leakage)
    prep_clonado = clone(preprocesador)
    X_t, X_v, y_t, y_v = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_t_trans = prep_clonado.fit_transform(X_t)
    X_v_trans = prep_clonado.transform(X_v)
    
    # 3. Configuración del Estimador (Horizonte Expandido)
    params_monitor = params.copy()
    params_monitor['n_estimators'] = 900  # Provisionamos de sobra para ver la asíntota
    params_monitor['verbose'] = -1
    
    modelo = LGBMClassifier(**params_monitor)
    
    # 4. Entrenamiento y Captura de Gradientes
    print("Calculando gradientes iterativos...")
    modelo.fit(
        X_t_trans, y_t,
        eval_set=[(X_t_trans, y_t), (X_v_trans, y_v)],
        eval_names=['Train', 'Validacion'],
        eval_metric='auc'
    )
    
    evals_result = modelo.evals_result_
    train_auc = np.array(evals_result['Train']['auc'])
    valid_auc = np.array(evals_result['Validacion']['auc'])
    iterations = np.arange(len(train_auc))
    
    # ==============================================================================
    # 5. CÁLCULOS MATEMÁTICOS DE DIAGNÓSTICO (Actionable Intelligence)
    # ==============================================================================
    # Encontramos el punto donde la métrica de validación se maximiza (Early Stopping point)
    opt_idx = np.argmax(valid_auc)
    opt_trees = iterations[opt_idx] + 1  # Ajustamos índice 0
    max_auc_val = valid_auc[opt_idx]
    
    # Calculamos el Gap de Generalización en el punto óptimo
    gap_optimo = train_auc[opt_idx] - max_auc_val
    
    # ==============================================================================
    # 6. RENDERIZADO DEL LIENZO (Senior Visualization Design)
    # ==============================================================================
    fig, ax = plt.figure(figsize=(10, 6), dpi=100), plt.gca()
    
    # A. Dibujamos las Curvas Base con grosores y alphas diferenciados
    ax.plot(iterations, train_auc, label='Curva de Entrenamiento (Sesgo)', linewidth=2.5, color='#1f77b4', alpha=0.8)
    ax.plot(iterations, valid_auc, label='Curva de Validación (Varianza)', linewidth=3, color='#ff7f0e')
    
    # B. Sombreamos el Generalization Gap (Visualización del riesgo de sobreajuste)
    # Esto muestra intuitivamente la magnitud del error de generalización
    ax.fill_between(iterations, valid_auc, train_auc, color='#1f77b4', alpha=0.1, label='Generalization Gap (Overfitting Area)')
    
    # C. RESALTADO DEL PUNTO DE DECISIÓN (The "Go/No-Go" Line)
    # Dibujamos una línea vertical roja gruesa en el óptimo matemático
    ax.axvline(x=opt_trees-1, color='#d62728', linestyle='--', linewidth=2.5, label=f'Punto Óptimo: N={opt_trees}')
    # Marcamos el punto exacto con un 'médico' blanco para contraste
    ax.scatter(opt_trees-1, max_auc_val, color='white', edgecolor='#d62728', s=100, zorder=5)
    
    # D. ANOTACIONES EN EL LIENZO (Metadata-Driven Visualization)
    # Añadimos un cuadro de texto con las métricas clave para que no haga falta leer el output
    texto_resumen = (
        f"MÉTRICAS CLAVE (EARLY STOPPING):\n"
        f"----------------------------------\n"
        f"Optimal N-Estimators : {opt_trees}\n"
        f"Max ROC-AUC (Val)    : {max_auc_val:.4f}\n"
        f"ROC-AUC (Train @ N)  : {train_auc[opt_idx]:.4f}\n"
        f"Generalization Gap   : {gap_optimo:.4f}\n"
        f"----------------------------------\n"
        f"Recomendación MLOps  : Fijar N={opt_trees}"
    )
    
    # Posicionamos el cuadro en la zona inferior derecha (típicamente vacía en estas curvas)
    ax.text(0.97, 0.05, texto_resumen, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='0.8', alpha=0.9))
    
    # E. AJUSTES FINALES DE FORMATO Y LEGEND
    ax.set_title('DIAGNÓSTICO DE CONVERGENCIA TOPOLÓGICA: Análisis de Early Stopping', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Iteraciones de Boosting (Número de Árboles)', fontsize=13, labelpad=10)
    ax.set_ylabel('ROC-AUC Score (Capacidad Discriminativa)', fontsize=13, labelpad=10)
    
    # Ajustamos ticks para porcentajes si el rango es pequeño (opcional)
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x*100))))
    
    # Leyenda limpia en la parte superior izquierda
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=False, facecolor='white')
    
    # Eliminamos bordes innecesarios (despine) para un look moderno
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.show()
    
    # Devolvemos el valor óptimo para usarlo programáticamente si se desea
    return opt_trees