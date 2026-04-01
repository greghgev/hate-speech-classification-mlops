import pandas as pd
from typing import List
import numpy as np
from typing import Tuple


def load_and_audit_data(filepath: str):
    """
    Carga el dataset, optimiza tipos de datos y realiza una auditoría de calidad
    buscando nulos lógicos y duplicados (crítico para prevenir data leakage en NLP).
    """
    # Carga defensiva (Fail-fast paradigm)
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Shape inicial: {df.shape[0]} filas | {df.shape[1]} columnas.")
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {filepath} no existe en la ruta.")

    # Optimización de Memoria (Downcasting)
    if 'label' in df.columns:
        df['label'] = df['label'].astype('int8')

    # Detección de duplicados (Data Leakage check)
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        print(f"⚠️ ALERTA: {duplicados} filas duplicadas detectadas. Esto sesgará la validación.")
    else:
        print("✅ Integridad: 0 filas duplicadas.")

    # Generación de tabla de auditoría (incluyendo nulos encubiertos)
    audit_df = pd.DataFrame({
        "Dtype": df.dtypes,
        "Nulos_NaN": df.isnull().sum(),
        "Nulos_Vacios": (df == " ").sum() + (df == "").sum(), # Busca strings vacíos (típico en NLP)
        "Unicos": df.nunique()
    })
    
    return df, audit_df



def auditar_calidad_target(df: pd.DataFrame, target_col: str) -> Tuple[float, int]:
    """
    Realiza una auditoría avanzada de la variable objetivo calculando el 
    baseline determinista (Zero Rule) y detectando ruido en las etiquetas 
    (contradicciones en el espacio de características).
    """
    print(f"--- Auditoría Avanzada del Target: '{target_col}' ---")
    
    # 1. Cálculo del Baseline Determinista (Zero Rule)
    # Es el Accuracy que obtendrías si el modelo fuera una "roca" y predijera siempre la clase mayoritaria
    proporciones = df[target_col].value_counts(normalize=True) * 100
    baseline_acc = proporciones.max()
    clase_mayoritaria = proporciones.idxmax()
    
    print(f"🎯 Baseline (Zero Rule): {baseline_acc:.2f}%")
    print(f"   (Cualquier modelo de ML debe superar este Accuracy prediciendo la clase {clase_mayoritaria})")
    
    # 2. Detección de Label Noise (Contradicciones Lógicas)
    # Buscamos si existen filas con idénticas características (X) pero distinta etiqueta (y)
    features = [col for col in df.columns if col != target_col]
    
    # Agrupamos por todas las features y contamos cuántas etiquetas distintas tienen
    ruido_df = df.groupby(features)[target_col].nunique().reset_index()
    contradicciones = ruido_df[ruido_df[target_col] > 1].shape[0]
    
    if contradicciones > 0:
        print(f"⚠️ RUIDO DETECTADO (Label Noise): Hay {contradicciones} vectores de características idénticos con etiquetas opuestas.")
        print("   Esto establece un límite superior al rendimiento (Error de Bayes > 0). El modelo nunca alcanzará 100% de Accuracy.")
    else:
        print("✅ Integridad Lógica: 0 contradicciones detectadas entre características y etiquetas.")
        
    print("-" * 50)
    
    return baseline_acc, contradicciones




def auditar_esparsidad(df: pd.DataFrame, features: List[str], umbral_alerta: float = 50.0) -> pd.DataFrame:
    """
    Evalúa la proporción de ceros (esparsidad) en las características numéricas.
    
    Devuelve un DataFrame ordenado para integrarse en pipelines de validación 
    de datos o logs de experimentación.
    """
    # Cálculo vectorizado de la proporción
    prop_ceros = (df[features] == 0).mean() * 100
    
    # Estructuración en formato tabular
    df_esparsidad = pd.DataFrame({
        'Caracteristica': prop_ceros.index,
        'Ceros_Pct': prop_ceros.values
    })
    
    # Filtrado (solo variables con ceros) y ordenamiento
    df_esparsidad = df_esparsidad[df_esparsidad['Ceros_Pct'] > 0]
    df_esparsidad = df_esparsidad.sort_values(by='Ceros_Pct', ascending=False).reset_index(drop=True)
    
    # Alerta temprana para variables críticas
    variables_criticas = df_esparsidad[df_esparsidad['Ceros_Pct'] >= umbral_alerta]['Caracteristica'].tolist()
    if variables_criticas:
        print(f"WARNING: Variables altamente esparsas (>{umbral_alerta}% ceros): {variables_criticas}")
        
    return df_esparsidad