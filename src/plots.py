import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math


def plot_distribucion_target(df: pd.DataFrame, target_col: str):
    """
    Analiza y grafica la distribución de clases de la variable objetivo.
    Imprime la proporción exacta para evaluar el desbalanceo de clases.
    """
    # Cálculos estadísticos
    counts = df[target_col].value_counts(normalize=True) * 100
    
    print("--- Distribución de Clases ---")
    for label, pct in counts.items():
        print(f"Clase {label}: {pct:.2f}%")
    print("-" * 30)

    # Configuración visual (Tus colores encapsulados)
    colores = {0: "#4E8F88", 1: "#F0A84B"}
    bordes = {0: "#2F4F4F", 1: "#8A4F00"}

    plt.figure(figsize=(3, 3))
    
    ax = sns.countplot(
        x=target_col,
        hue=target_col,
        data=df,
        palette=colores,
        legend=False
    )

    # Aplicamos tus bordes personalizados
    for i, patch in enumerate(ax.patches):
        # Evitamos error si hay menos parches que colores esperados
        if i in bordes:
            patch.set_edgecolor(bordes[i])
            patch.set_linewidth(1.5)

    ax.set_ylabel("Frecuencia")
    ax.set_xlabel(f"Variable Objetivo: {target_col}")
    plt.title("Distribución de Clases", fontsize=10)
    
    # Desaturamos los bordes superiores y derechos para mayor limpieza (Buenas prácticas de visualización)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_estabilidad_secuencial(df: pd.DataFrame, target_col: str, window: int = 500):
    """
    Evalúa la estacionariedad de la variable objetivo a lo largo del dataset.
    Detecta si los datos están ordenados (Concept Drift sintético) lo cual 
    arruinaría un Train/Test Split que no sea aleatorio.
    """
    plt.figure(figsize=(8, 3))
    
    # Calculamos la media móvil (probabilidad empírica local de la clase 1)
    rolling_mean = df[target_col].rolling(window=window, center=True).mean()
    global_mean = df[target_col].mean()
    
    plt.plot(rolling_mean.index, rolling_mean, color="#3E5C76", linewidth=1.5, label=f"Media móvil (ventana={window})")
    
    # --- CORRECCIÓN APLICADA: 'fr' (Raw f-string) ---
    # Esto evita que el parser de Python intente leer '\m' como una secuencia de escape
    # y permite que Matplotlib reciba el comando LaTeX puro para renderizar el símbolo mu.
    plt.axhline(global_mean, color="red", linestyle="--", linewidth=1.5, label=fr"Media global ($\mu={global_mean:.3f}$)")
    
    plt.title("Estabilidad Secuencial de la Variable Objetivo", fontsize=12, pad=15)
    plt.xlabel("Índice de la fila (Secuencia temporal/extracción)")
    plt.ylabel(f"Proporción local de Clase 1")
    
    # Forzamos los límites de Y entre 0 y 1 porque es una probabilidad
    plt.ylim(0, 1)
    
    # Limpieza visual
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(alpha=0.2, linestyle="--")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()


def plot_distribucion_numericas(df, features, color_principal="#3E5C76"):
    """
    Genera un panel de Histogramas para analizar la distribución y 
    tendencia central (media/mediana) de variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataset a analizar.
        features (list): Lista de nombres de las variables a graficar.
        color_principal (str): Código hexadecimal del color a usar.
    """
    n_features = len(features)
    if n_features == 0:
        raise ValueError("La lista de variables (features) está vacía.")
        
    # Cuadrícula 1D adaptativa
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4), sharey=False)
    
    # Aseguramos que 'axes' sea siempre un array iterable, incluso con 1 sola feature
    axes = np.atleast_1d(axes)
        
    for i, var in enumerate(features):
        data = df[var].dropna()
        media = data.mean()
        mediana = data.median()
        
        ax = axes[i]
        ax.hist(
            data,
            bins=30,
            color=color_principal,
            edgecolor="white",
            alpha=0.85
        )
        
        ax.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Media: {media:.2f}")
        ax.axvline(mediana, color="blue", linestyle="-.", linewidth=1.5, label=f"Med: {mediana:.2f}")
        
        ax.set_title(f"Distribución: {var}", fontsize=11, fontweight='bold')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()



def plot_boxplot_numericas(df, features, color_principal="#3E5C76"):
    """
    Genera un panel de Boxplots para analizar la dispersión y 
    valores atípicos (outliers) de variables numéricas continuas.
    Nota: No recomendado para variables con Zero-Inflation masiva.
    
    Args:
        df (pd.DataFrame): Dataset a analizar.
        features (list): Lista de nombres de las variables a graficar.
        color_principal (str): Código hexadecimal del color a usar.
    """
    n_features = len(features)
    if n_features == 0:
        raise ValueError("La lista de variables (features) está vacía.")
        
    # Cuadrícula 1D con altura reducida para optimizar el ratio Data-Ink
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 3), sharey=False)
    
    axes = np.atleast_1d(axes)
        
    for i, var in enumerate(features):
        data = df[var].dropna()
        
        ax = axes[i]
        ax.boxplot(
            data,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor=color_principal, alpha=0.85),
            medianprops=dict(color="blue", linewidth=1.5),
            whiskerprops=dict(color=color_principal, linewidth=1.5),
            capprops=dict(color=color_principal, linewidth=1.5),
            flierprops=dict(marker='o', markersize=4, alpha=0.4, markerfacecolor="red", markeredgecolor="none")
        )
        
        ax.set_title(f"Dispersión (Boxplot): {var}", fontsize=11, fontweight='bold')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_yticks([])
        
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_densidad_predictiva(df: pd.DataFrame, features: list, target_col: str):
    """
    Genera gráficos de densidad (KDE) superpuestos segmentados por la clase objetivo
    en una disposición estrictamente horizontal (1 sola fila).
    Objetivo MLOps: Evaluar visualmente la divergencia entre distribuciones.
    """
    n_cols = len(features)
    
    # figsize dinámico: ancho proporcional al número de columnas (4.5 pulgadas por gráfica), alto fijo (4)
    # squeeze=False obliga a Matplotlib a devolver siempre un array 2D, evitando fallos si len(features) == 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    axes = axes.flatten() 
    
    colores = {0: "#4E8F88", 1: "#F0A84B"}
    
    for i, col in enumerate(features):
        sns.kdeplot(
            data=df, x=col, hue=target_col, fill=True, 
            palette=colores, ax=axes[i], common_norm=False, alpha=0.5
        )
        axes[i].set_title(f"Poder Discriminativo: {col}", fontsize=11, fontweight='bold')
        
        # Optimización visual: Solo mostramos la etiqueta "Densidad" en la primera gráfica de la izquierda
        axes[i].set_ylabel("Densidad" if i == 0 else "") 
        
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        
    plt.tight_layout()
    plt.show()


def plot_categorica_proporcional(df: pd.DataFrame, cat_col: str, target_col: str):
    """
    Genera un gráfico de barras apiladas al 100% (Probabilidad Condicionada).
    Objetivo MLOps: Evaluar si los diferentes niveles de una variable categórica 
    alteran significativamente la probabilidad base del target.
    """
    # Calculamos la tabla de contingencia normalizada por filas (suma 100% horizontal)
    crosstab = pd.crosstab(df[cat_col], df[target_col], normalize='index') * 100
    
    colores = ["#4E8F88", "#F0A84B"]
    
    ax = crosstab.plot(kind='bar', stacked=True, figsize=(6, 4), color=colores, edgecolor='black', width=0.7)
    
    # Línea de referencia del Baseline (Zero Rule) que calculamos antes (~52.5%)
    baseline = df[target_col].value_counts(normalize=True).max() * 100
    clase_mayoritaria = df[target_col].value_counts().idxmax()
    
    plt.axhline(baseline, color='red', linestyle='--', linewidth=1.5, 
                label=f'Baseline Clase {clase_mayoritaria} ({baseline:.1f}%)')
    
    plt.title(f"Probabilidad Condicionada del Target según '{cat_col}'", pad=15)
    plt.xlabel(f"Categorías de {cat_col}")
    plt.ylabel("Proporción (%)")
    
    # Forzamos Y a 100%
    plt.ylim(0, 100)
    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_matriz_correlacion(df: pd.DataFrame, features: list):
    """
    Genera un Heatmap triangular de correlación de Pearson.
    Objetivo MLOps: Detectar y auditar multicolinealidad severa (|r| > 0.85) 
    para purga de características redundantes. (Versión Compacta)
    """
    corr_matrix = df[features].corr(method='pearson')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 1. Reducimos el lienzo a 9x7 pulgadas (más compacto)
    plt.figure(figsize=(9, 7))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=.5, 
        cbar_kws={"shrink": .75},  # 2. Acortamos la barra lateral al 75%
        annot=True, fmt=".2f",
        annot_kws={"size": 8}      # 3. Reducimos la fuente para que encaje perfecta
    )
    
    plt.title("Auditoría de Multicolinealidad (Pearson)", pad=15, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()