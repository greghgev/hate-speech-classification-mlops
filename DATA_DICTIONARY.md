## Descarga del dataset
1. Descarga el dataset original desde [este enlace de Google Drive](https://drive.google.com/file/d/1VgiwsnLsH3z1kGQ_sTSn_21xAykGU79J/view?usp=sharing).
2. Nombra el archivo como `dataset_hate_speech.csv`.
3. Colócalo exactamente en la ruta: `data/raw/dataset_hate_speech.csv`.

## Especificaciones del dataset
El dataset ha sido adaptado para la realización de esta actividad. Esa adaptación ha incluido:

- Eliminación de nulos y duplicados
- Eliminación de URLs, emojis y menciones a los periódicos
- Eliminación de filas vacías
- Limpieza y homogeneización de datos.
    - Convertir la totalidad del texto a minúscula
    - Eliminar signos de puntuación
    - Eliminar números
    - Eliminar espacios en blanco adicionales
    - Eliminar palabras con longitud menor a 2 caracteres
    - Eliminar stopwords
    - Tokenización
    - Lematización
- Proceso de extracción de características
    - Conteo de palabras positivas (A)
    - Conteo de palabras negativas (B)
    - Conteo del número de bigrams más comunes (C)
    - Conteo del número de menciones a otros usuarios (D)
    - Categoría del sentimiento según librería ‘pysentimiento’ en español (E)

- Estandarización de las características (A_t,..E_t)
- Combinación de características f1*fi (iA..iE) (Valor1,..Valor10).