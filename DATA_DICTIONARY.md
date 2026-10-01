## Dataset download
1. Download the original dataset from [this Google Drive link](https://drive.google.com/file/d/1VgiwsnLsH3z1kGQ_sTSn_21xAykGU79J/view?usp=sharing).
2. Name the file `dataset_hate_speech.csv`.
3. Place it exactly at the path: `data/raw/dataset_hate_speech.csv`.

## Dataset specifications
The dataset has been adapted for this activity. This adaptation included:

- Removal of nulls and duplicates
- Removal of URLs, emojis and mentions of the newspapers
- Removal of empty rows
- Data cleaning and homogenization.
    - Converting all text to lowercase
    - Removing punctuation marks
    - Removing numbers
    - Removing extra whitespace
    - Removing words shorter than 2 characters
    - Removing stopwords
    - Tokenization
    - Lemmatization
- Feature extraction process
    - Count of positive words (A)
    - Count of negative words (B)
    - Count of the most common bigrams (C)
    - Count of mentions of other users (D)
    - Sentiment category according to the Spanish ‘pysentimiento’ library (E)

- Standardization of the features (A_t,..E_t)
- Combination of features f1*fi (iA..iE) (Valor1,..Valor10).
