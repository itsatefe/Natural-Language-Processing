import hazm
import pandas as pd
from sklearn.model_selection import train_test_split
from taaghche.text_cleaning import normalize_text

def load_raw(csv_path) -> pd.DataFrame:
    data = pd.read_csv(csv_path, encoding='utf-8')[['comment', 'rate']]
    data['rate'] = data['rate'].apply(lambda r: r if r < 6 else None)
    data = data.dropna(subset=['rate', 'comment'])
    data = data.drop_duplicates(subset=['comment'], keep='first').reset_index(drop=True)
    return data

def add_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df['comment_len_by_words'] = df['comment'].apply(lambda t: len(hazm.word_tokenize(str(t))))
    return df

def filter_lengths(df: pd.DataFrame, min_words: int, max_words: int) -> pd.DataFrame:
    df['comment_len_by_words'] = df['comment_len_by_words'].apply(
        lambda n: n if min_words < n <= max_words else None
    )
    return df.dropna(subset=['comment_len_by_words']).reset_index(drop=True)

def labelize(df: pd.DataFrame, threshold: float):
    def rate_to_label(rate: float) -> str:
        return 'negative' if rate <= threshold else 'positive'
    df['label'] = df['rate'].apply(rate_to_label)
    labels = sorted(df['label'].unique())
    return df, labels

def clean_texts(df: pd.DataFrame, min_words: int, max_words: int) -> pd.DataFrame:
    df['comment'] = df['comment'].apply(normalize_text)
    df['comment_len_by_words'] = df['comment'].apply(lambda t: len(hazm.word_tokenize(t)))
    df['comment_len_by_words'] = df['comment_len_by_words'].apply(
        lambda n: n if min_words < n <= max_words else None
    )
    return df.dropna(subset=['comment_len_by_words']).reset_index(drop=True)[['comment', 'label']]

def balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    target = min(len(df[df.label == 'negative']), len(df[df.label == 'positive']))
    groups = [g.sample(n=target, random_state=seed).reset_index(drop=True) for _, g in df.groupby('label')]
    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)

def split(df: pd.DataFrame, labels, seed: int):
    df['label_id'] = df['label'].apply(lambda t: labels.index(t))
    train, test = train_test_split(df, test_size=0.1, random_state=seed, stratify=df['label'])
    train, valid = train_test_split(train, test_size=0.1, random_state=seed, stratify=train['label'])
    for part in (train, valid, test):
        part.reset_index(drop=True, inplace=True)
    return train, valid, test

def prepare_dataset(csv_path, min_words, max_words, threshold, balance_data, seed):
    data = load_raw(csv_path)
    data = add_lengths(data)
    data = filter_lengths(data, min_words, max_words)
    data, labels = labelize(data, threshold)
    data = clean_texts(data, min_words, max_words)
    if balance_data:
        data = balance(data, seed)
    train, valid, test = split(data.copy(), labels, seed)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}
    return train, valid, test, labels, label2id, id2label
