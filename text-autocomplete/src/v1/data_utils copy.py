import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """Очищает текст: приводит к нижнему регистру, удаляет ссылки, упоминания и спецсимволы."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)  # Удаляем упоминания (@username)
    text = re.sub(r"[^a-zа-я\s]", " ", text)  # Оставляем только буквы и пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean_data(raw_path: str) -> pd.DataFrame:
    """Загружает и очищает датасет sentiment140."""
    df = pd.read_csv(raw_path, encoding='latin1', header=None, 
 #                    names=["target", "ids", "date", "flag", "user", "text"])
                      names=["text"])
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 10]  # Удаляем слишком короткие тексты
    return df[["cleaned_text"]].reset_index(drop=True)

def split_and_save(df: pd.DataFrame, train_path: str, val_path: str, test_path: str):
    """Разбивает датасет на обучающую, валидационную и тестовую выборки и сохраняет в CSV."""
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)