# src/data_utils.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """Очищает текст: приводит к нижнему регистру, удаляет ссылки, упоминания и спецсимволы."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zа-я\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean_data(raw_path: str, min_length: int = 10) -> pd.DataFrame:
    """
    Загружает файл ПОСТРОЧНО: каждая строка файла = один твит.
    Игнорирует запятые, кавычки и структуру CSV — никаких ParserError.
    """
    tweets = []
    encodings = ["utf-8", "latin1", "cp1252"]
    
    for enc in encodings:
        try:
            with open(raw_path, "r", encoding=enc) as f:
                for line in f:
                    tweet = line.rstrip("\n\r")  # удаляем только конец строки
                    if tweet:  # пропускаем пустые строки
                        tweets.append(tweet)
            print(f"✅ Успешно прочитано {len(tweets)} строк с кодировкой '{enc}'.")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Не удалось прочитать {raw_path} ни с одной из кодировок: {encodings}")
    
    # Создаём DataFrame и очищаем
    df = pd.DataFrame({"text": tweets})
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > min_length].reset_index(drop=True)
    
    print(f"Загружено {len(df)} очищенных твитов (min_length={min_length}).")
    return df[["cleaned_text"]]

def split_and_save(
    df: pd.DataFrame,
    train_path: str,
    val_path: str,
    test_path: str
) -> None:
    """Разбивает датасет и сохраняет выборки в UTF-8."""
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(train_path, index=False, encoding="utf-8")
    val.to_csv(val_path, index=False, encoding="utf-8")
    test.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"✅ Saved splits:\n  Train: {len(train)}\n  Val: {len(val)}\n  Test: {len(test)}")