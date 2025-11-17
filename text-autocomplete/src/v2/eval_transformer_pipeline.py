import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_distilgpt2(
    test_csv_path: str,
    seq_len: int = 10,
    max_new_tokens: int = 8,
    num_samples: int = 200,
    device: int = 0 if torch.cuda.is_available() else -1
):
    """
    Оценивает DistilGPT2 на задаче автозавершения текста.
    
    Параметры:
        test_csv_path: путь к test.csv
        seq_len: длина контекста (должна совпадать с LSTM)
        max_new_tokens: сколько токенов генерировать
        num_samples: сколько примеров использовать
        device: 0 для GPU, -1 для CPU
    """
    # Загрузка данных
    df = pd.read_csv(test_csv_path, encoding="utf-8")
    texts = df["cleaned_text"].tolist()
    
    # Инициализация модели и токенизатора
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 не имеет pad_token по умолчанию
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    predictions = []
    references = []

    processed = 0
    for text in tqdm(texts, desc="Оценка DistilGPT2"):
        if processed >= num_samples:
            break
            
        words = text.split()
        if len(words) < seq_len + 1:  # нужно хотя бы seq_len + 1 слово
            continue

        # Берём контекст и реальное продолжение
        context_words = words[:seq_len]
        reference_words = words[seq_len:seq_len + max_new_tokens]
        
        if not reference_words:
            continue
            
        prompt = " ".join(context_words)
        ref_text = " ".join(reference_words).strip()

        try:
            # Генерация (greedy decoding = do_sample=False, num_beams=1)
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding, как в LSTM
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False  # возвращаем ТОЛЬКО сгенерированное
            )
            
            pred_text = result[0]["generated_text"].strip()
            
            if pred_text and ref_text:
                predictions.append(pred_text)
                references.append(ref_text)
                processed += 1
                
        except Exception as e:
            continue  # пропускаем проблемные примеры

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0}

    # Вычисление ROUGE
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
    rouge2 = sum(s['rouge2'].fmeasure for s in scores) / len(scores)
    
    return {"rouge1": rouge1, "rouge2": rouge2}