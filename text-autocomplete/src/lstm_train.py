import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import json
import os
from src.lstm_model import LSTMNextToken
from src.next_token_dataset import NextTokenDataset
from src.eval_lstm import evaluate_model

def train_lstm(config_path: str = "configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    df_train = pd.read_csv(config["data"]["train_path"], encoding="utf-8")
    df_val = pd.read_csv(config["data"]["val_path"], encoding="utf-8")

    if config["data"].get("debug", False):
        df_train = df_train.iloc[:config["data"]["debug_train_size"]]
        df_val = df_val.iloc[:config["data"]["debug_val_size"]]
        print(f"⚠️ Debug mode: train={len(df_train)}, val={len(df_val)}")

    train_dataset = NextTokenDataset(
        texts=df_train["cleaned_text"].tolist(),
        seq_len=config["model"]["seq_len"],
        max_vocab_size=config["model"]["max_vocab_size"]
    )
    val_dataset = NextTokenDataset(
        texts=df_val["cleaned_text"].tolist(),
        vocab=train_dataset.vocab,
        seq_len=config["model"]["seq_len"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    print(f"Размер словаря: {len(train_dataset.vocab)}")
    print(f"Обучающих последовательностей: {len(train_dataset)}")

    model = LSTMNextToken(
        vocab_size=len(train_dataset.vocab),
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"].get("dropout", 0.3)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    os.makedirs("models", exist_ok=True)
    model_save_path = "models/lstm_autocomplete_best.pth"
    vocab_save_path = "models/vocab.json"

    # История для графиков
    history = {"train_loss": [], "val_rouge1": [], "val_rouge2": [], "epochs": []}

    best_rouge1 = 0.0
    patience_counter = 0
    max_patience = config["training"].get("early_stopping_patience", 3)

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(train_loader, desc=f"Эпоха {epoch + 1}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        rouge_scores = evaluate_model(
            model=model,
            data_loader=val_loader,
            vocab=train_dataset.vocab,
            device=device,
            max_new_tokens=config["training"]["max_gen_tokens"],
            seq_len=config["model"]["seq_len"]
        )
        history["val_rouge1"].append(rouge_scores['rouge1'])
        history["val_rouge2"].append(rouge_scores['rouge2'])

        print(f"Эпоха {epoch+1} | Потери: {avg_train_loss:.4f} | ROUGE-1: {rouge_scores['rouge1']:.4f}")

        if rouge_scores['rouge1'] > best_rouge1:
            best_rouge1 = rouge_scores['rouge1']
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            with open(vocab_save_path, "w", encoding="utf-8") as f:
                json.dump(train_dataset.vocab, f, ensure_ascii=False)
            print(f"✅ Лучшая модель сохранена! ROUGE-1: {best_rouge1:.4f}")
        else:
            patience_counter += 1

        scheduler.step(rouge_scores['rouge1'])
        if patience_counter >= max_patience:
            print("⏹️ Early stopping.")
            break

    print("Обучение завершено.")
    return model, train_dataset.vocab, history