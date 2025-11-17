import torch
from torch.utils.data import Dataset
from collections import Counter
import json

class NextTokenDataset(Dataset):
    def __init__(self, texts, vocab=None, seq_len=10, max_vocab_size=10000):
        self.seq_len = seq_len
        
        all_tokens = []
        filtered_texts = []
        for text in texts:
            tokens = [t for t in text.split() if t]
            if len(tokens) >= 2:
                all_tokens.extend(tokens)
                filtered_texts.append(tokens)
        
        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1}
            counter = Counter(all_tokens)
            for i, (word, _) in enumerate(counter.most_common(max_vocab_size - 2)):
                vocab[word] = i + 2
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        self.sequences = []
        for tokens in filtered_texts:
            if len(tokens) <= seq_len:
                continue
            for i in range(len(tokens) - seq_len):
                src = [self.vocab.get(t, 1) for t in tokens[i:i+seq_len]]
                tgt = [self.vocab.get(t, 1) for t in tokens[i+1:i+seq_len+1]]
                self.sequences.append((src, tgt))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        src, tgt = self.sequences[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

    def decode(self, ids):
        return " ".join([self.inv_vocab.get(int(i), "<unk>") for i in ids])
    
    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
    
    @staticmethod
    def load_vocab(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)