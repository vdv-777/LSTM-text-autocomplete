import torch
import torch.nn as nn

class LSTMNextToken(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

    def generate(self, input_ids, max_new_tokens=10, temperature=1.0, top_k=None):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            input_seq = input_ids.clone().to(device)
            for _ in range(max_new_tokens):
                logits = self(input_seq)
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_seq = torch.cat([input_seq, next_token], dim=1)
        return input_seq