import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_model(model, data_loader, vocab, device, max_new_tokens=8, seq_len=10):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(data_loader, desc="Оценка LSTM"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            generated = model.generate(
                input_ids=src,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_k=50
            )
            gen_continuation = generated[:, seq_len:]
            ref_continuation = tgt[:, -max_new_tokens:]

            for i in range(src.size(0)):
                pred_text = " ".join([
                    inv_vocab.get(int(tok), "<unk>") 
                    for tok in gen_continuation[i].cpu().tolist()
                ]).strip()
                ref_text = " ".join([
                    inv_vocab.get(int(tok), "<unk>") 
                    for tok in ref_continuation[i].cpu().tolist()
                ]).strip()
                
                if pred_text and ref_text:
                    predictions.append(pred_text)
                    references.append(ref_text)

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0}

    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
    rouge2 = sum(s['rouge2'].fmeasure for s in scores) / len(scores)
    
    return {"rouge1": rouge1, "rouge2": rouge2}