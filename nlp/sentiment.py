# nlp/sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_tok = None
_model = None

def _ensure():
    global _tok, _model
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def score_sentiment(text: str):
    if not text:
        return "neutral", 0.0
    _ensure()
    inputs = _tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,  # you can set 128, 256, or 512 depending on your tradeoff
        padding="max_length"
    )
    with torch.no_grad():
        out = _model(**inputs).logits
        probs = torch.softmax(out, dim=-1).tolist()[0]
    idx = int(max(range(3), key=lambda i: probs[i]))
    label_map = {0:"negative", 1:"neutral", 2:"positive"}
    return label_map.get(idx, "neutral"), float(probs[idx])
