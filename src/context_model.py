"""Locally trained follow-up intent head over frozen MiniLM embeddings."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/context_intents.json'
ARTIFACTS = ROOT / 'artifacts/context'


@lru_cache(maxsize=1)
def encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    directory = ROOT / 'artifacts/pretrained/minilm'
    model = AutoModel.from_pretrained(directory, local_files_only=True, use_safetensors=True)
    model.eval()
    return AutoTokenizer.from_pretrained(directory, local_files_only=True), model


def embed(texts):
    import numpy as np
    import torch
    tokenizer, model = encoder()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 32):
            inputs = tokenizer(texts[offset:offset + 32], padding=True, truncation=True, max_length=128, return_tensors='pt')
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            result.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
    return np.concatenate(result)


@lru_cache(maxsize=1)
def load_head():
    metadata = json.loads((ARTIFACTS / 'model.json').read_text())
    if metadata['dataset_sha256'] != hashlib.sha256(DATASET.read_bytes()).hexdigest():
        raise ValueError('Context intent training data changed; retrain context model.')
    return metadata


def classify_intent(text):
    import numpy as np
    metadata = load_head()
    vector = embed([text])[0]
    logits = np.asarray(metadata['weights']) @ vector + np.asarray(metadata['intercepts'])
    scores = np.exp(logits - logits.max())
    scores /= scores.sum()
    rank = np.argsort(scores)[::-1]
    confidence = float(scores[rank[0]])
    accepted = confidence >= metadata['threshold'] and scores[rank[0]] - scores[rank[1]] >= .15
    return metadata['labels'][rank[0]] if accepted else 'other'
