"""Pretrained MiniLM encoder, mean pooling, and a learned condition head."""
import json
import os
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import joblib
import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.data import ARTIFACT_DIR, fingerprint, symptom_text

BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BASE_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MAX_LENGTH = 128


class ConditionTransformer(nn.Module):
    def __init__(self, encoder, class_count):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, class_count)

    def embed(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, inputs):
        return self.head(self.embed(inputs))


def load_bundle(directory=ARTIFACT_DIR):
    directory = Path(directory)
    metadata = json.loads((directory / 'metrics.json').read_text())
    if metadata['data_fingerprint'] != fingerprint():
        raise ValueError('JSON data changed. Run python scripts/train_chatbot.py to rebuild models.')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(directory / 'transformer', local_files_only=True)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory / 'transformer', local_files_only=True))
    model = ConditionTransformer(encoder, len(metadata['labels']))
    model.load_state_dict(load_file(str(directory / 'transformer/model.safetensors')))
    model.eval()
    return {
        'transformer': model,
        'tokenizer': tokenizer,
        'metrics': metadata,
        'ml': joblib.load(directory / 'baseline.joblib'),
    }


def predict(bundle, symptoms, model_key='transformer'):
    text = symptom_text(symptoms)
    if model_key == 'ml':
        probabilities = bundle['ml'].predict_proba([text])[0]
        labels = bundle['ml'].classes_
    elif model_key == 'transformer':
        inputs = bundle['tokenizer'](
            [text], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt'
        )
        with torch.inference_mode():
            probabilities = bundle['transformer'](inputs).softmax(-1)[0].numpy()
        labels = bundle['metrics']['labels']
    else:
        raise ValueError('Unknown model selection')
    order = np.argsort(probabilities)[::-1][:3]
    return [{'condition': str(labels[i]), 'probability': float(probabilities[i])} for i in order]
