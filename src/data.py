"""JSON knowledge base. Raw CSVs are read only by scripts/prepare_json.py."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
ARTIFACT_DIR = ROOT / 'artifacts/chatbot'
JSON_FILES = ['training.json', 'conditions.json', 'symptoms.json', 'manifest.json']


def fingerprint():
    return hashlib.sha256(b''.join((DATA_DIR / name).read_bytes() for name in JSON_FILES)).hexdigest()


def load_data():
    data = {key: json.loads((DATA_DIR / f'{key}.json').read_text())
            for key in ['training', 'conditions', 'symptoms', 'manifest']}
    if len({row['id'] for row in data['training']}) != len(data['training']):
        raise ValueError('Duplicate training record IDs')
    for name, expected in data['manifest']['json_sha256'].items():
        actual = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'{name} changed. Run python scripts/prepare_json.py, then retrain.')
    data['by_condition'] = {row['name']: row for row in data['conditions']}
    data['by_symptom'] = {row['id']: row for row in data['symptoms']}
    return data


def load_corrections():
    return json.loads((DATA_DIR / 'input_corrections.json').read_text())


def symptom_text(symptom_ids):
    labels = ', '.join(s.replace('_', ' ').strip() for s in sorted(symptom_ids))
    return f'Symptoms: {labels}.'
