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
    data['reference_conditions'] = json.loads((DATA_DIR / 'reference_conditions.json').read_text())
    data['symptom_metadata'] = json.loads((DATA_DIR / 'symptom_metadata.json').read_text())
    data['pakistan_hospital_symptoms'] = json.loads((DATA_DIR / 'pakistan_hospital_symptoms.json').read_text())
    symptoms = {row['id']: row for row in data['symptoms']}
    from src.hospital_symptoms import validate_hospital_symptoms
    validate_hospital_symptoms(data['pakistan_hospital_symptoms'], symptoms)
    for row in data['pakistan_hospital_symptoms']:
        if row['feature_id']:
            aliases = symptoms[row['feature_id']]['aliases']
            if row['term'] not in aliases:
                aliases.append(row['term'])
    seen = set()
    for metadata in data['symptom_metadata']:
        key = metadata['id']
        weight = metadata['source_severity_weight']
        if key in seen or key not in symptoms or type(weight) is not int or not 1 <= weight <= 7:
            raise ValueError('Invalid or duplicate symptom metadata')
        if metadata['source_sha256'] != data['manifest']['source_sha256'].get(metadata['source_file']):
            raise ValueError('Symptom metadata source mismatch')
        seen.add(key)
        symptoms[key]['source_severity_weight'] = weight
    data['by_condition'] = {row['name']: row for row in data['conditions']}
    data['by_symptom'] = {row['id']: row for row in data['symptoms']}
    from src.health_library import load_health_library
    data['health_library'] = load_health_library()
    return data


def load_corrections():
    return json.loads((DATA_DIR / 'input_corrections.json').read_text())


def symptom_text(symptom_ids):
    labels = ', '.join(s.replace('_', ' ').strip() for s in sorted(symptom_ids))
    return f'Symptoms: {labels}.'
