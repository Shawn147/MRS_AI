"""Convert the supervisor's CSV bundle to versioned, traceable JSON. No model fitting."""
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'references/supervisor/dataset'
OUTPUT = ROOT / 'data'
REVISION = 'aa6b145d4838ac1dd6fd21b4db7a43c3098e19f9'


def read(name):
    with (SOURCE / name).open(encoding='utf-8-sig', newline='') as f:
        if name != 'Training.csv':
            return list(csv.DictReader(f))
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for values in reader:
            if len(values) != len(header):
                raise ValueError('Training row has the wrong number of columns')
            row = {}
            for key, value in zip(header, values):
                if key in row:
                    # Source has fluid_overload twice. Preserve either positive flag.
                    if row[key] not in ('0', '1') or value not in ('0', '1'):
                        raise ValueError('Duplicate non-binary column')
                    row[key] = str(max(int(row[key]), int(value)))
                else:
                    row[key] = value
            rows.append(row)
        return rows


def clean(value):
    text = re.sub(r'\s+', ' ', str(value).strip())
    return {'Peptic ulcer diseae': 'Peptic ulcer disease'}.get(text, text)


def list_field(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError('Expected a list of strings')
    return list(dict.fromkeys(clean(x) for x in parsed if clean(x)))


def validate_condition_tables(diseases):
    for filename, key in [('description.csv', 'Disease'), ('medications.csv', 'Disease'),
                          ('diets.csv', 'Disease'), ('precautions_df.csv', 'Disease')]:
        names = [clean(row[key]) for row in read(filename)]
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate condition in {filename}')
        if set(names) != set(diseases):
            raise ValueError(f'Condition coverage mismatch in {filename}')


def main():
    rows = read('Training.csv')
    features = [k for k in rows[0] if k != 'prognosis' and not k.startswith('Unnamed')]
    patterns = {}
    for index, row in enumerate(rows):
        if any(row[k] not in ('0', '1') for k in features):
            raise ValueError(f'Non-binary training value in row {index + 2}')
        symptoms = tuple(sorted(k for k in features if row[k] == '1'))
        disease = clean(row['prognosis'])
        if not symptoms or not disease:
            raise ValueError('Empty training record')
        if symptoms in patterns and patterns[symptoms]['condition'] != disease:
            raise ValueError('Conflicting labels for identical symptom patterns')
        if symptoms not in patterns:
            patterns[symptoms] = {'id': hashlib.sha256('|'.join(symptoms).encode()).hexdigest()[:16],
                                 'condition': disease, 'symptoms': list(symptoms), 'source_rows': []}
        patterns[symptoms]['source_rows'].append(index + 2)
    validate_condition_tables({r['condition'] for r in patterns.values()})
    descriptions = {clean(r['Disease']): clean(r['Description']) for r in read('description.csv')}
    medications = {clean(r['Disease']): list_field(r['Medication']) for r in read('medications.csv')}
    diets = {clean(r['Disease']): list_field(r['Diet']) for r in read('diets.csv')}
    precautions = {clean(r['Disease']): [clean(r[f'Precaution_{i}']) for i in range(1, 5)
                    if clean(r[f'Precaution_{i}'])] for r in read('precautions_df.csv')}
    workouts = {}
    for row in read('workout_df.csv'):
        workouts.setdefault(clean(row['disease']), []).append(clean(row['workout']))
    conditions = []
    for disease in sorted({r['condition'] for r in patterns.values()}):
        counts = Counter(s for r in patterns.values() if r['condition'] == disease for s in r['symptoms'])
        conditions.append({'name': disease, 'description': descriptions[disease],
                           'medications': medications[disease], 'diet': diets[disease],
                           'precautions': precautions[disease], 'workout': workouts.get(disease, []),
                           'symptoms': [s for s, _ in counts.most_common()],
                           'medicine_order': 'source order; effectiveness and suitability not assessed',
                           'clinically_reviewed': False})
    severity = {clean(r['Symptom']): int(r['weight']) for r in read('Symptom-severity.csv')}
    # Source labels are preserved as IDs; aliases only improve text input matching.
    aliases = {
        'high_fever': ['fever', 'high temperature'], 'mild_fever': ['mild fever', 'low grade fever'],
        'muscle_pain': ['body aches', 'body pain', 'muscle aches'],
        'continuous_sneezing': ['sneezing'], 'runny_nose': ['runny nose', 'running nose'],
        'congestion': ['blocked nose', 'nasal congestion', 'stuffy nose'],
        'throat_irritation': ['sore throat', 'scratchy throat'],
        'fatigue': ['tiredness', 'tired', 'exhausted'], 'headache': ['head ache'],
        'vomiting': ['throwing up'], 'diarrhoea': ['diarrhea', 'loose stools'],
        'skin_rash': ['rash'], 'itching': ['itchy skin'],
        'watering_from_eyes': ['watery eyes'], 'visual_disturbances': ['light sensitivity', 'sensitivity to light'],
        'stomach_pain': ['stomach ache'], 'burning_micturition': ['burning urination', 'painful urination'],
        'acidity': ['heartburn', 'stomach burning'], 'dizziness': ['dizzy'],
        'breathlessness': ['shortness of breath', 'trouble breathing', 'difficulty breathing'],
        'dischromic _patches': ['discolored patches'], 'spotting_ urination': ['spotting urination'],
    }
    symptom_records = [{'id': key, 'label': clean(key.replace('_', ' ')),
                        'aliases': sorted(set([clean(key.replace('_', ' '))] + aliases.get(key, []))),
                        'source_severity_weight': severity.get(key)} for key in features]
    files = {'training.json': list(patterns.values()), 'conditions.json': conditions, 'symptoms.json': symptom_records}
    manifest = {'schema_version': 1, 'repository': 'https://github.com/dr-mushtaq/Medicine-Recommendation-System',
                'revision': REVISION, 'raw_rows': len(rows), 'unique_rows': len(patterns),
                'duplicates_removed': len(rows)-len(patterns), 'condition_count': len(conditions),
                'feature_count': len(features), 'source_sha256': {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.glob('*.csv'))},
                'notes': ['No patient-level identifiers in training; independence of patients cannot be established.',
                          'CSV medicine mappings are unreviewed educational reference data, not prescribing rules.',
                          'Severity weights are stored for provenance, not used as clinical triage thresholds.']}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    from scripts.expand_dataset import apply_expansion
    apply_expansion(files, manifest)
    for name, value in files.items():
        (OUTPUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    manifest['json_sha256'] = {name: hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in files}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Additive metadata keeps original model fingerprints and source IDs stable.
    source_ids = {'spotting_ urination': 'spotting_urination',
                  'foul_smell_of urine': 'foul_smell_ofurine',
                  'dischromic _patches': 'dischromic_patches'}
    metadata = [{'id': key, 'source_symptom': source_id,
                 'source_severity_weight': severity[source_id],
                 'source_file': 'Symptom-severity.csv',
                 'source_sha256': manifest['source_sha256']['Symptom-severity.csv'],
                 'reason': 'Explicit match for inconsistent whitespace in source IDs'}
                for key, source_id in source_ids.items()]
    (OUTPUT/'symptom_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['raw_rows','unique_rows','duplicates_removed','condition_count','feature_count']}))

if __name__ == '__main__':
    main()
