"""Validate the knowledge base and emit a reproducible quality report."""
import json
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_data


def validate(data):
    errors = []
    conditions = [r['name'] for r in data['conditions']]
    symptoms = [r['id'] for r in data['symptoms']]
    for label, values in [('condition', conditions), ('symptom', symptoms),
                          ('training ID', [r['id'] for r in data['training']])]:
        if len(values) != len(set(values)):
            errors.append(f'Duplicate {label}')
    seen = {}
    source_rows = []
    for row in data['training']:
        pattern = tuple(sorted(row['symptoms']))
        if not pattern or len(pattern) != len(set(pattern)):
            errors.append(f"Empty or repeated symptoms: {row['id']}")
        if set(pattern) - set(symptoms) or row['condition'] not in conditions:
            errors.append(f"Unknown reference: {row['id']}")
        if pattern in seen:
            errors.append(f"Duplicate or conflicting pattern: {row['id']}")
        seen[pattern] = row['condition']
        source_rows.extend(row['source_rows'])
    if sorted(source_rows) != list(range(2, data['manifest']['raw_rows'] + 2)):
        errors.append('Source-row provenance is incomplete or duplicated')
    for key, actual in [('unique_rows', len(seen)), ('condition_count', len(conditions)),
                        ('feature_count', len(symptoms)),
                        ('duplicates_removed', len(source_rows) - len(seen))]:
        if data['manifest'][key] != actual:
            errors.append(f'Manifest count mismatch: {key}')
    for row in data['conditions']:
        for field in ['description', 'medications', 'diet', 'precautions', 'workout', 'symptoms']:
            if not row[field]:
                errors.append(f"Missing {field}: {row['name']}")
        if set(row['symptoms']) - set(symptoms):
            errors.append(f"Unknown condition symptoms: {row['name']}")
    references = data['reference_conditions']
    names = [r['name'] for r in references]
    if len(names) != len(set(names)):
        errors.append('Duplicate reference condition')
    for row in references:
        if row['eligible_for_training'] is not False or row['clinically_reviewed'] is not False:
            errors.append(f"Unsupported approval status: {row['name']}")
        if (not row['sources'] or not row['description'] or not row.get('care_notes')
                or not row.get('seek_help_notes') or
                (row['data_type'] != 'medicine_safety_reference' and not row['symptom_terms'])):
            errors.append(f"Incomplete reference: {row['name']}")
        for source in row['sources']:
            if (not source['url'].startswith(('https://www.nhs.uk/', 'https://www.fda.gov/'))
                    or not source.get('accessed_on')
                    or not (source.get('page_last_reviewed') or source.get('updated_on'))):
                errors.append(f"Missing source provenance: {row['name']}")
    return errors


def main():
    data = load_data()
    errors = validate(data)
    source_dir = ROOT / 'references/supervisor/dataset'
    for name, expected in data['manifest']['source_sha256'].items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != expected:
            errors.append(f'Source checksum mismatch: {name}')
    with (source_dir / 'Symptom-severity.csv').open(newline='', encoding='utf-8-sig') as source:
        weights = {row['Symptom'].strip(): int(row['weight']) for row in csv.DictReader(source)}
    for row in data['symptom_metadata']:
        if weights.get(row['source_symptom']) != row['source_severity_weight']:
            errors.append(f"Severity correction differs from source: {row['id']}")
    counts = Counter(r['condition'] for r in data['training'])
    report = {
        'schema_version': 1, 'errors': errors,
        'classifier_conditions': len(data['conditions']),
        'reference_only_conditions': len(data['reference_conditions']),
        'unique_patterns_per_condition': dict(sorted(counts.items())),
        'missing_severity_metadata': [r['id'] for r in data['symptoms'] if r['source_severity_weight'] is None],
        'warnings': [
            'Only 5–10 unique patterns per class; repeated source rows are not additional evidence.',
            'All legacy medicine mappings need clinical review before recommendation use.',
            'Reference symptom lists are not patient observations or validated training examples.',
            'NHS guidance is UK-specific; local prescribing and eligibility require separate review.'
        ]
    }
    (ROOT / 'data/quality_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
