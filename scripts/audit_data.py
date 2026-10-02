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
    manifest = data['manifest']
    legacy = [r for r in data['training'] if r.get('data_type') != 'illustrative_symptom_pattern']
    illustrative = [r for r in data['training'] if r.get('data_type') == 'illustrative_symptom_pattern']
    legacy_raw = manifest.get('legacy_raw_rows', manifest['raw_rows'])
    if sorted(source_rows) != list(range(2, legacy_raw + 2)):
        errors.append('Source-row provenance is incomplete or duplicated')
    for key, actual in [('unique_rows', len(seen)), ('condition_count', len(conditions)),
                        ('feature_count', len(symptoms)),
                        ('duplicates_removed', len(source_rows) - len(legacy))]:
        if data['manifest'][key] != actual:
            errors.append(f'Manifest count mismatch: {key}')
    if manifest.get('schema_version') == 2:
        parents = {r['id']: r for r in legacy}
        profiles = {r['name']: r for r in data['conditions'] if r.get('data_type') == 'illustrative_condition_profile'}
        for row in illustrative:
            parent = parents.get(row.get('family_id'))
            if parent:
                valid = parent['split'] == 'train' and parent['condition'] == row['condition'] and set(row['symptoms']) <= set(parent['symptoms'])
            else:
                profile = profiles.get(row['condition'])
                valid = bool(profile and row.get('family_id') == 'profile:' + row['condition'] and row.get('sources') == profile['sources'] and set(row['symptoms']) <= set(profile['symptoms']))
            if not valid or row.get('split') != 'train' or row['source_rows'] or row.get('clinically_reviewed') is not False:
                errors.append(f"Invalid illustrative provenance: {row['id']}")
        fixed = json.loads((ROOT / 'data/legacy_splits.json').read_text())
        for part, ids in fixed.items():
            if set(ids) != {r['id'] for r in legacy if r.get('split') == part}:
                errors.append('Legacy holdout provenance changed')
        for key, count in [('legacy_unique_rows', len(legacy)), ('illustrative_rows', len(illustrative)), ('raw_rows', legacy_raw + len(illustrative))]:
            if manifest.get(key) != count:
                errors.append(f'Manifest count mismatch: {key}')
        for filename, key in [('expansion_profiles.json', 'expansion_config_sha256'), ('legacy_splits.json', 'legacy_splits_sha256')]:
            if hashlib.sha256((ROOT / 'data' / filename).read_bytes()).hexdigest() != manifest.get(key):
                errors.append(f'Expansion provenance checksum mismatch: {filename}')
    for row in data['conditions']:
        fields = ['description', 'precautions', 'symptoms'] if row.get('data_type') == 'illustrative_condition_profile' else ['description', 'medications', 'diet', 'precautions', 'workout', 'symptoms']
        if row.get('data_type') == 'illustrative_condition_profile' and (not row.get('sources') or row['medications'] or row.get('clinically_reviewed') is not False):
            errors.append(f"Invalid illustrative condition provenance: {row['name']}")
        for field in fields:
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
            if (not source['url'].startswith(('https://www.nhs.uk/', 'https://www.fda.gov/', 'https://hospitals.aku.edu/pakistan/', 'https://www.shifa.com.pk/', 'https://pkli.org.pk/'))
                    or not source.get('accessed_on')
                    or not (source.get('page_last_reviewed') or source.get('updated_on') or source.get('date_not_published') is True)):
                errors.append(f"Missing source provenance: {row['name']}")
    from src.hospital_symptoms import validate_hospital_symptoms
    try:
        validate_hospital_symptoms(data['pakistan_hospital_symptoms'], data['by_symptom'])
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
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
        'source_linked_reference_records': len(data['reference_conditions']),
        'pakistan_hospital_symptoms': {
            'topic_symptom_links': len(data['pakistan_hospital_symptoms']),
            'distinct_terms': len({r['term'] for r in data['pakistan_hospital_symptoms']}),
            'source_pages': len({r['source']['url'] for r in data['pakistan_hospital_symptoms']}),
            'mapped_links': sum(r['feature_id'] is not None for r in data['pakistan_hospital_symptoms']),
            'sha256': hashlib.sha256((ROOT / 'data/pakistan_hospital_symptoms.json').read_bytes()).hexdigest(),
        },
        'unique_patterns_per_condition': dict(sorted(counts.items())),
        'missing_severity_metadata': [r['id'] for r in data['symptoms'] if r['source_severity_weight'] is None],
        'warnings': [
            'Illustrative subsets increase pattern coverage, not independent clinical evidence. Some labels still have few patterns.',
            'New condition profiles have no independent held-out evaluation; original holdouts are preserved.',
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
