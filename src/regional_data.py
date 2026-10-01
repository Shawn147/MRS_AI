"""Source-aware audit and preview access for research-only clinical cohorts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data import DATA_DIR

REGIONAL_DIR = DATA_DIR / 'regional'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_regional(directory=REGIONAL_DIR):
    directory = Path(directory)
    catalog = json.loads((directory / 'sources.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    errors, warnings = [], []
    if _sha((directory / 'sources.json').read_bytes()) != manifest['catalog_sha256']:
        errors.append('Catalog checksum mismatch')
    if manifest.get('eligible_for_classifier_training') is not False or manifest.get('classifier_10x_target_met_by_collection') is not False:
        errors.append('Research collection cannot claim classifier training eligibility')
    source_map = {s['id']: s for s in catalog['sources']}
    summary_map = {s['source_id']: s for s in manifest['sources']}
    if (len(source_map) != len(catalog['sources']) or len(summary_map) != len(manifest['sources'])
            or set(source_map) != set(summary_map)):
        errors.append('Source IDs or manifest coverage mismatch')
    all_ids = set()
    totals = Counter()
    for ident, source in source_map.items():
        if ident not in summary_map:
            continue
        summary = summary_map[ident]
        if source.get('eligible_for_classifier_training') is not False:
            errors.append(f'{ident}: research approval status changed')
        if source['license']['name'] != 'CC BY 4.0' or not source.get('attribution') or not source.get('source_url'):
            errors.append(f'{ident}: missing licence or attribution')
        files = [summary['file'], summary['conflicts_file']]
        if files != [ident + '.jsonl', ident + '.conflicts.jsonl']:
            errors.append(f'{ident}: unexpected prepared filenames')
            continue
        for name, expected in [(files[0], summary['sha256']), (files[1], summary['conflicts_sha256'])]:
            if _sha((directory / name).read_bytes()) != expected:
                errors.append(f'{ident}: {name} checksum mismatch')
        rows = _records(directory / files[0])
        conflicts = _records(directory / files[1])
        features_seen, provenance = set(), []
        for row in rows + conflicts:
            if (row.get('source_id') != ident or row.get('eligible_for_classifier_training') is not False
                    or set(row.get('features', {})) != set(source['features'])
                    or set(row.get('target', {})) != set(source['targets'])):
                errors.append(f'{ident}: invalid schema or eligibility')
                continue
            values = list(row['features'].values()) + list(row['target'].values())
            if any(v is not None and not isinstance(v, str) for v in values):
                errors.append(f'{ident}: values must be source strings or null')
            if any(row['target'][k] not in allowed for k, allowed in source['target_values'].items()):
                errors.append(f'{ident}: unknown target value')
            canonical_features = _canonical(row['features'])
            group_id = _sha((ident + ':' + canonical_features).encode())
            if group_id != row['feature_group_id']:
                errors.append(f'{ident}: incorrect feature group hash')
            indices = row['source_rows']
            if not indices or any(type(n) is not int or n < 2 for n in indices):
                errors.append(f'{ident}: invalid source-row references')
            else:
                provenance.extend(indices)
        for row in rows:
            expected = _sha((ident + ':' + _canonical([row['features'], row['target']])).encode())
            if row['id'] != expected or row['id'] in all_ids:
                errors.append(f'{ident}: invalid or duplicate record ID')
            all_ids.add(row['id'])
            feature_key = _canonical(row['features'])
            if feature_key in features_seen:
                errors.append(f'{ident}: duplicate or conflicting feature pattern in prepared records')
            features_seen.add(feature_key)
        if sorted(provenance) != list(range(2, source['expected_rows'] + 2)):
            errors.append(f'{ident}: source-row coverage mismatch')
        actual = {'raw_rows': len(provenance), 'unique_rows': len(rows),
                  'duplicates_removed': sum(len(r['source_rows']) - 1 for r in rows),
                  'conflicting_rows': len(conflicts)}
        for key, value in actual.items():
            if summary[key] != value:
                errors.append(f'{ident}: {key} count mismatch')
            totals[key] += value
        missing = {k: sum(r['features'][k] is None for r in rows) for k in source['features']}
        targets = dict(Counter(_canonical(r['target']) for r in rows))
        if missing != summary['missing_feature_values'] or targets != summary['target_counts']:
            errors.append(f'{ident}: missingness or target distribution mismatch')
        if any(missing.values()):
            warnings.append(f'{ident}: missing features retained as null; no imputation')
        if ident == 'pakistan_yicss':
            outside = sum(not 0 <= float(r['features']['compage']) <= 59 for r in rows)
            if outside:
                warnings.append(f'{ident}: {outside} observations have computed age outside the published 0-59 day range')
    for key, value in totals.items():
        if manifest[key] != value:
            errors.append(f'Total {key} mismatch')
    expected_target = manifest['legacy_classifier_patterns'] * 10
    core = json.loads((DATA_DIR / 'manifest.json').read_text())
    classifier_count = core['unique_rows']
    if manifest['legacy_classifier_patterns'] != core.get('legacy_unique_rows', classifier_count):
        errors.append('Classifier baseline changed; rebuild research collection manifest')
    if manifest['collection_target_unique_rows'] != expected_target:
        errors.append('Collection target mismatch')
    if manifest['target_met_for_research_collection'] != (totals['unique_rows'] >= expected_target):
        errors.append('Research collection target status mismatch')
    warnings.extend([
        'Cohorts have different targets, ages and measurement schemas; they are not interchangeable classifier examples.',
        'Distinct observations are not verified independent patients; row references are provenance, not patient IDs.',
        'Verify coded-value meanings, outcome leakage, units and clinical applicability before task-specific modelling.',
        f'The symptom classifier has {classifier_count} educational patterns; no regional cohort is used in chat inference.'
    ])
    return {'schema_version': 1, 'passed': not errors, 'errors': errors, 'warnings': warnings,
            'counts': dict(totals), 'countries': sorted({s['country'] for s in source_map.values()}),
            'source_count': len(source_map)}


def load_regional_preview(directory=REGIONAL_DIR):
    directory = Path(directory)
    audit = audit_regional(directory)
    if audit['errors']:
        raise ValueError('; '.join(audit['errors']))
    return (json.loads((directory / 'sources.json').read_text()),
            json.loads((directory / 'manifest.json').read_text()), audit)
