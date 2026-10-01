"""Add explicitly illustrative examples, without manufacturing hospital cases.

Original holdouts are fixed. Augmentations of legacy rows belong to their
training parent only; new source profiles are training-only and unevaluated.
"""
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def apply_expansion(files, manifest):
    config_path = DATA / 'expansion_profiles.json'
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    split = json.loads((DATA / 'legacy_splits.json').read_text())
    references = {r['name']: r for r in json.loads((DATA / 'reference_conditions.json').read_text())}
    legacy = files['training.json']
    old_ids = {r['id'] for r in legacy}
    if set().union(*(set(v) for v in split.values())) != old_ids:
        raise ValueError('Fixed legacy split does not cover original examples')
    split_lookup = {ident: part for part, ids in split.items() for ident in ids}
    for row in legacy:
        row.update(data_type='legacy_educational_pattern', split=split_lookup[row['id']],
                   family_id=row['id'], clinically_reviewed=False)
    known_symptoms = {r['id'] for r in files['symptoms.json']}
    for symptom in config['symptoms']:
        if symptom['id'] in known_symptoms:
            raise ValueError('Expanded symptom duplicates legacy ID')
        if not symptom['sources']:
            raise ValueError('Expanded symptom requires source')
        files['symptoms.json'].append({**symptom, 'source_severity_weight': None})
        known_symptoms.add(symptom['id'])
    conditions = files['conditions.json']
    for profile in config['conditions']:
        reference = references[profile['reference_name']]
        if profile['name'] in {r['name'] for r in conditions}:
            raise ValueError('Duplicate expanded condition')
        conditions.append({'name': profile['name'], 'description': reference['description'],
                           'medications': [], 'diet': [], 'workout': [],
                           'precautions': reference['seek_help_notes'], 'symptoms': profile['symptoms'],
                           'medicine_order': 'No medication mapping supplied',
                           'clinically_reviewed': False, 'data_type': 'illustrative_condition_profile',
                           'sources': reference['sources'], 'reference_name': reference['name'],
                           'review_status': 'Illustrative training coverage; not clinically validated'})
    # Candidate ambiguity is checked before any label is selected.
    candidates = defaultdict(list)
    parent_rows = [r for r in legacy if r['split'] == 'train']
    parents = [(r['id'], r['condition'], sorted(r['symptoms']), 'legacy_training_parent', []) for r in parent_rows]
    for profile in config['conditions']:
        parents.append(('profile:' + profile['name'], profile['name'], sorted(profile['symptoms']),
                        'source_summarized_profile', references[profile['reference_name']]['sources']))
    occupied = {tuple(sorted(r['symptoms'])) for r in legacy}
    for parent, condition, symptoms, kind, sources in parents:
        if set(symptoms) - known_symptoms:
            raise ValueError('Unknown profile symptom')
        minimum = max(3, (len(symptoms) + 1) // 2)
        for length in range(minimum, len(symptoms) + 1):
            for pattern in itertools.combinations(symptoms, length):
                if pattern not in occupied:
                    candidates[pattern].append((condition, parent, kind, sources))
    queues = defaultdict(list)
    ambiguous = 0
    for pattern, options in candidates.items():
        labels = {o[0] for o in options}
        if len(labels) != 1:
            ambiguous += 1
            continue
        # Keep one reproducible parent and all clearly labelled synthetic provenance.
        condition, parent, kind, sources = sorted(options, key=lambda o: o[1])[0]
        queues[condition].append((pattern, parent, kind, sources))
    rng = random.Random(42)
    for name in sorted(queues):
        rng.shuffle(queues[name])
    target = config['target_unique_patterns']
    added = []
    while len(legacy) + len(added) < target:
        progress = False
        for condition in sorted(queues):
            if not queues[condition] or len(legacy) + len(added) >= target:
                continue
            pattern, parent, kind, sources = queues[condition].pop()
            ident = hashlib.sha256('|'.join(pattern).encode()).hexdigest()[:16]
            added.append({'id': ident, 'condition': condition, 'symptoms': list(pattern),
                          'source_rows': [], 'data_type': 'illustrative_symptom_pattern',
                          'split': 'train', 'family_id': parent, 'parent_type': kind,
                          'sources': sources, 'clinically_reviewed': False,
                          'provenance': 'Generated subset for educational model development; not an observed patient or validated symptom combination.'})
            progress = True
        if not progress:
            raise ValueError('Insufficient unambiguous illustrative patterns for target')
    files['training.json'] = legacy + added
    conditions.sort(key=lambda r: r['name'])
    manifest.update(schema_version=2, legacy_raw_rows=manifest['raw_rows'],
                    legacy_unique_rows=len(legacy), illustrative_rows=len(added),
                    raw_rows=manifest['raw_rows'] + len(added), unique_rows=len(legacy) + len(added),
                    condition_count=len(conditions), feature_count=len(files['symptoms.json']),
                    provenance_counts=dict(Counter(r['data_type'] for r in files['training.json'])),
                    split_policy='Fixed legacy holdouts; generated training-parent subsets stay in training. New condition profiles have no independent evaluation.',
                    ambiguity_excluded=ambiguous,
                    expansion_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    legacy_splits_sha256=hashlib.sha256((DATA / 'legacy_splits.json').read_bytes()).hexdigest())
    manifest['notes'].extend([
        'Illustrative rows are generated educational examples, not hospital patients or new clinical evidence.',
        'Original CSV row count remains 4920. Total input rows include explicitly illustrative additions.',
        'New condition labels lack independent clinical evaluation; no medication mappings have been generated.'
    ])
