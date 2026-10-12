"""Build an explicitly illustrative development set from cited symptom inventories.

Retains original IDs and holdouts. Source phrases are model features, not a
claim that each phrase describes a different clinical symptom.
"""
import hashlib
import json
from pathlib import Path
import re

DATA = Path(__file__).resolve().parents[1] / 'data'


def key(value):
    return re.sub(r'\W+', ' ', value.casefold().replace('\u200b', '')).strip()


def main():
    config = json.loads((DATA / 'expansion_profiles.json').read_text())
    generated_names = {r['name'] for r in config['conditions'] if r.get('profile_type') == 'illustrative_source_inventory'}
    references = json.loads((DATA / 'reference_conditions.json').read_text())
    config['conditions'] = [r for r in config['conditions'] if r.get('profile_type') != 'illustrative_source_inventory']
    config['symptoms'] = [r for r in config['symptoms'] if not r['id'].startswith('source_')]
    references = [r for r in references if r['data_type'] != 'public_source_excerpt']
    current = json.loads((DATA / 'conditions.json').read_text())
    features = json.loads((DATA / 'symptoms.json').read_text())
    current = [r for r in current if r['name'] not in generated_names]
    features = [r for r in features if not r['id'].startswith('source_')]
    lookup = {key(a): r['id'] for r in features for a in [r['label'], *r['aliases']]}
    reserved_names = {key(r['name']) for r in current + references}
    reserved_urls = {s['url'].rstrip('/') for r in references for s in r['sources']}
    sources = (json.loads((DATA / 'pakistan_hospital_topics.json').read_text()) +
               json.loads((DATA / 'health_library/conditions.json').read_text()))
    candidates = []
    inventories = set()
    for row in sources:
        if key(row['name']) in reserved_names or any(s['url'].rstrip('/') in reserved_urls for s in row['sources']):
            continue
        terms = list(dict.fromkeys(t for t in row['symptom_terms'] if 1 <= len(t.split()) <= 12))
        inventory = tuple(sorted(key(t) for t in terms))
        if len(terms) < 4 or not row['description'] or inventory in inventories:
            continue
        # Procedures and generic informational pages are not condition labels.
        if re.search(r'\b(test|tests|screening|surgery|repair|replacement|transplant|vaccination|abdominoplasty|blepharoplasty|facelift|replant)\b', row['name'], re.I):
            continue
        reserved_names.add(key(row['name']))
        inventories.add(inventory)
        candidates.append((row, terms))
    needed = 510 - len(current)
    if len(candidates) < needed:
        raise ValueError('Insufficient distinct source profiles for 510 labels')
    selected = candidates[:needed]
    new_profiles = []
    def feature(term, row):
        normalized = key(term)
        if normalized in lookup:
            return lookup[normalized]
        if len(features) >= 1560:
            return None
        ident = 'source_' + hashlib.sha256(normalized.encode()).hexdigest()[:16]
        record = {'id': ident, 'label': term, 'aliases': [term],
                  'sources': row['sources'], 'data_type': 'source_summarized_symptom',
                  'clinically_reviewed': False}
        features.append(record)
        config['symptoms'].append(record)
        lookup[normalized] = ident
        return ident
    # First allocate a source-backed minimum to every label, then broaden each.
    for row, terms in selected:
        ids = list(dict.fromkeys(feature(t, row) for t in terms[:4]))
        if None in ids or len(ids) < 3:
            raise ValueError('Insufficient feature budget for each condition profile')
        profile = {'name': row['name'], 'reference_name': row['name'], 'symptoms': ids,
                   'profile_type': 'illustrative_source_inventory'}
        new_profiles.append(profile)
        references.append({**row, 'review_status': 'Published source excerpt; clinical and local applicability review pending'})
    for row, terms in selected:
        profile = next(p for p in new_profiles if p['name'] == row['name'])
        for term in terms[4:]:
            ident = feature(term, row)
            if ident and ident not in profile['symptoms']:
                profile['symptoms'].append(ident)
    if len(features) != 1560:
        raise ValueError('Insufficient distinct used source phrases for 1560 features')
    config['conditions'].extend(new_profiles)
    config['target_unique_patterns'] = 6000
    config['provenance'] = ('Source-linked symptom phrase inventories. All added combinations are illustrative, '
                            'not observed patients. Phrase features are not counts of medically distinct symptoms. '
                            'New labels have no independent held-out clinical evaluation.')
    for filename, value in [('expansion_profiles.json', config), ('reference_conditions.json', references)]:
        (DATA / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print('new profiles', len(new_profiles), 'source summaries', len(references), 'phrase features', len(features))


if __name__ == '__main__':
    main()
