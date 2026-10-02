"""Reference-only public health library; never changes trained classifier features."""
import hashlib
import json
import re
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / 'data/health_library'


def load_health_library():
    if not (DIRECTORY / 'manifest.json').exists():
        return {'conditions': [], 'symptoms': [], 'medicines': [], 'manifest': {}}
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    result = {'manifest': manifest}
    for name in ('conditions', 'symptoms', 'medicines'):
        path = DIRECTORY / (name + '.json')
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != manifest['sha256'][path.name]:
            raise ValueError('Health library checksum mismatch: ' + path.name)
        result[name] = json.loads(contents)
        if len(result[name]) != manifest['collected'][name]:
            raise ValueError('Health library count mismatch: ' + name)
    return result


def answer_records(library):
    """Format library fields for grounded topic answers, without prescribing rules."""
    for condition in library['conditions']:
        name = condition['name']
        aliases = [name, re.sub(r'\([^)]*\)', '', name).strip()]
        aliases.extend(re.findall(r'\(([^)]*)\)', name))
        yield {**condition, 'aliases': list(dict.fromkeys(a for a in aliases if len(a) >= 4))}
    for medicine in library['medicines']:
        notes = [s for s in (medicine['warnings_excerpt'], medicine['contraindications_excerpt']) if s]
        yield {'name': medicine['name'], 'description': medicine['indications_excerpt'],
               'symptom_terms': [], 'care_notes': [], 'seek_help_notes': notes,
               'sources': [medicine['source']], 'data_type': 'drug_label_excerpt',
               'scope': medicine['scope'] + ' This excerpt omits other label sections; read the full label.',
               'eligible_for_training': False, 'clinically_reviewed': False}
