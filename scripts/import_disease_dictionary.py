"""Import licensed, defined disease concepts, separately labelled from guidance.

Requires a complete local official Mondo release (.obo). No symptom or medicine
rules are inferred from a disease name or ontology relationship.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/health_library'


def key(value):
    return re.sub(r'\W+', ' ', value.casefold()).strip()


def import_records(path, target=6390):
    raw = path.read_bytes()
    content = raw.decode()
    version = re.search(r'^data-version: (.+)$', content, re.M)[1]
    release = version.rsplit('/', 1)[-1]
    blocks = content.split('\n[Term]\n')[1:]
    parents = set(re.findall(r'^is_a: (MONDO:\d+)', content, re.M))
    rows = json.loads((OUT / 'conditions.json').read_text())
    rows = [r for r in rows if r.get('data_type') != 'ontology_definition']
    seen = {key(r['name']) for r in rows}
    for row in json.loads((ROOT / 'data/pakistan_hospital_topics.json').read_text()):
        if row['description'] and key(row['name']) not in seen:
            rows.append(row)
            seen.add(key(row['name']))
    added = 0
    for block in blocks:
        fields = dict(re.findall(r'^(id|name|is_obsolete): (.+)$', block, re.M))
        ident = fields.get('id', '')
        definitions = re.findall(r'^def: "((?:[^"\\]|\\.)*)"', block, re.M)
        if (not ident.startswith('MONDO:0') or ident in parents or
                fields.get('is_obsolete') == 'true' or not definitions or len(rows) >= target):
            continue
        definition = definitions[0].replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
        if not 5 <= len(definition.split()) <= 100:
            continue
        aliases = re.findall(r'^synonym: "((?:[^"\\]|\\.)*)" EXACT', block, re.M)
        names = {key(fields['name']), *(key(a) for a in aliases)}
        if names & seen:
            continue
        seen.update(names)
        iri = 'http://purl.obolibrary.org/obo/' + ident.replace(':', '_')
        rows.append({'name': fields['name'], 'concept_id': ident, 'description': definition,
                     'aliases': aliases, 'symptom_terms': [], 'care_notes': [], 'seek_help_notes': [],
                     'sources': [{'publisher': 'Mondo Disease Ontology',
                                  'url': 'https://www.ebi.ac.uk/ols4/ontologies/mondo/classes?iri=' + quote(iri, safe=''),
                                  'accessed_on': '2026-10-07', 'updated_on': release,
                                  'attribution': 'Mondo Disease Ontology, ' + version + ', CC BY 4.0.',
                                  'license_url': 'https://creativecommons.org/licenses/by/4.0/'}],
                     'data_type': 'ontology_definition', 'eligible_for_training': False,
                     'clinically_reviewed': False,
                     'scope': 'Disease dictionary definition only; no symptom likelihoods, treatment guidance or clinical training evidence.'})
        added += 1
    if len(rows) < target:
        raise ValueError('Insufficient non-obsolete defined leaf concepts for target')
    (OUT / 'conditions.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    provenance = {'release': version, 'sha256': hashlib.sha256(raw).hexdigest(),
                  'download_url': 'https://github.com/monarch-initiative/mondo/releases/download/v' + release + '/mondo.obo',
                  'license': 'CC BY 4.0', 'definitions_added': added,
                  'selection': 'Active MONDO:0 leaf concepts with complete 5–100 word definitions; exact names and exact synonyms deduplicated against existing references.'}
    (OUT / 'disease_dictionary_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('reference topics', len(rows), 'dictionary definitions', added)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    import_records(parser.parse_args().path)
