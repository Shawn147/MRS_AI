import json
import unittest

from scripts.collect_pakistan_hospital import parse_page
from src.data import load_data, symptom_text
from src.dialogue import extract_symptoms
from src.health_library import load_health_library


class SourceExpansionTests(unittest.TestCase):
    def test_parser_ignores_treatment_lists_and_rejects_redirect(self):
        item = lambda value: {'type': 'list_item', 'content': [{'type': 'text', 'text': value}]}
        story = {'name': 'Example', 'full_slug': 'pakistan/encyclopedia/example',
                 'updated_at': '2026-10-06', 'content': {'blocks': [
                     {'component': 'accordion', 'item': [
                         {'title': 'Symptoms', 'content': {'content': [item('headache')]}},
                         {'title': 'Treatment', 'content': {'content': [item('medicine')]}},
                     ]}]}}
        html = '<script id="__NEXT_DATA__">' + json.dumps({'props': {'pageProps': {'story': story}}}) + '</script>'
        row = parse_page(html, 'https://hospitals.aku.edu/pakistan/encyclopedia/example')
        self.assertEqual(row['symptom_terms'], ['headache'])
        with self.assertRaises(ValueError):
            parse_page(html, 'https://hospitals.aku.edu/pakistan/encyclopedia/wrong')

    def test_features_encode_words_and_have_training_examples(self):
        data = load_data()
        feature = next(r for r in data['symptoms'] if r['id'].startswith('source_'))
        self.assertIn(feature['label'], symptom_text([feature['id']]))
        self.assertNotIn(feature['id'], symptom_text([feature['id']]))
        self.assertIn(feature['id'], extract_symptoms(feature['label'].lower().replace('\u200b', ''), [feature])[0])
        used = {s for r in data['training'] for s in r['symptoms']}
        self.assertEqual(used, set(data['by_symptom']))
        self.assertEqual({r['condition'] for r in data['training']}, set(data['by_condition']))

    def test_dictionary_never_supplies_training_or_treatment_rules(self):
        library = load_health_library()
        records = [r for r in library['conditions'] if r.get('data_type') == 'ontology_definition']
        self.assertGreater(len(records), 5000)
        self.assertEqual(len({r['concept_id'] for r in records}), len(records))
        self.assertTrue(all(not r['eligible_for_training'] and not r['symptom_terms'] and
                            not r['care_notes'] and not r['seek_help_notes'] for r in records))


if __name__ == '__main__':
    unittest.main()
