import copy
import unittest

from src.data import load_data
from src.dialogue import extract_symptoms
from src.hospital_symptoms import validate_hospital_symptoms


class HospitalSymptomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_published_sources_and_mappings(self):
        rows = self.data['pakistan_hospital_symptoms']
        validate_hospital_symptoms(rows, self.data['by_symptom'])
        self.assertEqual(len(rows), 71)
        self.assertEqual(len({r['term'] for r in rows}), 58)
        self.assertEqual(len({r['source']['url'] for r in rows}), 9)
        self.assertTrue(any(r['feature_id'] is None for r in rows))
        self.assertEqual(len(self.data['symptoms']), 156)

    def test_new_phrase_is_recognized_and_negation_is_preserved(self):
        found, _ = extract_symptoms('I have upper abdominal pain and yellow eyes', self.data['symptoms'])
        self.assertIn('abdominal_pain', found)
        self.assertIn('yellowing_of_eyes', found)
        absent, negated = extract_symptoms('I have no yellow eyes', self.data['symptoms'])
        self.assertNotIn('yellowing_of_eyes', absent)
        self.assertIn('yellowing_of_eyes', negated)

    def test_unrelated_source_and_patient_claim_rejected(self):
        for field, value in [('patient_record', True), ('feature_id', 'not_trained')]:
            rows = copy.deepcopy(self.data['pakistan_hospital_symptoms'])
            rows[0][field] = value
            with self.assertRaises(ValueError):
                validate_hospital_symptoms(rows, self.data['by_symptom'])
        rows = copy.deepcopy(self.data['pakistan_hospital_symptoms'])
        rows[0]['source']['url'] = 'https://pkli.org.pk.fake.example/test'
        with self.assertRaises(ValueError):
            validate_hospital_symptoms(rows, self.data['by_symptom'])

    def test_duplicate_reference_rejected(self):
        rows = copy.deepcopy(self.data['pakistan_hospital_symptoms'])
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):
            validate_hospital_symptoms(rows, self.data['by_symptom'])
