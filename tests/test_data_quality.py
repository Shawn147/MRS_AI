import copy
import unittest
from unittest.mock import patch

from scripts.audit_data import validate
from scripts.prepare_json import validate_condition_tables
from src.data import load_data


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_current_data_passes(self):
        self.assertEqual(validate(self.data), [])

    def test_missing_metadata_is_resolved_without_changing_ids(self):
        expected = {'spotting_ urination': 6, 'foul_smell_of urine': 5, 'dischromic _patches': 6}
        for key, weight in expected.items():
            self.assertEqual(self.data['by_symptom'][key]['source_severity_weight'], weight)

    def test_unknown_training_symptom_is_rejected(self):
        self.data['training'][0]['symptoms'].append('unknown_symptom')
        self.assertTrue(any('Unknown reference' in e for e in validate(self.data)))

    def test_conflicting_pattern_is_rejected(self):
        duplicate = copy.deepcopy(self.data['training'][0])
        duplicate.update(id='different', condition='Allergy')
        self.data['training'].append(duplicate)
        self.assertTrue(any('conflicting pattern' in e for e in validate(self.data)))

    def test_duplicate_source_row_is_rejected(self):
        self.data['training'][0]['source_rows'].append(2)
        self.assertTrue(any('provenance' in e for e in validate(self.data)))

    def test_reference_records_cannot_silently_become_training_data(self):
        self.data['reference_conditions'][0]['eligible_for_training'] = True
        self.assertTrue(any('approval status' in e for e in validate(self.data)))

    def test_duplicate_csv_conditions_cannot_silently_overwrite(self):
        with patch('scripts.prepare_json.read', return_value=[{'Disease': 'A'}, {'Disease': ' A '}]):
            with self.assertRaisesRegex(ValueError, 'Duplicate condition'):
                validate_condition_tables({'A'})

    def test_missing_csv_condition_is_rejected(self):
        with patch('scripts.prepare_json.read', return_value=[{'Disease': 'A'}]):
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                validate_condition_tables({'A', 'B'})
