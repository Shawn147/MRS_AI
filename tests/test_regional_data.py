import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.collect_regional_data import clean, prepare_records
from src.data import ARTIFACT_DIR, fingerprint, load_data
from src.regional_data import REGIONAL_DIR, audit_regional


class RegionalDataTests(unittest.TestCase):
    def test_collected_cohorts_pass_audit_without_changing_classifier(self):
        report = audit_regional()
        self.assertTrue(report['passed'], report['errors'])
        self.assertEqual(report['counts']['unique_rows'], 3900)
        self.assertEqual(report['counts']['raw_rows'], 4169)
        self.assertEqual(report['counts']['duplicates_removed'], 269)
        self.assertEqual(report['countries'], ['Bangladesh', 'India', 'Pakistan'])
        self.assertEqual(load_data()['manifest']['legacy_unique_rows'], 304)
        saved = json.loads((ARTIFACT_DIR / 'metrics.json').read_text())
        self.assertEqual(fingerprint(), saved['data_fingerprint'])

    def test_deduplication_keeps_provenance_and_quarantines_conflicting_targets(self):
        source = {'id': 'sample', 'source_columns': ['id', 'sign', 'outcome'],
                  'features': ['sign'], 'targets': ['outcome'],
                  'target_values': {'outcome': ['0', '1']}}
        rows = [(2, {'id': 'a', 'sign': 'yes', 'outcome': '1'}),
                (3, {'id': 'b', 'sign': 'yes', 'outcome': '1'}),
                (4, {'id': 'c', 'sign': 'no', 'outcome': '0'}),
                (5, {'id': 'd', 'sign': 'no', 'outcome': '1'})]
        records, conflicts, summary = prepare_records(source, rows)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['source_rows'], [2, 3])
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(summary['duplicates_removed'], 1)
        self.assertEqual(summary['conflicting_rows'], 2)
        self.assertEqual(set(records[0]['features']), {'sign'})
        self.assertFalse(records[0]['eligible_for_classifier_training'])

    def test_missing_values_preserve_zero_and_source_codes(self):
        for value in [None, '', '?', ' NaN ', 'nan']:
            self.assertIsNone(clean(value))
        self.assertEqual(clean(0), '0')
        self.assertEqual(clean('-9'), '-9')
        self.assertEqual(clean('notckd'), 'notckd')

    def test_prepared_data_contains_only_clinical_allowlist(self):
        catalog = json.loads((REGIONAL_DIR / 'sources.json').read_text())
        for source in catalog['sources']:
            raw = (REGIONAL_DIR / (source['id'] + '.jsonl')).read_text()
            for line in raw.splitlines():
                row = json.loads(line)
                self.assertEqual(set(row['features']), set(source['features']))
                self.assertNotIn('id1', row['features'])
                self.assertNotIn('ttrig', row['features'])
                self.assertNotIn('site', row['features'])
                self.assertNotIn('presentingcomplaints', row['features'])

    def test_tampered_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'regional'
            shutil.copytree(REGIONAL_DIR, directory)
            path = directory / 'pakistan_heart_failure.jsonl'
            path.write_text(path.read_text() + '\n')
            self.assertTrue(any('checksum' in e for e in audit_regional(directory)['errors']))

    def test_research_rows_cannot_silently_become_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'regional'
            shutil.copytree(REGIONAL_DIR, directory)
            path = directory / 'pakistan_heart_failure.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['eligible_for_classifier_training'] = True
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertTrue(any('eligibility' in e for e in audit_regional(directory)['errors']))


if __name__ == '__main__':
    unittest.main()
