import hashlib
import json
import unittest
from pathlib import Path
from src.regional_data import REGIONAL_DIR


class HospitalBenchmarkTests(unittest.TestCase):
    def test_benchmarks_keep_tasks_and_groups_separate(self):
        report = json.loads(Path('artifacts/research/metrics.json').read_text())
        self.assertEqual(len(report['models']), 3)
        for model in report['models']:
            raw = (REGIONAL_DIR / (model['source_id'] + '.jsonl')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), model['input_sha256'])
            rows = {row['id']: row for row in map(json.loads, raw.splitlines())}
            groups = []
            for part in ['train', 'validation', 'test']:
                ids = model['splits'][part]
                self.assertEqual(len(ids), model['split_sizes'][part])
                groups.append({rows[i]['feature_group_id'] for i in ids})
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertEqual(sum(map(len, groups)), len(rows))
            self.assertEqual(model['test']['test_rows'], len(model['splits']['test']))
            if model['source_id'] == 'pakistan_heart_failure':
                self.assertNotIn('time', model['features'])
                self.assertIn('time', model['excluded_features'])
