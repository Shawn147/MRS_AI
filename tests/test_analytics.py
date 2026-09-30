import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from src.analytics import record_event, summarize


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'events.sqlite3'
        self.data = {'by_condition': {'Example': {'medications': ['A', 'B', 'A']}}}

    def event(self, key, **extra):
        message = {'id': key, 'timestamp': '2026-09-25T10:00:00+00:00', 'condition': 'Example',
                   'symptoms': ['cough', 'cough'], 'content': 'Private patient message', **extra}
        record_event('session', 'conversation', message, self.data, self.path)

    def test_empty(self):
        report = summarize(path=self.path)
        self.assertEqual(report['responses'], 0)
        self.assertEqual(report['medicines'], [])

    def test_reruns_and_duplicate_lists_do_not_inflate_counts(self):
        self.event('one')
        self.event('one')
        self.event('two')
        report = summarize(path=self.path)
        self.assertEqual(report['responses'], 2)
        self.assertEqual(report['sessions'], 1)
        self.assertEqual(report['medicines'][0]['Responses'], 2)
        self.assertEqual(report['medicines'][0]['Conversations'], 1)
        self.assertEqual(report['symptoms'][0]['Responses'], 2)

    def test_withheld_uncertain_and_urgent_exclude_medicines(self):
        self.event('one', medicine_withheld=True)
        self.event('two', uncertain=True)
        self.event('three', urgent=True)
        self.event('four', urgent=True)
        report = summarize(path=self.path)
        self.assertEqual(report['medicines'], [])
        self.assertEqual(report['urgent_conversations'], 1)
        self.assertEqual(report['withheld_responses'], 1)

    def test_date_filter_and_privacy(self):
        self.event('one')
        self.event('old', timestamp='2026-01-01T00:00:00+00:00')
        self.assertEqual(summarize('2026-09-01', self.path)['responses'], 1)
        connection = sqlite3.connect(self.path)
        try:
            dumped = '\n'.join(connection.iterdump())
            self.assertNotIn('Private patient message', dumped)
            self.assertNotIn('content', dumped)
        finally:
            connection.close()

    def test_model_failures_are_counted_without_question_text(self):
        self.event('failed', condition=None, intent='general_question_unavailable')
        self.assertEqual(summarize(path=self.path)['model_failures'], 1)
        self.assertNotIn('Private patient message', self.path.read_bytes().decode('latin1'))
