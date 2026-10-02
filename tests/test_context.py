import json
import unittest
from unittest.mock import Mock
from src.data import load_data
from src.dialogue import new_state, respond
from src.context_model import DATASET, ARTIFACTS, classify_intent


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def setUp(self):
        self.state = new_state()
        self.predict = Mock(return_value=[{'condition':'Common Cold','probability':.9},
                                         {'condition':'Allergy','probability':.06},
                                         {'condition':'Pneumonia','probability':.02}])

    def say(self, text, profile=None):
        return respond(text, self.state, self.data, self.predict, profile)

    def test_duration_followup_preserves_pending_and_symptoms(self):
        self.say('cough')
        pending = self.state['pending']
        self.predict.reset_mock()
        answer = self.say('It started two days ago')
        self.assertEqual(answer['intent'], 'duration')
        self.assertIn('two days', self.state['details']['duration'])
        self.assertEqual(self.state['pending'], pending)
        self.assertIn('cough', self.state['symptoms'])
        self.predict.assert_not_called()
        self.say('yes, sometimes')
        self.assertIn(pending, self.state['symptoms'])

    def test_summary_and_explanation_do_not_repredict(self):
        self.say('cough, fever and runny nose for two days')
        self.predict.reset_mock()
        summary = self.say('Summarize my symptoms')
        explanation = self.say('Explain the previous result')
        self.assertEqual(summary['intent'], 'summary')
        self.assertIn('cough', summary['text'])
        self.assertIn('Common Cold', explanation['text'])
        self.predict.assert_not_called()

    def test_medicine_and_dose_requests_respect_context(self):
        self.say('cough, fever and runny nose for two days')
        answer = self.say('What medicine can I take?', {'allergies': 'penicillin'})
        self.assertTrue(answer['medicine_withheld'])
        self.assertNotIn('condition', answer)
        self.assertTrue(self.say('How many tablets should I take?')['medicine_withheld'])

    def test_emergency_cannot_be_overridden_by_summary(self):
        self.say('chest pain')
        answer = self.say('Summarize my symptoms')
        self.assertTrue(answer['urgent'])
        self.predict.assert_not_called()

    def test_no_context_does_not_invent_a_match(self):
        self.assertNotIn('condition', self.say('Why did you suggest that condition?'))
        self.predict.assert_not_called()

    def test_bare_medicine_uses_symptoms_after_uncertain_result(self):
        self.state['symptoms'] = ['cough', 'fatigue']
        self.state['last_predictions'] = [
            {'condition': 'Bronchial Asthma', 'probability': .4},
            {'condition': 'Influenza (flu)', 'probability': .3},
        ]
        answer = self.say('medicine')
        self.assertEqual(answer['intent'], 'medicine_question')
        self.assertIn('cough and fatigue', answer['text'])
        self.assertNotIn('Describe your symptoms first', answer['text'])
        self.assertIn('How long', answer['text'])
        self.assertTrue(answer['medicine_withheld'])
        self.assertNotIn('condition', answer)
        self.predict.assert_not_called()
        self.assertEqual(self.state['symptoms'], ['cough', 'fatigue'])

    def test_medicine_followup_does_not_repeat_known_duration(self):
        self.state['symptoms'] = ['cough', 'fatigue']
        self.state['details']['duration'] = 'for three days'
        answer = self.say('medicines please')
        self.assertIn('cough and fatigue', answer['text'])
        self.assertNotIn('How long', answer['text'])
        self.predict.assert_not_called()

    def test_new_symptoms_do_not_use_stale_match_for_medicine_question(self):
        self.say('cough and fever')
        answer = self.say('What medicine can I take for diarrhea?')
        self.assertNotIn('condition', answer)
        self.assertIsNone(self.state['last_condition'])

    def test_authored_splits_are_disjoint(self):
        rows = json.loads(DATASET.read_text())['examples']
        metrics = json.loads((ARTIFACTS / 'metrics.json').read_text())
        self.assertEqual(len({r['text'].lower() for r in rows}), len(rows))
        splits = [set(v) for v in metrics['splits'].values()]
        self.assertEqual(sum(map(len, splits)), len(set.union(*splits)))
        self.assertFalse(metrics['encoder_fine_tuned'])
        self.assertEqual(classify_intent('Can you summarize what I told you?'), 'summary')

    def test_report_help_routes_to_upload_instructions_without_prediction(self):
        answer = self.say('How do I upload my medical report')
        self.assertEqual(answer['intent'], 'report_help')
        self.assertIn('plus icon', answer['text'])
        self.predict.assert_not_called()

    def test_added_condition_has_no_medicine_output(self):
        self.state['last_condition'] = 'Sinusitis'
        answer = self.say('What medicine can I take?')
        self.assertTrue(answer['medicine_withheld'])
        self.assertIn('checked medicine information', answer['text'])
        self.predict.assert_not_called()
