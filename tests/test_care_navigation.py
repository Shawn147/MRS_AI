import unittest
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse
import re

from src.care_navigation import navigation_reply
from src.data import load_data
from src.dialogue import new_state, respond


class CareNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_location_reply_completes_hospital_search_without_symptom_prediction(self):
        state = new_state()
        state['symptoms'] = ['fatigue', 'headache', 'cough']
        predictor, generator = Mock(), Mock()
        first = respond('which hospital is near to me?', state, self.data, predictor, general_answer=generator)
        self.assertEqual(first['text'], 'Which area and city are you in?')
        second = respond('i live in surjani town karachi', state, self.data, predictor, general_answer=generator)
        self.assertEqual(second['intent'], 'hospital_search')
        self.assertEqual(state['location'], 'surjani town karachi')
        url = re.search(r'\]\((https://[^)]+)\)', second['text'])[1]
        self.assertEqual(parse_qs(urlparse(url).query)['query'], ['hospitals near surjani town karachi'])
        self.assertEqual(state['symptoms'], ['fatigue', 'headache', 'cough'])
        predictor.assert_not_called()
        generator.assert_not_called()

    def test_saved_hospital_question_accepts_bare_area(self):
        state = new_state()
        state['recent_turns'] = [
            {'role': 'user', 'content': 'which hospital is near to me?'},
            {'role': 'assistant', 'content': 'Use your maps app.'}]
        reply = navigation_reply('Surjani Town, Karachi', state)
        self.assertEqual(reply['intent'], 'hospital_search')
        self.assertIn('Surjani Town, Karachi', reply['text'])

    def test_inline_location_and_later_hospital_question_use_shared_area(self):
        state = new_state()
        first = navigation_reply('find a hospital near Surjani Town Karachi', state)
        self.assertEqual(first['intent'], 'hospital_search')
        self.assertEqual(navigation_reply('which hospital is nearest?', state)['intent'], 'hospital_search')

    def test_symptoms_are_not_taken_as_location(self):
        state = new_state()
        navigation_reply('which hospital is near me?', state)
        self.assertIsNone(navigation_reply('I have cough and fever', state))
        self.assertNotIn('location', state)
        navigation_reply('which hospital is near me?', state)
        self.assertIsNone(navigation_reply('I am in pain', state))
        self.assertFalse(state['pending_location'])
        self.assertNotIn('location', state)

    def test_location_outside_hospital_conversation_is_not_intercepted(self):
        self.assertIsNone(navigation_reply('I live in Karachi', new_state()))

    def test_urgent_state_can_still_request_help_finding_hospital(self):
        state = new_state()
        state['urgent'] = True
        result = respond('which hospital is near me?', state, self.data, Mock())
        self.assertEqual(result['intent'], 'hospital_location')
        self.assertTrue(state['urgent'])


if __name__ == '__main__':
    unittest.main()
