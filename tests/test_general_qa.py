import json
import unittest
from unittest.mock import Mock, patch

from src.data import load_data
from src.dialogue import new_state, respond
from src.general_qa import (answer_general_question, is_information_question,
                            is_unrecognized_health_report, is_contextual_symptom_report)


class GeneralQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_routing_preserves_symptom_state(self):
        state = new_state()
        state['symptoms'] = ['headache']
        state['pending'] = 'cough'
        predictor = Mock()
        with patch('src.general_qa.answer_general_question', return_value={
            'text': 'General information', 'predictions': [], 'intent': 'general_question',
        }) as answer:
            result = respond('what about hepatitis B', state, self.data, predictor)
        self.assertEqual(result['intent'], 'general_question')
        self.assertEqual(state['symptoms'], ['headache'])
        self.assertEqual(state['pending'], 'cough')
        predictor.assert_not_called()
        answer.assert_called_once()

    def test_personal_symptom_report_is_not_general_question(self):
        self.assertFalse(is_information_question('I have a headache, what could it be?'))
        self.assertTrue(is_information_question('What are symptoms of sinusitis?'))
        self.assertTrue(is_unrecognized_health_report('i feel sleepy after eating onion'))
        self.assertTrue(is_unrecognized_health_report("I'm sleepy after meals"))
        self.assertTrue(is_unrecognized_health_report('I have an odd sensation, what is it?'))

    def test_unrecognized_health_report_uses_qwen_without_erasing_state(self):
        state = new_state()
        state['symptoms'] = ['headache']
        state['pending'] = 'cough'
        predictor = Mock()
        with patch('src.general_qa.answer_general_question', return_value={
            'text': 'When did that start?', 'predictions': [], 'intent': 'general_question',
        }) as answer:
            result = respond('i feel sleepy after eating onion', state, self.data, predictor)
        self.assertEqual(result['text'], 'When did that start?')
        self.assertEqual(state['symptoms'], ['headache'])
        self.assertEqual(state['pending'], 'cough')
        predictor.assert_not_called()
        answer.assert_called_once()

    def test_known_symptom_with_context_uses_qwen_and_remembers_symptom(self):
        state = new_state()
        predictor = Mock()
        with patch('src.general_qa.answer_general_question', return_value={
            'text': 'When does that happen?', 'predictions': [], 'intent': 'general_question',
        }):
            result = respond('I get a headache when I stand up', state, self.data, predictor)
        self.assertTrue(is_contextual_symptom_report('headache when I stand up'))
        self.assertEqual(result['intent'], 'general_question')
        self.assertIn('headache', state['symptoms'])
        predictor.assert_not_called()

    def test_emergency_takes_priority(self):
        with patch('src.general_qa.answer_general_question') as answer:
            result = respond('What about chest pain?', new_state(), self.data, Mock())
        self.assertTrue(result['urgent'])
        answer.assert_not_called()

    def test_reference_answer_has_only_real_source(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read = Mock(return_value=json.dumps({'message': {'content': 'Sinusitis information.'}}).encode())
        with patch('src.general_qa.urlopen', return_value=response) as send:
            result = answer_general_question('What is sinusitis?', new_state(), self.data)
        self.assertEqual(result['intent'], 'general_question')
        self.assertEqual(result['sources'], ['https://www.nhs.uk/conditions/sinusitis-sinus-infection/'])
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload['model'], 'qwen3:4b')
        self.assertIn('Sinusitis', payload['messages'][0]['content'])

    def test_reasoning_is_not_shown_to_user(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read = Mock(return_value=json.dumps({
            'message': {'content': 'Planning details.</think>\n\nSinusitis is inflammation of the sinuses.'}
        }).encode())
        with patch('src.general_qa.urlopen', return_value=response):
            result = answer_general_question('What is sinusitis?', new_state(), self.data)
        self.assertEqual(result['text'], 'Sinusitis is inflammation of the sinuses.')


if __name__ == '__main__':
    unittest.main()
