import json
import os
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

    def test_hosted_openai_compatible_endpoint(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read = Mock(return_value=json.dumps({
            'choices': [{'message': {'content': 'A short answer.'}}]
        }).encode())
        settings = {'MRS_LLM_BACKEND': 'openai',
                    'MRS_LLM_URL': 'https://example.test/v1/chat/completions',
                    'MRS_LLM_MODEL': 'hosted-qwen', 'MRS_LLM_API_KEY': 'test-key'}
        with patch.dict(os.environ, settings), patch('src.general_qa.urlopen', return_value=response) as send:
            result = answer_general_question('How can I support gut health?', new_state(), self.data)
        self.assertEqual(result['text'], 'A short answer.')
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, settings['MRS_LLM_URL'])
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')
        self.assertEqual(request.get_header('User-agent'), 'MRS-AI/1.0')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'hosted-qwen')
        self.assertEqual(payload['messages'][-1]['content'], 'How can I support gut health?')

    def test_hosted_endpoint_requires_configuration(self):
        with patch.dict(os.environ, {'MRS_LLM_BACKEND': 'openai',
                                     'MRS_LLM_URL': '', 'MRS_LLM_API_KEY': ''}), \
                patch('src.general_qa.urlopen') as send:
            result = answer_general_question('How can I support gut health?', new_state(), self.data)
        self.assertEqual(result['intent'], 'general_question_unavailable')
        self.assertIn('hosted language model', result['text'])
        send.assert_not_called()

    def test_reference_summary_fallback_without_downloaded_medembed(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read = Mock(return_value=json.dumps({'message': {'content': 'Sinusitis is sinus inflammation.'}}).encode())
        with patch('src.general_qa.search_references', side_effect=FileNotFoundError), \
                patch('src.general_qa.urlopen', return_value=response) as send:
            result = answer_general_question('What is sinusitis?', new_state(), self.data)
        self.assertEqual(result['sources'], ['https://www.nhs.uk/conditions/sinusitis-sinus-infection/'])
        self.assertIn('Sinusitis', json.loads(send.call_args.args[0].data)['messages'][0]['content'])


if __name__ == '__main__':
    unittest.main()
