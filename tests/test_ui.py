import unittest
import os
import tempfile
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    return at


class UITests(unittest.TestCase):
    def setUp(self):
        self.analytics_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.analytics_dir.cleanup)
        patcher = patch.dict(os.environ, {'MRS_ANALYTICS_DB': self.analytics_dir.name + '/events.sqlite3'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self):
        at = AppTest.from_file('app.py', default_timeout=60).run()
        self.assertFalse(at.exception)
        return at

    def test_chat_navigation_and_isolated_history(self):
        at = self.app()
        at.chat_input[0].set_value('I have a runny nose, sneezing and cough').run()
        self.assertFalse(at.exception)
        first = at.session_state['active_chat']
        messages = at.session_state['chats'][first]['messages']
        self.assertEqual(len(messages), 2)
        self.assertIn('one possible condition', messages[-1]['content'])
        self.assertEqual(messages[-1]['model'], 'transformer')
        self.assertTrue(any('Medicine information' in e.label for e in list(at.expander) + list(at.status)))
        self.assertFalse(any('Model score:' in m.value for m in at.markdown))
        click(at, 'About & sources')
        click(at, 'Dataset Preview')
        for table in ['reference_conditions', 'symptom_metadata', 'symptoms']:
            next(s for s in at.selectbox if s.label == 'JSON table').select(table).run()
            self.assertFalse(at.exception, table)
        click(at, 'About & sources')
        click(at, 'Model Evaluation')
        self.assertFalse(at.exception)
        click(at, 'Conversation history')
        self.assertTrue(at.get('download_button'))
        click(at, 'New conversation')
        second = at.session_state['active_chat']
        self.assertNotEqual(first, second)
        self.assertFalse(at.session_state['chats'][second]['state']['symptoms'])
        at.chat_input[0].set_value('chest pain').run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertTrue(at.session_state['chats'][second]['state']['urgent'])
        self.assertFalse(at.session_state['chats'][first]['state']['urgent'])
        at.button(key='open_' + first).click().run()
        self.assertEqual(at.session_state['active_chat'], first)
        self.assertEqual(len(at.session_state['chats'][first]['messages']), 2)

    def test_starter_and_followup_are_functional(self):
        at = self.app()
        click(at, 'I have a cough and a sore throat')
        key = at.session_state['active_chat']
        self.assertEqual(len(at.session_state['chats'][key]['messages']), 2)
        click(at, 'New conversation')
        at.chat_input[0].set_value('cough').run()
        key = at.session_state['active_chat']
        pending = at.session_state['chats'][key]['state']['pending']
        self.assertIsNotNone(pending)
        click(at, 'Yes, I do')
        self.assertIn(pending, at.session_state['chats'][key]['state']['symptoms'])
        self.assertEqual(len(at.session_state['chats'][key]['messages']), 4)
        self.assertFalse(at.exception)

    def test_context_hides_previous_medicines_and_is_isolated(self):
        at = self.app()
        at.chat_input[0].set_value('runny nose, sneezing and cough').run()
        key = at.session_state['active_chat']
        click(at, 'Health context')
        next(t for t in at.text_input if t.label == 'Known medicine allergies').set_value('penicillin')
        click(at, 'Save health context')
        self.assertEqual(at.session_state['chats'][key]['profile']['allergies'], 'penicillin')
        click(at, 'Back to conversation')
        self.assertFalse(any('Medicine information' in e.label for e in list(at.expander) + list(at.status)))
        self.assertTrue(any('withheld' in i.value for i in at.info))
        click(at, 'New conversation')
        new_key = at.session_state['active_chat']
        self.assertFalse(at.session_state['chats'][new_key]['profile'])
        click(at, 'Health context')
        self.assertEqual(at.text_input[0].value, '')
        self.assertFalse(at.exception)

    def test_urgent_notice_persists_without_medicine_expanders(self):
        at = self.app()
        at.chat_input[0].set_value('chest pain').run()
        at.chat_input[0].set_value('cough and fever').run()
        key = at.session_state['active_chat']
        self.assertTrue(at.session_state['chats'][key]['state']['urgent'])
        self.assertFalse(any('Medicine information' in e.label for e in list(at.expander) + list(at.status)))
        self.assertTrue(at.error)
        self.assertFalse(at.exception)

    def test_analytics_empty_then_records_real_activity_once(self):
        at = self.app()
        click(at, 'Analytics')
        self.assertEqual(at.metric[0].value, '0')
        click(at, 'New conversation')
        at.chat_input[0].set_value('runny nose, sneezing and cough').run()
        click(at, 'Analytics')
        self.assertFalse(at.exception)
        self.assertEqual([m.value for m in at.metric[:3]], ['1', '1', '1'])
        at.run()
        self.assertEqual(at.metric[2].value, '1')
        self.assertTrue(any(t.label == 'Medicine references' for t in at.tabs))

    def test_context_summary_uses_previous_messages(self):
        at = self.app()
        at.chat_input[0].set_value('runny nose, sneezing and cough').run()
        click(at, 'Summarize my symptoms')
        key = at.session_state['active_chat']
        message = at.session_state['chats'][key]['messages'][-1]
        self.assertEqual(message['intent'], 'summary')
        self.assertIn('runny nose', message['content'])
        self.assertFalse(at.exception)

    def test_edit_last_message_replaces_reply_and_restores_state(self):
        at = self.app()
        at.chat_input[0].set_value('chest pain').run()
        key = at.session_state['active_chat']
        self.assertTrue(at.session_state['chats'][key]['state']['urgent'])
        # A chat started by an older app version lacks this snapshot.
        at.session_state['chats'][key]['messages'][0].pop('state_before')
        at.run()
        click(at, 'Edit last prompt')
        next(t for t in at.text_area if t.label == 'Edit your last message').set_value('headache after sex')
        click(at, 'Save and regenerate')
        chat = at.session_state['chats'][key]
        self.assertFalse(at.exception)
        self.assertEqual(len(chat['messages']), 2)
        self.assertEqual(chat['messages'][0]['content'], 'headache after sex')
        self.assertEqual(chat['messages'][1]['intent'], 'sex_headache')
        self.assertEqual(chat['state']['symptoms'], ['headache'])
        self.assertFalse(chat['state']['urgent'])
        from src.analytics import summarize
        self.assertEqual(summarize()['responses'], 1)
        self.assertEqual(summarize()['urgent_conversations'], 0)


if __name__ == '__main__':
    unittest.main()
