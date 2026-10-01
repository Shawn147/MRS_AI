import unittest
import base64
import os
import tempfile
from urllib.error import HTTPError
from unittest.mock import Mock
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    return at


def send(at, text):
    next(field for field in at.text_input if field.label == 'Message').set_value(text)
    return click(at, 'Send')


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
        send(at, 'I have a runny nose, sneezing and cough for two days')
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
        send(at, 'chest pain')
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertTrue(at.session_state['chats'][second]['state']['urgent'])
        self.assertFalse(at.session_state['chats'][first]['state']['urgent'])
        at.button(key='open_' + first).click().run()
        self.assertEqual(at.session_state['active_chat'], first)
        self.assertEqual(len(at.session_state['chats'][first]['messages']), 2)

    def test_clear_saved_conversations_resets_session(self):
        at = self.app()
        send(at, 'cough')
        click(at, 'Conversation history')
        click(at, 'Clear saved conversations')
        self.assertEqual(len(at.session_state['chats']), 1)
        active = at.session_state['active_chat']
        self.assertEqual(at.session_state['chats'][active]['messages'], [])
        self.assertFalse(at.exception)

    def test_hospital_dataset_preview_separates_research_from_training(self):
        at = self.app()
        click(at, 'About & sources')
        click(at, 'Dataset Preview')
        metrics = {metric.label: metric.value for metric in at.metric}
        self.assertEqual(metrics['Unique patterns'], '3040')
        self.assertEqual(metrics['Distinct observations'], '3900')
        self.assertTrue(any('not used by the symptom classifier' in item.value for item in at.info))
        for index in range(4):
            next(s for s in at.selectbox if s.label == 'Hospital dataset').select_index(index).run()
            self.assertFalse(at.exception)
            self.assertTrue(any(b.label == 'Download clinical observations' for b in at.get('download_button')))

    def test_starter_and_followup_are_functional(self):
        at = self.app()
        click(at, 'I have a cough and a sore throat')
        key = at.session_state['active_chat']
        self.assertEqual(len(at.session_state['chats'][key]['messages']), 2)
        click(at, 'New conversation')
        send(at, 'cough')
        key = at.session_state['active_chat']
        pending = at.session_state['chats'][key]['state']['pending']
        self.assertIsNotNone(pending)
        click(at, 'Yes, I do')
        self.assertIn(pending, at.session_state['chats'][key]['state']['symptoms'])
        self.assertEqual(len(at.session_state['chats'][key]['messages']), 4)
        self.assertFalse(at.exception)

    def test_context_hides_previous_medicines_and_is_isolated(self):
        at = self.app()
        send(at, 'runny nose, sneezing and cough for two days')
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

    def test_denied_followups_render_results_with_message(self):
        at = self.app()
        send(at, 'I have a headache and feel tired')
        at.run()  # Refresh after switching from the welcome composer to the chat composer.
        for prompt in ['around 1 week', 'no', 'no']:
            send(at, prompt)
        self.assertFalse(at.exception)
        key = at.session_state['active_chat']
        answer = at.session_state['chats'][key]['messages'][-1]
        self.assertTrue(answer['result_ready'])
        self.assertTrue(any('Available symptom matches' in item.value for item in at.markdown))
        self.assertTrue(any('available dataset matches' in item.value for item in at.info))
        self.assertTrue(any(item.label == answer['predictions'][0]['condition'] for item in at.expander))
        self.assertFalse(any('Medicine information' in item.label for item in at.expander))

    def test_attachment_menu_and_report_explanation(self):
        at = self.app()
        self.assertTrue(any(item.label == 'Choose files' for item in at.get('file_uploader')))
        key = at.session_state['active_chat']
        at.session_state['chats'][key]['draft_files'] = [
            {'id': 'sample', 'name': 'lab.txt', 'text': '[Text file]\nHemoglobin 12.4 g/dL'}]
        at.run()
        with patch('src.general_qa.answer_general_question', return_value={
                'text': 'Your report lists hemoglobin as 12.4 g/dL (lab.txt).',
                'predictions': [], 'intent': 'general_question'}):
            click(at, 'Send')
        chat = at.session_state['chats'][key]
        self.assertEqual(chat['messages'][-2]['attachments'], ['lab.txt'])
        self.assertTrue(chat['messages'][-2]['file_only'])
        self.assertFalse(chat['draft_files'])
        self.assertFalse(any(button.label in {'Attach files', 'Explain attached reports'} for button in at.button))
        self.assertIn('12.4 g/dL', chat['messages'][-1]['content'])
        self.assertFalse(at.exception)

    def test_file_and_message_send_together(self):
        at = self.app()
        key = at.session_state['active_chat']
        at.session_state['chats'][key]['draft_files'] = [
            {'id': 'demo', 'name': 'lab.txt', 'text': 'Hemoglobin 12.4 g/dL'}]
        at.run()
        self.assertFalse(at.session_state['chats'][key]['state'].get('medical_files'))
        with patch('src.general_qa.answer_general_question', return_value={
                'text': 'Report explanation', 'predictions': []}) as answer:
            send(at, 'Explain the hemoglobin value')
        chat = at.session_state['chats'][key]
        self.assertEqual(chat['messages'][-2]['attachments'], ['lab.txt'])
        self.assertEqual(chat['messages'][-2]['content'], 'Explain the hemoglobin value')
        self.assertFalse(chat['messages'][-2]['file_only'])
        self.assertEqual(answer.call_args.args[1]['medical_files'][0]['name'], 'lab.txt')

    def test_file_remains_attached_when_editing_and_retrying(self):
        at = self.app()
        key = at.session_state['active_chat']
        report = {'name': 'lab.txt', 'text': 'Fictional hemoglobin 10.2 g/dL'}
        at.session_state['chats'][key]['draft_files'] = [dict(report, id='edit-file')]
        at.run()
        response = {'text': 'Report explanation', 'predictions': [], 'intent': 'general_question', 'retryable': True}
        with patch('src.general_qa.answer_general_question', return_value=response) as answer:
            click(at, 'Send')
            chat = at.session_state['chats'][key]
            self.assertTrue(chat['messages'][-2]['file_only'])
            self.assertEqual(chat['messages'][-2]['attachment_reports'], [report])
            self.assertTrue(any('sent-file-card' in m.value and 'lab.txt' in m.value for m in at.markdown))
            click(at, 'Edit last prompt')
            self.assertTrue(any('sent-file-card' in m.value and 'lab.txt' in m.value for m in at.markdown))
            next(t for t in at.text_area if t.label == 'Edit your last message').set_value('Explain the hemoglobin value')
            click(at, 'Save and regenerate')
            chat = at.session_state['chats'][key]
            self.assertEqual(chat['messages'][-2]['attachments'], ['lab.txt'])
            self.assertEqual(chat['messages'][-2]['attachment_reports'], [report])
            self.assertEqual(chat['messages'][-2]['content'], 'Explain the hemoglobin value')
            self.assertFalse(chat['messages'][-2]['file_only'])
            self.assertEqual(answer.call_args.args[1]['medical_files'], [report])
            click(at, 'Retry answer')
        chat = at.session_state['chats'][key]
        self.assertEqual(len(chat['messages']), 2)
        self.assertEqual(chat['messages'][-2]['attachments'], ['lab.txt'])
        self.assertEqual(chat['state']['medical_files'], [report])
        self.assertTrue(any('sent-file-card' in m.value and 'lab.txt' in m.value for m in at.markdown))
        self.assertFalse(at.exception)

    def test_removing_draft_allows_text_only_send(self):
        at = self.app()
        key = at.session_state['active_chat']
        at.session_state['chats'][key]['draft_files'] = [
            {'id': 'demo', 'name': 'lab.txt', 'text': 'Hemoglobin 12.4 g/dL'}]
        at.run()
        click(at, 'Remove file')
        send(at, 'cough')
        chat = at.session_state['chats'][key]
        self.assertFalse(chat['messages'][-2]['attachments'])
        self.assertFalse(chat['state'].get('medical_files'))
        self.assertIn('cough', chat['state']['symptoms'])

    def test_edit_file_close_preserves_text_cancel_and_removes_on_save(self):
        at = self.app()
        key = at.session_state['active_chat']
        first = {'name': 'first.txt', 'text': 'Fictional first report'}
        second = {'name': 'second.txt', 'text': 'Fictional second report'}
        at.session_state['chats'][key]['draft_files'] = [dict(first, id='first'), dict(second, id='second')]
        at.run()
        with patch('src.general_qa.answer_general_question', return_value={
                'text': 'Report explanation', 'predictions': [], 'intent': 'general_question'}) as answer:
            click(at, 'Send')
            click(at, 'Edit last prompt')
            next(t for t in at.text_area if t.label == 'Edit your last message').set_value('Explain the remaining report').run()
            ident = at.session_state['chats'][key]['edit_files'][0]['id']
            at.button(key='remove_draft_' + ident).click().run()
            self.assertEqual(next(t for t in at.text_area if t.label == 'Edit your last message').value,
                             'Explain the remaining report')
            self.assertEqual([r['name'] for r in at.session_state['chats'][key]['edit_files']], ['second.txt'])
            click(at, 'Cancel edit')
            self.assertEqual(at.session_state['chats'][key]['messages'][-2]['attachments'], ['first.txt', 'second.txt'])
            click(at, 'Edit last prompt')
            ident = at.session_state['chats'][key]['edit_files'][0]['id']
            at.button(key='remove_draft_' + ident).click().run()
            next(t for t in at.text_area if t.label == 'Edit your last message').set_value('Explain only the second report')
            click(at, 'Save and regenerate')
            self.assertEqual(answer.call_args.args[1]['medical_files'], [second])
        chat = at.session_state['chats'][key]
        self.assertEqual(chat['messages'][-2]['attachments'], ['second.txt'])
        self.assertEqual(chat['messages'][-2]['content'], 'Explain only the second report')
        self.assertEqual(chat['state']['medical_files'], [second])
        self.assertFalse(at.exception)

    def test_only_new_reply_has_entrance_animation(self):
        at = self.app()
        send(at, 'cough')
        key = at.session_state['active_chat']
        self.assertEqual(len(at.session_state['chats'][key]['messages']), 2)
        self.assertTrue(any('class="message-author message-enter-assistant"' in item.value for item in at.markdown))
        at.run()
        self.assertFalse(any('class="message-author message-enter-assistant"' in item.value for item in at.markdown))
        self.assertEqual(len(at.session_state['chats'][key]['messages']), 2)

    def test_returning_to_chat_reveals_last_message_once(self):
        at = self.app()
        send(at, 'cough')
        key = at.session_state['active_chat']
        last_id = at.session_state['chats'][key]['messages'][-1]['id']
        with patch('src.ui.scroll_to_message') as scroll:
            click(at, 'Health context')
            click(at, 'Back to conversation')
            scroll.assert_called_once_with('chat-message-' + last_id, smooth=False)
            scroll.reset_mock()
            at.run()
            scroll.assert_not_called()
        self.assertFalse(at.exception)

    def test_urgent_notice_persists_without_medicine_expanders(self):
        at = self.app()
        send(at, 'chest pain')
        send(at, 'cough and fever')
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
        send(at, 'runny nose, sneezing and cough')
        click(at, 'Analytics')
        self.assertFalse(at.exception)
        self.assertEqual([m.value for m in at.metric[:3]], ['1', '1', '1'])
        at.run()
        self.assertEqual(at.metric[2].value, '1')
        self.assertTrue(any(t.label == 'Medicine references' for t in at.tabs))

    def test_context_summary_uses_previous_messages(self):
        at = self.app()
        send(at, 'runny nose, sneezing and cough')
        send(at, 'Summarize my symptoms')
        key = at.session_state['active_chat']
        message = at.session_state['chats'][key]['messages'][-1]
        self.assertEqual(message['intent'], 'summary')
        self.assertIn('runny nose', message['content'])
        self.assertFalse(any(button.label == 'Summarize my symptoms' for button in at.button))
        at.run()
        self.assertFalse(any(button.label == 'Summarize my symptoms' for button in at.button))
        self.assertFalse(at.exception)

    def test_more_options_is_removed(self):
        at = self.app()
        send(at, 'runny nose, sneezing and cough for two days')
        self.assertFalse(any(item.proto.popover.label == 'More options' for item in at.get('popover')))
        self.assertFalse(any(button.label in {'Summarize my symptoms', 'Explain the previous result'} for button in at.button))

    def test_voice_transcript_is_editable_removable_and_can_send_alone(self):
        at = self.app()
        key = at.session_state['active_chat']
        with patch('src.voice_input.voice_input', return_value={'type': 'transcript', 'id': 'speech1', 'text': 'cough'}):
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state['chats'][key]['voice_draft'], 'cough')
            click(at, 'Remove voice transcript')
            self.assertNotIn('voice_draft', at.session_state['chats'][key])
            self.assertEqual(len(at.session_state['chats'][key]['messages']), 0)
        with patch('src.voice_input.voice_input', return_value={'type': 'transcript', 'id': 'speech2', 'text': 'fever'}):
            at.run()
            field = next(t for t in at.text_area if t.label == 'Voice transcript · review before sending')
            field.set_value('headache').run()
            click(at, 'Send')
        chat = at.session_state['chats'][key]
        self.assertEqual(chat['messages'][-2]['content'], 'headache')
        self.assertNotIn('voice_draft', chat)
        self.assertFalse(at.exception)

    def test_dropped_file_preserves_message_can_remove_and_send(self):
        at = self.app()
        key = at.session_state['active_chat']
        next(field for field in at.text_input if field.label == 'Message').set_value('Explain the flagged values').run()
        event = {'type': 'files', 'id': 'drop1', 'chat_id': key,
                 'files': [{'name': 'fictional-report.txt', 'content': base64.b64encode(
                     b'Fictional report: Hemoglobin 10.2 g/dL; sample range 12-16; LOW').decode()}]}
        with patch('src.composer_drop.composer_drop', return_value=event):
            at.run()
            chat = at.session_state['chats'][key]
            self.assertEqual(len(chat['draft_files']), 1)
            self.assertEqual(len(chat['messages']), 0)
            self.assertEqual(next(t for t in at.text_input if t.label == 'Message').value, 'Explain the flagged values')
            click(at, 'Remove file')
            self.assertEqual(chat['draft_files'], [])
            self.assertEqual(len(chat['messages']), 0)
        event['id'] = 'drop2'
        with patch('src.composer_drop.composer_drop', return_value=event), patch('src.ui.respond', return_value={
                'text': 'This fictional report flags hemoglobin as low.', 'intent': 'general'}):
            at.run()
            click(at, 'Send')
        chat = at.session_state['chats'][key]
        self.assertEqual(chat['messages'][-2]['content'], 'Explain the flagged values')
        self.assertEqual(chat['messages'][-2]['attachments'], ['fictional-report.txt'])
        self.assertEqual(chat['draft_files'], [])
        self.assertFalse(at.exception)

    def test_retry_replaces_failed_general_answer(self):
        settings = {'MRS_LLM_BACKEND': 'openai', 'MRS_LLM_MODEL': 'test-model',
                    'MRS_LLM_URL': 'https://example.test/chat', 'MRS_LLM_API_KEY': 'test-key'}
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read = Mock(return_value=b'{"choices":[{"message":{"content":"A helpful answer."}}]}')
        error = HTTPError(settings['MRS_LLM_URL'], 429, 'Rate limited', {}, None)
        with patch.dict(os.environ, settings), patch('src.general_qa.urlopen', side_effect=[error, response]):
            at = self.app()
            send(at, 'How can I stay healthy?')
            key = at.session_state['active_chat']
            self.assertTrue(at.session_state['chats'][key]['messages'][-1]['retryable'])
            click(at, 'Retry answer')
        messages = at.session_state['chats'][key]['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]['content'], 'A helpful answer.')
        self.assertFalse(at.exception)

    def test_edit_last_message_replaces_reply_and_restores_state(self):
        at = self.app()
        send(at, 'chest pain')
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
