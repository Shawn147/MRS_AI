import base64
import unittest

from src.composer_drop import accept_drop
from src.medical_files import MAX_BYTES


def file(name='sample.txt', text=b'Fictional lab report: Hemoglobin 10.2 g/dL'):
    return {'name': name, 'content': base64.b64encode(text).decode()}


class ComposerDropTests(unittest.TestCase):
    def setUp(self):
        self.chat = {'id': 'chat-one', 'draft_files': []}

    def event(self, files=None, ident='drop-one'):
        return {'type': 'files', 'chat_id': 'chat-one', 'id': ident, 'files': files or [file()]}

    def test_accepts_files_once_without_sending_and_retains_existing_drafts(self):
        self.chat['draft_files'] = [{'id': 'old', 'name': 'old.txt', 'text': 'Old report'}]
        event = self.event()
        self.assertTrue(accept_drop(self.chat, event))
        self.assertEqual(len(self.chat['draft_files']), 2)
        self.assertIn('Hemoglobin 10.2', self.chat['draft_files'][1]['text'])
        self.assertFalse(accept_drop(self.chat, event))
        self.assertEqual(len(self.chat['draft_files']), 2)
        self.assertNotIn('messages', self.chat)

    def test_rejects_whole_batch_and_preserves_drafts(self):
        existing = [{'id': 'old', 'name': 'old.txt', 'text': 'Old report'}]
        self.chat['draft_files'] = existing.copy()
        self.assertTrue(accept_drop(self.chat, self.event([file(), file('invalid.exe')])))
        self.assertEqual(self.chat['draft_files'], existing)
        self.assertIn('Supported files', self.chat['drop_error'])

    def test_checks_count_encoding_size_and_sensitive_filename(self):
        cases = [([file()] * 4, '3 files'),
                 ([{'name': 'sample.txt', 'content': '%%%'}], 'could not be read'),
                 ([file(text=b'x' * (MAX_BYTES + 1))], '10 MB'),
                 ([file('npm_recovery_codes.txt')], 'account or recovery-code')]
        for index, (files, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                self.assertTrue(accept_drop(self.chat, self.event(files, str(index))))
                self.assertIn(expected, self.chat['drop_error'])
                self.assertEqual(self.chat['draft_files'], [])

    def test_wrong_chat_and_disabled_events_cannot_attach_later(self):
        event = self.event()
        event['chat_id'] = 'other-chat'
        self.assertFalse(accept_drop(self.chat, event))
        event['chat_id'] = self.chat['id']
        self.assertFalse(accept_drop(self.chat, event, disabled=True))
        self.assertFalse(accept_drop(self.chat, event))
        self.assertEqual(self.chat['draft_files'], [])

    def test_browser_errors_are_consumed_once(self):
        event = {'type': 'error', 'chat_id': 'chat-one', 'id': 'error-one',
                 'message': 'Supported files: PDF, TXT, PNG and JPEG.'}
        self.assertTrue(accept_drop(self.chat, event))
        self.assertIn('Supported files', self.chat['drop_error'])
        self.assertFalse(accept_drop(self.chat, event))
