import unittest

from src.browser_history import valid_history


class BrowserHistoryTests(unittest.TestCase):
    def test_accepts_saved_chat_and_active_selection(self):
        chat = {'id': 'a', 'title': 'Headache', 'created_at': '2026-09-30T00:00:00+00:00',
                'messages': [{'role': 'user', 'content': 'headache'}], 'state': {}, 'profile': {}}
        self.assertEqual(valid_history({'version': 1, 'chats': {'a': chat}, 'active_chat': 'a'}),
                         ({'a': chat}, 'a'))

    def test_rejects_malformed_history(self):
        self.assertIsNone(valid_history({'version': 1, 'chats': {'a': {'id': 'b'}},
                                         'active_chat': 'a'}))
        self.assertIsNone(valid_history({'version': 2, 'chats': {}}))


if __name__ == '__main__':
    unittest.main()
