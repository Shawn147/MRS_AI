import unittest
from copy import deepcopy
from unittest.mock import Mock

from src.chat_edit import replace_last_turn
from src.data import load_data
from src.dialogue import new_state


class ChatEditTests(unittest.TestCase):
    def test_restores_state_before_latest_turn(self):
        before = new_state()
        before['symptoms'] = ['cough']
        chat = {'title': 'Cough', 'state': deepcopy(before), 'messages': [
            {'role': 'user', 'content': 'cough', 'state_before': new_state()},
            {'role': 'assistant', 'content': 'Question', 'id': 'first'},
            {'role': 'user', 'content': 'sudden severe headache during sex', 'state_before': deepcopy(before)},
            {'role': 'assistant', 'content': 'Urgent', 'id': 'second'},
        ]}
        chat['state']['urgent'] = True
        chat['state']['symptoms'].append('headache')
        self.assertEqual(replace_last_turn(chat), 'second')
        self.assertEqual(chat['state'], before)
        self.assertEqual(len(chat['messages']), 2)
        self.assertEqual(chat['messages'][-1]['id'], 'first')

    def test_first_turn_edit_resets_title(self):
        chat = {'title': 'Old text', 'state': new_state(), 'messages': [
            {'role': 'user', 'content': 'Old text', 'state_before': new_state()},
            {'role': 'assistant', 'content': 'Answer'},
        ]}
        replace_last_turn(chat)
        self.assertEqual(chat['title'], 'New conversation')
        self.assertEqual(chat['messages'], [])

    def test_older_chat_without_snapshots_replays_prior_turns(self):
        chat = {'title': 'Cough', 'profile': {}, 'state': new_state(), 'messages': [
            {'role': 'user', 'content': 'cough'},
            {'role': 'assistant', 'content': 'Question'},
            {'role': 'user', 'content': 'headache after sex'},
            {'role': 'assistant', 'content': 'Later answer', 'id': 'old-event'},
        ]}
        predictor = Mock(return_value=[
            {'condition': 'Common Cold', 'probability': .9},
            {'condition': 'Allergy', 'probability': .06},
            {'condition': 'Pneumonia', 'probability': .02},
        ])
        self.assertEqual(replace_last_turn(chat, load_data(), predictor), 'old-event')
        self.assertEqual(chat['state']['symptoms'], ['cough'])
        self.assertEqual(len(chat['messages']), 2)


if __name__ == '__main__':
    unittest.main()
