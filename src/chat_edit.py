"""Restore the state before the latest user turn so it can be replaced."""
from copy import deepcopy

from src.dialogue import new_state, respond


def _replay_prior_messages(messages, data, predictor, profile):
    """Recover state for chats created before per-turn snapshots were available."""
    if data is None or predictor is None:
        raise ValueError('Earlier messages need conversation data to restore their state.')
    state = new_state()
    def no_generation(question, state, data):
        return {'text': '', 'predictions': [], 'intent': 'general_question'}
    for message in messages:
        if message['role'] == 'user':
            respond(message['content'], state, data, predictor, profile, general_answer=no_generation)
    return state


def replace_last_turn(chat, data=None, predictor=None):
    messages = chat['messages']
    if len(messages) < 2 or messages[-2]['role'] != 'user' or messages[-1]['role'] != 'assistant':
        raise ValueError('There is no completed user turn to edit.')
    previous = messages[-2].get('state_before')
    if previous is None:
        previous = _replay_prior_messages(messages[:-2], data, predictor, chat.get('profile'))
    old_event_id = messages[-1].get('id')
    chat['state'] = deepcopy(previous)
    chat['messages'] = messages[:-2]
    if not chat['messages']:
        chat['title'] = 'New conversation'
    return old_event_id
