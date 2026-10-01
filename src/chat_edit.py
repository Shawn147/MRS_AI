"""Restore the state before the latest user turn so it can be replaced."""
from copy import deepcopy

from src.dialogue import new_state, respond

REPORT_PROMPT = 'Please summarize and explain my attached medical reports.'


def turn_attachments(message, state):
    """Recover this turn's report text, including older saved chat formats."""
    reports = message.get('attachment_reports')
    if isinstance(reports, list) and reports:
        return deepcopy(reports)
    available = (message.get('state_before') or {}).get('medical_files') or state.get('medical_files', [])
    names = message.get('attachments', [])
    if not names and message.get('content') == REPORT_PROMPT:
        return deepcopy(available)
    remaining = list(available)
    selected = []
    for name in names:
        for index in range(len(remaining) - 1, -1, -1):
            if remaining[index]['name'] == name:
                selected.append(deepcopy(remaining.pop(index)))
                break
    return selected


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


def replace_last_turn(chat, data=None, predictor=None, *, attachments=None):
    messages = chat['messages']
    if len(messages) < 2 or messages[-2]['role'] != 'user' or messages[-1]['role'] != 'assistant':
        raise ValueError('There is no completed user turn to edit.')
    previous = messages[-2].get('state_before')
    if previous is None:
        previous = _replay_prior_messages(messages[:-2], data, predictor, chat.get('profile'))
    old_event_id = messages[-1].get('id')
    original = messages[-2]
    original_reports = turn_attachments(original, chat['state'])
    reports = original_reports if attachments is None else deepcopy(attachments)
    chat['replacement_turn'] = {
        'content': original['content'], 'attachment_reports': reports,
        'attachments': ((original.get('attachments') or [report['name'] for report in reports])
                        if attachments is None else [report['name'] for report in reports]),
        'file_only': bool(reports) and original.get('file_only', original['content'] == REPORT_PROMPT),
    }
    chat['state'] = deepcopy(previous)
    if attachments is not None:
        removed = [r for r in original_reports if r not in reports]
        prior_reports = [r for m in messages[:-2] for r in m.get('attachment_reports', [])]
        chat['state']['medical_files'] = [r for r in chat['state'].get('medical_files', [])
                                          if r not in removed or r in prior_reports]
    chat['messages'] = messages[:-2]
    if not chat['messages']:
        chat['title'] = 'New conversation'
    return old_event_id
