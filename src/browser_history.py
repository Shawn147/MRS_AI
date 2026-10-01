"""Bridge between Streamlit session state and this browser's local storage."""

from pathlib import Path

import streamlit.components.v1 as components

STORAGE_KEY = 'mrs_ai_conversations_v1'
_component = components.declare_component(
    'mrs_ai_browser_history', path=str(Path(__file__).resolve().parents[1] / 'assets' / 'browser_history')
)


def read_history():
    return _component(action='read', storage_key=STORAGE_KEY, key='history_reader', default=None)


def write_history(chats, active_chat, clear=False):
    return _component(action='clear' if clear else 'write', storage_key=STORAGE_KEY,
                      payload={'version': 1, 'chats': chats, 'active_chat': active_chat},
                      key='history_writer', default=None)


def scroll_to_message(anchor, smooth=True, request_id=None):
    """Ask the browser bridge to reveal a newly inserted message."""
    return _component(action='scroll', anchor=anchor, smooth=smooth, request_id=request_id,
                      key='message_scroll_' + anchor + ('_new' if smooth else '_landing'), default=None)


def valid_history(value):
    if not isinstance(value, dict) or value.get('version') != 1:
        return None
    chats = value.get('chats')
    if not isinstance(chats, dict) or len(chats) > 100:
        return None
    cleaned = {}
    for key, chat in chats.items():
        if not isinstance(key, str) or not isinstance(chat, dict) or chat.get('id') != key:
            return None
        if not isinstance(chat.get('messages'), list) or not isinstance(chat.get('state'), dict):
            return None
        if not isinstance(chat.get('profile'), dict):
            return None
        if not isinstance(chat.get('title'), str) or not isinstance(chat.get('created_at'), str):
            return None
        cleaned[key] = chat
    active = value.get('active_chat')
    return (cleaned, active if active in cleaned else next(iter(cleaned), None))
