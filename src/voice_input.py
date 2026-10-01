"""Browser speech recognition; only a reviewed transcript becomes a chat message."""
from pathlib import Path
import streamlit.components.v1 as components

_component = components.declare_component(
    'mrs_ai_voice_input', path=str(Path(__file__).resolve().parents[1] / 'assets/voice_input'))


def voice_input(chat_id, disabled=False):
    return _component(disabled=disabled, language='en-PK', key='voice_' + chat_id, default=None)


def accept_transcript(chat, event):
    """Consume browser events once, enforcing the same text limit as typed messages."""
    if not isinstance(event, dict) or event.get('type') != 'transcript':
        return False
    ident, text = event.get('id'), event.get('text')
    if not isinstance(ident, str) or not isinstance(text, str) or not text.strip():
        return False
    if ident == chat.get('voice_event_id'):
        return False
    chat['voice_event_id'] = ident
    chat['voice_draft'] = text.strip()[:2000]
    return True
