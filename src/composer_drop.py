"""Accept bounded drag-and-drop files into the unsent composer draft."""
import base64
import binascii
from pathlib import Path
from uuid import uuid4

import streamlit.components.v1 as components

from src.medical_files import MAX_BYTES, MAX_FILES, read_report

_component = components.declare_component(
    'mrs_ai_composer_drop', path=str(Path(__file__).resolve().parents[1] / 'assets/file_drop'))


def composer_drop(chat, disabled=False):
    return _component(chat_id=chat['id'], disabled=disabled,
                      remaining=max(0, MAX_FILES - len(chat.get('draft_files', []))),
                      max_bytes=MAX_BYTES, key='drop_' + chat['id'], default=None)


def accept_drop(chat, event, disabled=False):
    """Validate an event once; a failed batch leaves existing drafts intact."""
    if not isinstance(event, dict) or event.get('type') not in {'files', 'error'}:
        return False
    ident = event.get('id')
    if (not isinstance(ident, str) or not ident or event.get('chat_id') != chat['id']
            or ident == chat.get('drop_event_id')):
        return False
    chat['drop_event_id'] = ident
    if disabled:
        return False
    chat.pop('drop_error', None)
    if event['type'] == 'error':
        chat['drop_error'] = str(event.get('message') or 'This file could not be added. Please try the + button.')[:250]
        return True
    try:
        files = event.get('files')
        drafts = chat.get('draft_files', [])
        if not isinstance(files, list) or not files:
            raise ValueError('Drop a PDF, TXT or report photo to attach it.')
        if len(drafts) + len(files) > MAX_FILES:
            raise ValueError('Send up to 3 files at a time. Remove a file before adding another.')
        reports = []
        for file in files:
            if not isinstance(file, dict) or not isinstance(file.get('name'), str):
                raise ValueError('This file could not be added. Please try the + button.')
            encoded = file.get('content')
            if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_BYTES + 2) // 3):
                raise ValueError('Each file must be nonempty and no larger than 10 MB.')
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError('This file could not be read. Please try the + button.') from error
            reports.append(dict(read_report(file['name'], content), id=uuid4().hex))
        chat['draft_files'] = drafts + reports
    except (ValueError, ImportError) as error:
        chat['drop_error'] = str(error)
    return True
