"""Streamlit interface for MRS AI BOT."""
import json
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path
import logging
import sqlite3
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.data import ARTIFACT_DIR, DATA_DIR, fingerprint, load_data
from src.regional_data import load_regional_preview
from src.dialogue import new_state, respond, profile_has_context
from src.browser_history import read_history, write_history, valid_history, scroll_to_message

MODEL_NAMES = {
    'transformer': 'MiniLM · fine-tuned transformer',
    'ml': 'TF-IDF · Logistic Regression',
}
EVAL_NAMES = {
    'ml': 'TF-IDF + Logistic Regression',
    'frozen': 'Frozen MiniLM + trained head',
    'transformer': 'Fine-tuned MiniLM transformer',
}
ASSETS = Path(__file__).resolve().parents[1] / 'assets'
LOGO = (ASSETS / 'logo.svg').read_text()


def model_privacy_notice():
    from src.general_qa import _model_setting, OLLAMA_URL
    from urllib.parse import urlparse
    backend = _model_setting('MRS_LLM_BACKEND', 'ollama').lower()
    host = urlparse(_model_setting('MRS_LLM_URL') or (OLLAMA_URL if backend == 'ollama' else '')).hostname
    if host in {'localhost', '127.0.0.1', '::1'}:
        return 'AI answers process your questions and readable report text using a model on this server.'
    provider = 'Groq' if host == 'api.groq.com' else 'OpenAI' if host == 'api.openai.com' else 'the configured AI provider'
    return 'Some questions and readable report text may be shared with ' + provider + ' to prepare an answer.'


@st.cache_data
def cached_data(version):
    return load_data()


@st.cache_resource
def cached_models(data_version, model_version):
    from src.models import load_bundle
    return load_bundle()


def create_chat():
    key = uuid4().hex
    st.session_state.chats[key] = {
        'id': key,
        'title': 'New conversation',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'messages': [],
        'state': new_state(),
        'profile': {},
        'model': 'transformer',
    }
    st.session_state.active_chat = key
    st.session_state.navigation = 'Chat'


def clear_saved_conversations():
    st.session_state.chats = {}
    create_chat()
    st.session_state.clear_browser_history = True


def choose_chat(key):
    st.session_state.active_chat = key
    st.session_state.navigation = 'Chat'
    st.session_state.chat_landing = True


def model_metrics(version):
    path = ARTIFACT_DIR / 'metrics.json'
    if not path.exists():
        raise FileNotFoundError('Saved models are missing. Run python scripts/train_chatbot.py.')
    metrics = json.loads(path.read_text())
    if metrics['data_fingerprint'] != version:
        raise ValueError('The JSON data has changed. Run python scripts/train_chatbot.py again.')
    return metrics


def navigate(page):
    st.session_state.navigation = page


def queue_prompt(text):
    st.session_state.queued_prompt = text


def queue_retry():
    st.session_state.retry_last = True


def begin_edit(chat):
    from src.chat_edit import turn_attachments
    chat['edit_files'] = [dict(report, id=uuid4().hex) for report in
                          turn_attachments(chat['messages'][-2], chat['state'])]
    st.session_state['edit_text_' + chat['id']] = chat['messages'][-2]['content']
    chat['editing_last'] = True


def cancel_edit(chat):
    chat['editing_last'] = False
    chat.pop('edit_files', None)


def sidebar(chat):
    with st.sidebar:
        st.markdown(f'<div class="brand">{LOGO}<div><strong>MRS AI</strong>'
                    '<span>Health information &amp; symptom guidance</span></div></div>', unsafe_allow_html=True)
        st.button('New conversation', icon=':material/add:', on_click=create_chat,
                  type='primary', use_container_width=True)
        with st.container(key='recent_chats'):
            chats = [item for item in reversed(list(st.session_state.chats.values())) if item['messages']]
            if not chats:
                st.markdown('<div class="eyebrow">Your conversations</div>', unsafe_allow_html=True)
                st.caption('Your conversations will appear here. Start with how you’re feeling.')
            last_group = None
            today = datetime.now(timezone.utc).date()
            for item in chats[:12]:
                age = (today - datetime.fromisoformat(item['created_at']).date()).days
                group = 'Today' if age == 0 else ('Previous 7 days' if age < 7 else 'Earlier')
                if group != last_group:
                    st.markdown(f'<div class="eyebrow">{group}</div>', unsafe_allow_html=True)
                    last_group = group
                st.button(item['title'][:44], key='open_' + item['id'], icon=':material/chat_bubble_outline:',
                          type='primary' if item['id'] == chat['id'] and st.session_state.navigation == 'Chat' else 'tertiary',
                          on_click=choose_chat, args=(item['id'],), use_container_width=True)
        with st.container(key='sidebar_links'):
            for label, page, icon in [('Health context', 'Health context', 'favorite'),
                                      ('Conversation history', 'History', 'history'),
                                      ('Analytics', 'Analytics', 'bar_chart'),
                                      ('About & sources', 'About', 'menu_book')]:
                st.button(label, icon=f':material/{icon}:', on_click=navigate, args=(page,),
                          use_container_width=True)
        st.markdown('<div class="sidebar-note">Conversations are saved in this browser. ' + escape(model_privacy_notice()) +
                    ' Please leave out names and other identifying details.</div>',
                    unsafe_allow_html=True)


def page_header():
    with st.container(key='topbar'):
        left, right = st.columns([3, 1])
        left.markdown('<div class="topline"><strong>MRS AI</strong><span>Symptom guidance</span></div>',
                      unsafe_allow_html=True)
        right.button('Health context', icon=':material/tune:', on_click=navigate,
                     args=('Health context',), use_container_width=True, key='header_context')


def welcome(chat):
    st.markdown(f'<div class="welcome"><div class="welcome-mark">{LOGO}</div>'
                '<div class="kicker">A little clarity starts here</div>'
                '<h1>How are you feeling today?</h1>'
                '<p>Describe your symptoms, and I’ll help you explore possible causes '
                'and understand the next steps.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span>A place to start</span><span>Choose one, or tell me in your own words</span></div>',
                unsafe_allow_html=True)
    with st.container(key='starters'):
        examples = [('I have a headache and feel tired', 'psychology'),
                    ('I have a cough and a sore throat', 'air'),
                    ('I have stomach pain and nausea', 'healing')]
        for col, (text, icon) in zip(st.columns(3), examples):
            col.button(text, icon=f':material/{icon}:', on_click=queue_prompt, args=(text,), use_container_width=True)
    st.caption('Privacy: ' + model_privacy_notice() + ' '
               'Please leave out names and other identifying details. Conversations are saved in this browser '
               'until you clear them. We keep anonymous activity counts.')


def symptom_tags(symptoms, data):
    if symptoms:
        labels = [escape(data['by_symptom'][s]['label']) for s in symptoms if s in data['by_symptom']]
        st.markdown('<div class="result-label">Symptoms you mentioned</div><div class="tags">' +
                    ''.join(f'<span class="tag">{label}</span>' for label in labels) + '</div>', unsafe_allow_html=True)


def render_sources(message):
    details = message.get('source_details') or [{'url': url} for url in message.get('sources', [])]
    for source in details:
        url = source['url']
        publisher = source.get('publisher') or ('NHS' if 'nhs.uk' in url else
                                                'Mayo Clinic' if 'mayoclinic.org' in url else 'Source')
        if url.startswith('https://'):
            st.link_button('Read ' + publisher + ' source', url)
            date = source.get('page_last_reviewed')
            if date:
                st.caption(f'{publisher} · page last reviewed {date}')
            elif source.get('updated_on'):
                st.caption(f'{publisher} · updated {source["updated_on"]}')
            if source.get('attribution'):
                st.caption(source['attribution'])


def sent_file_cards(names):
    if not names:
        return
    cards = []
    for name in names:
        kind = Path(name).suffix.lstrip('.').upper() or 'FILE'
        cards.append('<div class="sent-file-card"><div class="draft-file-icon">📄</div>'
                     f'<div class="draft-file-kind">{escape(kind)}</div>'
                     f'<div class="draft-file-name" title="{escape(name, quote=True)}">{escape(name)}</div></div>')
    st.markdown('<div class="sent-file-cards">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def duration_choices(chat, message):
    key = 'duration_' + chat['id'] + '_' + message['id']
    options = {'Today': 'Since today', '1–2 days': 'For 1–2 days',
               '3–7 days': 'For 3–7 days', 'More than a week': 'For more than one week',
               'Not sure': 'Not sure', 'Other': None}
    with st.container(key='duration_choices'):
        selected = st.radio('How long have you had these symptoms?', list(options),
                            index=None, horizontal=True, key=key)
        answer = options.get(selected, '')
        if selected == 'Other':
            answer = st.text_input('Other duration', placeholder='For example, since yesterday or about three weeks',
                                   max_chars=100, key=key + '_other').strip()
        st.button('Continue', key=key + '_submit', type='primary', disabled=not answer,
                  on_click=queue_prompt, args=(answer,))


def render_message(message, chat, data, editable=False, latest=False, animate=False):
    is_user = message['role'] == 'user'
    duration_slot = None
    with st.chat_message(message['role'], avatar=':material/person:' if is_user else str(ASSETS / 'logo.svg')):
        author = 'You' if is_user else 'MRS AI'
        time = datetime.fromisoformat(message['timestamp']).astimezone().strftime('%H:%M') if message.get('timestamp') else ''
        role_class = ' user-author' if is_user else ''
        if animate:
            role_class += ' message-enter-user' if is_user else ' message-enter-assistant'
        anchor = 'chat-message-' + message.get('id', '')
        st.markdown(f'<div id="{escape(anchor, quote=True)}" class="message-author{role_class}">{author}<span>{time}</span></div>', unsafe_allow_html=True)
        if is_user:
            from src.chat_edit import REPORT_PROMPT, turn_attachments
            names = message.get('attachments') or [r['name'] for r in turn_attachments(message, chat['state'])]
            sent_file_cards(names)
            if not message.get('file_only') and not (names and message['content'] == REPORT_PROMPT):
                st.markdown(message['content'])
            if editable:
                st.button('Edit last prompt', icon=':material/edit:', help='Edit message', key='edit_last_' + chat['id'],
                          on_click=begin_edit, args=(chat,))
            return
        if message.get('urgent'):
            st.error('**Please seek urgent medical assessment**\n\n' + message['content'], icon=':material/emergency:')
            render_sources(message)
            return
        if message.get('condition'):
            record = data['by_condition'][message['condition']]
            if message.get('correction_note'):
                st.caption(message['correction_note'])
            symptom_tags(message.get('symptoms', []), data)
            st.markdown('<div class="result-label">What this could suggest</div>'
                        f'<div class="result-title">{escape(record["name"])}</div>', unsafe_allow_html=True)
            st.write(record['description'])
            context = {**chat['state']['context'], **chat['profile']}
            withheld = message.get('medicine_withheld') or profile_has_context(context) or chat['state']['urgent']
            if record.get('sources'):
                with st.expander('Sources'):
                    render_sources({'source_details': record['sources']})
            if not withheld:
                with st.expander('Medicine information · educational reference', icon=':material/medication:'):
                    st.caption('These medicine names have not been checked by healthcare professionals. I can’t tell whether they would be safe or helpful for you.')
                    for medicine in record['medications'][:5]:
                        st.write('• ' + medicine)
                    st.caption('I can’t check the right dose, your allergies, or how these medicines may affect other medicines you take. '
                               'Discuss treatment with a qualified healthcare professional.')
            others = [p['condition'] for p in message.get('predictions', []) if p['condition'] != record['name']]
            if others:
                with st.expander('Other possible matches'):
                    st.write(' · '.join(others))
        elif message.get('result_ready'):
            symptom_tags(message.get('symptoms', []), data)
            st.markdown('### Possible causes')
            text = message['content']
            # Display older saved match notices with the current concise wording.
            if ('available dataset matches' in text or
                    'some possible causes based on' in text):
                text = 'Here are some possible causes based on your symptoms.'
            st.info(text, icon=':material/info:')
            for prediction in message.get('predictions', [])[:3]:
                record = data['by_condition'][prediction['condition']]
                with st.expander(record['name']):
                    st.write(record['description'])
            render_sources(message)
        elif message.get('uncertain'):
            st.info('**Let’s understand this a little better**\n\n' + message['content'], icon=':material/info:')
        else:
            st.markdown(message['content'])
            render_sources(message)
        if (latest and chat['state'].get('pending_detail') == 'duration'
                and not chat.get('editing_last') and not chat['state']['urgent']):
            duration_slot = st.empty()
            with duration_slot.container():
                duration_choices(chat, message)
        if latest and message.get('retryable'):
            st.button('Retry answer', icon=':material/refresh:', key='retry_' + chat['id'],
                      on_click=queue_retry)
    return duration_slot


def add_selected_files(chat, upload_key):
    """Stage selected uploads before rendering the composer on its next run."""
    uploads = st.session_state.get(upload_key, [])
    if not uploads:
        return
    from src.medical_files import read_report, MAX_FILES
    try:
        drafts = chat.get('draft_files', [])
        if len(drafts) + len(uploads) > MAX_FILES:
            raise ValueError('Send up to 3 files at a time. Remove a file before adding another.')
        reports = [dict(read_report(file.name, file.getvalue()), id=uuid4().hex) for file in uploads]
        chat['draft_files'] = drafts + reports
        chat['upload_version'] = chat.get('upload_version', 0) + 1
        chat.pop('drop_error', None)
    except (ValueError, ImportError) as error:
        chat['drop_error'] = str(error)


def attachment_menu(chat, disabled=False):
    with st.popover(r'\+', help='Choose a medical file', disabled=disabled):
        st.caption('PDF, TXT or report photo · up to 3 files, 10 MB each.')
        st.caption('Files are sent to the configured AI provider only when you press Send. '
                   'Readable contents are saved with this chat in your browser.')
        upload_key = 'medical_upload_' + chat['id'] + '_' + str(chat.get('upload_version', 0))
        st.file_uploader('Choose files', type=['pdf', 'txt', 'png', 'jpg', 'jpeg'],
                         accept_multiple_files=True, key=upload_key, disabled=disabled,
                         on_change=add_selected_files, args=(chat, upload_key))


def remove_file_draft(chat, report_id):
    chat['draft_files'] = [report for report in chat.get('draft_files', []) if report['id'] != report_id]


def remove_edit_file(chat, report_id):
    chat['edit_files'] = [report for report in chat.get('edit_files', []) if report['id'] != report_id]


def remove_voice_draft(chat):
    chat.pop('voice_draft', None)


def draft_file_cards(chat, editing=False):
    drafts = chat.get('edit_files' if editing else 'draft_files', [])
    if not drafts:
        return
    with st.container(key='edit_cards' if editing else 'draft_cards'):
        columns = st.columns([1, 1, 1, 5], gap='small')
        for index, report in enumerate(drafts):
            with columns[index], st.container(key='draft_tile_' + report['id']):
                kind = Path(report['name']).suffix.lstrip('.').upper() or 'FILE'
                st.markdown('<div class="draft-file-icon">📄</div>'
                            f'<div class="draft-file-kind">{escape(kind)}</div>'
                            f'<div class="draft-file-name" title="{escape(report["name"], quote=True)}">'
                            f'{escape(report["name"])}</div>', unsafe_allow_html=True)
                st.button('Remove file', icon=':material/close:', key='remove_draft_' + report['id'],
                          help='Remove ' + report['name'],
                          on_click=remove_edit_file if editing else remove_file_draft, args=(chat, report['id']))


def chat_page(chat, data, version, landing=False):
    page_header()
    if not chat.get('composer_files_v2'):
        sent_names = {name for message in chat['messages'] if message['role'] == 'user'
                      for name in message.get('attachments', [])}
        old_reports = chat['state'].get('medical_files', [])
        unsent = [dict(report, id=uuid4().hex) for report in old_reports if report['name'] not in sent_names]
        if unsent:
            chat['draft_files'] = (chat.get('draft_files', []) + unsent)[-3:]
            chat['state']['medical_files'] = [report for report in old_reports if report['name'] in sent_names]
        chat['composer_files_v2'] = True
    def predictor(symptoms):
        from src.models import predict
        model_metrics(version)
        bundle = cached_models(version, (ARTIFACT_DIR / 'metrics.json').stat().st_mtime_ns)
        return predict(bundle, symptoms, chat['model'])

    retry_prompt = None
    if (st.session_state.pop('retry_last', False) and len(chat['messages']) >= 2
            and chat['messages'][-1].get('retryable')):
        from src.chat_edit import replace_last_turn
        retry_prompt = chat['messages'][-2]['content']
        old_event_id = replace_last_turn(chat, data, predictor)
        try:
            from src.analytics import delete_event
            delete_event(old_event_id)
        except (OSError, sqlite3.Error):
            logging.exception('Unable to remove retried analytics event')
    pending_prompt = chat.pop('pending_prompt', None)
    animate_id = chat.pop('animate_message_id', None)
    if landing and chat['messages']:
        chat['messages'][-1].setdefault('id', uuid4().hex)
    thread = st.container(key='conversation_thread')
    with thread:
        if not chat['messages']:
            welcome(chat)
        else:
            st.markdown('<div class="conversation-heading"><h1>Let’s understand how you’re feeling</h1>'
                        '<p>You can add details or correct a symptom at any time.</p></div>', unsafe_allow_html=True)
        duration_slots = []
        for index, message in enumerate(chat['messages']):
            editable = (index == len(chat['messages']) - 2 and message['role'] == 'user')
            duration_slot = render_message(message, chat, data, editable=editable,
                           latest=index == len(chat['messages']) - 1,
                           animate=bool(animate_id and message.get('id') == animate_id))
            if duration_slot is not None:
                duration_slots.append(duration_slot)
        scroll_key = 'chat_scroll_' + chat['id']
        if animate_id:
            st.session_state[scroll_key] = ('chat-message-' + animate_id, True, uuid4().hex)
        elif chat['messages'] and (landing or scroll_key not in st.session_state):
            st.session_state[scroll_key] = ('chat-message-' + chat['messages'][-1]['id'], False, uuid4().hex)
        if scroll_key in st.session_state and chat['messages']:
            anchor, smooth, request_id = st.session_state[scroll_key]
            scroll_to_message(anchor, smooth=smooth, request_id=request_id)
    edited_prompt = None
    if chat.get('editing_last'):
        from src.chat_edit import turn_attachments, REPORT_PROMPT
        if 'edit_files' not in chat:
            chat['edit_files'] = [dict(report, id=uuid4().hex) for report in
                                  turn_attachments(chat['messages'][-2], chat['state'])]
        with st.container(key='edit_panel', border=True):
            with st.container(key='edit_file_slot'):
                if chat['edit_files']:
                    draft_file_cards(chat, editing=True)
                else:
                    st.empty()
            edit_key = 'edit_text_' + chat['id']
            if edit_key not in st.session_state:
                st.session_state[edit_key] = chat['messages'][-2]['content']
            revised = st.text_area('Edit your last message', key=edit_key, max_chars=2000)
            save_edit = st.button('Save and regenerate', type='primary')
        st.button('Cancel edit', on_click=cancel_edit, args=(chat,))
        if save_edit:
            if not revised.strip():
                st.warning('Enter a message before saving.')
            elif not chat['edit_files'] and revised.strip() == REPORT_PROMPT:
                st.warning('You removed all the files. Enter a new message before saving.')
            else:
                from src.chat_edit import replace_last_turn
                reports = [{key: report[key] for key in ('name', 'text')} for report in chat.pop('edit_files')]
                old_event_id = replace_last_turn(chat, data, predictor, attachments=reports)
                chat['editing_last'] = False
                st.session_state.pop('queued_prompt', None)
                edited_prompt = revised.strip()
                try:
                    from src.analytics import delete_event
                    delete_event(old_event_id)
                except (OSError, sqlite3.Error):
                    logging.exception('Unable to remove superseded analytics event')
    if not chat.get('editing_last') and chat['state']['pending'] and not chat['state']['urgent']:
        with st.container(key='followup'):
            for col, (label, prompt) in zip(st.columns(3), [('Yes, I do', 'yes'), ('No, I don’t', 'no'), ('Show matches', 'show results')]):
                col.button(label, on_click=queue_prompt, args=(prompt,), use_container_width=True)
    if chat['state']['urgent']:
        st.warning('Medicine suggestions are paused for this conversation. Contact local emergency services or seek urgent care.')
    # Preserve the composer's widget position when follow-up controls appear or disappear.
    st.empty()
    submitted, entered = False, ''
    if not chat.get('editing_last'):
        with st.container(key='chat_composer'):
            from src.composer_drop import composer_drop, accept_drop
            drop_disabled = bool(chat['state']['urgent'] or pending_prompt)
            with st.container(key='composer_drop'):
                drop_event = composer_drop(chat, disabled=drop_disabled)
            accept_drop(chat, drop_event, disabled=drop_disabled)
            with st.container(key='composer_notice_slot'):
                if chat.get('drop_error'):
                    st.warning(chat.pop('drop_error'))
                else:
                    st.empty()
            with st.container(key='composer_file_slot'):
                if chat.get('draft_files'):
                    draft_file_cards(chat)
                else:
                    st.empty()
            with st.container(key='composer_transcript_slot'):
                if chat.get('voice_draft'):
                    with st.container(key='voice_draft'):
                        transcript_column, close_column = st.columns([15, 1])
                        with transcript_column:
                            chat['voice_draft'] = st.text_area('Voice transcript · review before sending',
                                value=chat['voice_draft'], key='voice_draft_' + chat['id'] + '_' + chat['voice_event_id'],
                                max_chars=2000, height=68)
                        close_column.button('Remove voice transcript', icon=':material/close:', help='Delete the voice transcript',
                                            on_click=remove_voice_draft, args=(chat,))
                else:
                    st.empty()
            with st.container(key='composer_input_row'):
                attachment_column, input_column = st.columns([1, 15], gap='small')
                with attachment_column:
                    attachment_menu(chat, disabled=drop_disabled)
                with input_column:
                    from src.voice_input import voice_input, accept_transcript
                    with st.container(key='composer_voice'):
                        voice_event = voice_input(chat['id'], disabled=bool(pending_prompt))
                    if accept_transcript(chat, voice_event):
                        st.rerun()
                    with st.form('message_form_' + chat['id'], clear_on_submit=True, border=False):
                        entered = st.text_input('Message', placeholder='Describe your symptoms or drop a file…',
                                                key='message_' + chat['id'],
                                                max_chars=2000, label_visibility='collapsed',
                                                disabled=bool(pending_prompt))
                        submitted = st.form_submit_button('Send', icon=':material/arrow_upward:',
                                                          help='Send message or files', disabled=bool(pending_prompt))
    prompt = retry_prompt or edited_prompt or st.session_state.pop('queued_prompt', None)
    sending_files = chat.get('draft_files', []) if submitted else []
    file_only = submitted and not entered.strip() and bool(sending_files)
    if submitted:
        voice_text = chat.get('voice_draft', '').strip()
        combined = '\n'.join(part for part in [entered.strip(), voice_text] if part)
        if len(combined) > 2000:
            st.warning('Please shorten your message and voice transcript to 2,000 characters before sending.')
            return
        file_only = not combined and bool(sending_files)
        prompt = combined or ('Please summarize and explain my attached medical reports.' if sending_files else None)
    if prompt and not pending_prompt:
        for duration_slot in duration_slots:
            duration_slot.empty()
        if submitted:
            chat.pop('voice_draft', None)
        replacement = chat.pop('replacement_turn', {}) if (retry_prompt or edited_prompt) else {}
        reports = (replacement.get('attachment_reports', []) if replacement else
                   [{key: report[key] for key in ('name', 'text')} for report in sending_files])
        attached_names = replacement.get('attachments', []) if replacement else [report['name'] for report in reports]
        if replacement:
            file_only = bool(replacement.get('file_only') and prompt == replacement['content'])
        state_before = deepcopy(chat['state'])
        if reports:
            existing = [r for r in chat['state'].get('medical_files', []) if r not in reports]
            chat['state']['medical_files'] = (existing + reports)[-3:]
        if sending_files:
            chat['draft_files'] = []
        user_message_id = uuid4().hex
        chat['messages'].append({'role': 'user', 'content': prompt, 'id': user_message_id,
                                 'state_before': state_before, 'file_only': file_only,
                                 'attachments': attached_names, 'attachment_reports': deepcopy(reports),
                                 'timestamp': datetime.now(timezone.utc).isoformat()})
        if chat['title'] == 'New conversation':
            chat['title'] = attached_names[0] if file_only else prompt[:64]
        chat['pending_prompt'] = prompt
        chat['animate_message_id'] = user_message_id
        st.rerun()
    if pending_prompt:
        prompt = pending_prompt
        try:
            with st.spinner('Preparing your response…'):
                answer = respond(prompt, chat['state'], data, predictor, chat['profile'])
            message = {
                **answer,
                'role': 'assistant', 'content': answer['text'],
                'symptoms': list(chat['state']['symptoms']),
                'model': chat['model'], 'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        except (FileNotFoundError, ValueError, OSError, ImportError):
            logging.exception('Unable to prepare symptom response')
            message = {'role': 'assistant', 'content': 'I’m unable to review your symptoms right now. '
                       'Please try again later. If you feel very unwell, seek medical advice.'}
        message['id'] = uuid4().hex
        chat['animate_message_id'] = message['id']
        chat['messages'].append(message)
        if 'text' in message:
            from src.analytics import record_event
            try:
                record_event(st.session_state.analytics_session, chat['id'], message, data)
            except (OSError, sqlite3.Error):
                logging.exception('Unable to save anonymous analytics event')
                st.session_state.analytics_write_failed = True
        st.rerun()


def analytics_page():
    from src.analytics import summarize
    page_header()
    st.title('Conversation analytics')
    st.write('A clear view of activity, symptom patterns and medicine references offered.')
    st.caption('These are anonymous activity counts, not patient records. One person can have several sessions. '
               'Counts may reset when the app restarts; earlier conversations are not included.')
    period = st.selectbox('Time period', ['Last 7 days', 'Last 30 days', 'All time'], index=1)
    days = {'Last 7 days': 7, 'Last 30 days': 30}.get(period)
    since = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)).isoformat() if days else None
    try:
        report = summarize(since)
    except (OSError, sqlite3.Error):
        st.error('Analytics is currently unavailable. Your conversations can continue.')
        return
    if st.session_state.get('analytics_write_failed'):
        st.warning('Some activity could not be saved. These totals may be incomplete.')
    for col, label, key in zip(st.columns(4), ['Anonymous sessions', 'Conversations', 'Assistant replies', 'Urgent conversations'],
                               ['sessions', 'conversations', 'responses', 'urgent_conversations']):
        with col.container(border=True):
            st.metric(label, report[key])
    if not report['responses']:
        st.info('No activity in this period yet. Start a conversation and your real usage will appear here.')
        return
    st.metric('General-answer service failures', report['model_failures'],
              help='Provider errors, timeouts, rate limits or invalid responses. No question text is stored in analytics.')
    overview, medicines, conversations = st.tabs(['Overview', 'Medicine references', 'Conversation activity'])
    with overview:
        st.subheader('Activity over time')
        frame = pd.DataFrame(report['daily']).set_index('Date (UTC)')
        start = since[:10] if since else frame.index.min()
        dates = pd.date_range(start, datetime.now(timezone.utc).date()).strftime('%Y-%m-%d')
        frame = frame.reindex(dates, fill_value=0)
        frame.index.name = 'Date (UTC)'
        st.bar_chart(frame, color='#087f73')
        left, right = st.columns(2)
        for col, title, key, index in [(left, 'Common condition matches', 'conditions', 'Condition match'),
                                        (right, 'Common symptoms', 'symptoms', 'Symptom')]:
            with col:
                st.subheader(title)
                if report[key]:
                    st.dataframe(pd.DataFrame(report[key]).head(10), hide_index=True, use_container_width=True)
                else:
                    st.caption('No matches recorded in this period.')
        st.caption('Condition matches are model outputs, not diagnoses. Symptoms are counted once per assistant response snapshot.')
        left, right = st.columns(2)
        left.metric('Medicine-withheld replies', report['withheld_responses'])
        right.metric('Uncertain assessment replies', report['uncertain_responses'])
    with medicines:
        st.subheader('Most frequently offered medicine references')
        st.caption('Counts reflect lists offered in eligible replies, even if the section was not opened. '
                   'They do not measure prescriptions, medicines taken, or effectiveness. '
                   'Later context changes do not rewrite historical counts.')
        if report['medicines']:
            frame = pd.DataFrame(report['medicines'])
            st.bar_chart(frame.head(10).set_index('Medicine reference')[['Responses']], color='#087f73')
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.download_button('Download medicine counts', frame.to_csv(index=False), 'medicine-reference-counts.csv', 'text/csv')
        else:
            st.info('No medicine references were offered in this period.')
    with conversations:
        st.subheader('Most active conversations')
        frame = pd.DataFrame(report['activity'])
        frame['Conversation'] = frame['Conversation'].map(lambda value: 'Conversation ' + value[:10])
        st.dataframe(frame.head(20), hide_index=True, use_container_width=True)
        st.caption('Ranked by assistant replies, not patient visits. Names, message text and health-profile values are not stored in analytics.')
        st.subheader('Reply types')
        st.dataframe(pd.DataFrame(report['intents']), hide_index=True, use_container_width=True)
    st.download_button('Download analytics summary', json.dumps(report, indent=2), 'mrs-analytics.json', 'application/json')


def health_context_page(chat):
    page_header()
    st.title('Your health context')
    st.write('A little context helps us know when medicine information should be withheld.')
    st.caption('Optional and specific to this conversation. This does not check medicine safety or interactions. '
               'If you ask a general question, relevant symptoms may be shared with Groq to provide an answer.')
    with st.form('context_' + chat['id']):
        profile = {}
        profile['allergies'] = st.text_input('Known medicine allergies', value=chat['profile'].get('allergies', ''), placeholder='For example, penicillin')
        profile['history'] = st.text_input('Existing medical conditions', value=chat['profile'].get('history', ''), placeholder='Any conditions you’d like to mention')
        profile['medicines'] = st.text_input('Current medicines', value=chat['profile'].get('medicines', ''), placeholder='Include any medicines or supplements')
        options = ['Not provided', 'No', 'Yes', 'Unsure']
        previous = chat['profile'].get('pregnancy', '')
        pregnancy = st.selectbox('Pregnant or breastfeeding?', options, index=options.index(previous) if previous in options else 0)
        profile['pregnancy'] = '' if pregnancy == 'Not provided' else pregnancy
        profile['preferences'] = st.text_input('Age or other context', value=chat['profile'].get('preferences', ''), placeholder='Anything else you’d like to share')
        st.caption('Adding health context can withhold all medicine names; it does not establish that other medicines are suitable.')
        if st.form_submit_button('Save health context', type='primary', use_container_width=True):
            chat['profile'] = profile
            st.success('Health context saved for this conversation.')
    st.button('Back to conversation', icon=':material/arrow_back:', on_click=navigate, args=('Chat',))


def dataset_page(data):
    st.title('Dataset Preview')
    st.write('Original educational data and labelled illustrative additions, with separate hospital research cohorts.')
    library = data.get('health_library', {})
    if library.get('manifest'):
        st.subheader('Expanded health reference library')
        for col, label, key in zip(st.columns(3),
                                   ['Condition topics', 'Scoped symptom phrases', 'Distinct medicine names'],
                                   ['conditions', 'symptoms', 'medicines']):
            col.metric(label, len(library[key]))
        st.caption('Reference coverage is separate from trained condition labels and symptom features. '
                   'Medicine records are US label excerpts; availability in Pakistan has not been verified.')
        chosen = st.selectbox('Reference library', ['conditions', 'symptoms', 'medicines'])
        st.dataframe(pd.DataFrame(library[chosen]), use_container_width=True, hide_index=True)
        st.download_button('Download reference ' + chosen, json.dumps(library[chosen], indent=2),
                           file_name='health_library_' + chosen + '.json', mime='application/json')
    stats = data['manifest']
    labels = ['Total input rows', 'Unique patterns', 'Conditions', 'Symptom features']
    values = [stats['raw_rows'], stats['unique_rows'], stats['condition_count'], stats['feature_count']]
    for col, label, value in zip(st.columns(4), labels, values):
        col.metric(label, value)
    st.info(
        f"{stats['duplicates_removed']:,} repeated patterns were removed before splitting. "
        'Repeated variants are not independent clinical cases.'
    )
    st.info(f"{len(data['reference_conditions'])} source-linked reference records provide educational summaries. "
            'They are not clinically reviewed prescribing guidance or patient observations.')
    hospital = data['pakistan_hospital_symptoms']
    st.subheader('Pakistani hospital symptom references')
    for col, label, value in zip(st.columns(3),
                                ['Symptom terms', 'Topic–symptom links', 'Hospital pages'],
                                [len({r['term'] for r in hospital}), len(hospital),
                                 len({r['source']['url'] for r in hospital})]):
        col.metric(label, value)
    st.caption('Published guidance from Aga Khan University Hospital, Shifa International Hospital and PKLI. '
               'These references improve wording coverage and answers; they are separate from patient cases and classifier training.')
    table = st.selectbox('JSON table', ['training', 'conditions', 'symptoms', 'manifest',
                                      'reference_conditions', 'symptom_metadata', 'pakistan_hospital_symptoms'])
    if table == 'manifest':
        st.json(stats)
    else:
        st.dataframe(pd.DataFrame(data[table]), use_container_width=True, hide_index=True)
        st.download_button(
            'Download ' + table + '.json',
            json.dumps(data[table], indent=2, ensure_ascii=False),
            file_name=table + '.json',
            mime='application/json',
        )
    counts = pd.Series([row['condition'] for row in data['training']]).value_counts().rename('Unique patterns')
    st.bar_chart(counts)
    st.caption('Medicine, diet, exercise and precaution records are unreviewed source material. Only condition classification is evaluated.')
    if stats.get('illustrative_rows'):
        st.info(f"{stats['legacy_unique_rows']:,} original educational patterns plus {stats['illustrative_rows']:,} labelled illustrative examples. Generated subsets are not hospital cases or clinical evidence. The 10 added condition labels have no independent evaluation.")
    st.subheader('Hospital research datasets')
    try:
        catalog, regional, audit = load_regional_preview()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Hospital dataset preview unavailable: {exc}')
        return
    for col, label, value in zip(st.columns(3),
                                 ['Distinct observations', 'Research datasets', 'Countries'],
                                 [regional['unique_rows'], len(catalog['sources']), len(audit['countries'])]):
        col.metric(label, value)
    st.info('These public clinical cohorts are collected for research. Their targets include infant referral, '
            'heart-failure survival, diabetes and kidney disease. They are not used by the symptom classifier, '
            f"which contains {len(data['training']):,} educational patterns including labelled illustrative examples.")
    summaries = {s['source_id']: s for s in regional['sources']}
    st.dataframe(pd.DataFrame([{
        'Dataset': source['name'], 'Country': source['country'],
        'Raw rows': summaries[source['id']]['raw_rows'],
        'Distinct observations': summaries[source['id']]['unique_rows'],
        'Duplicates removed': summaries[source['id']]['duplicates_removed'],
        'Task': source['task'],
    } for source in catalog['sources']]), use_container_width=True, hide_index=True)
    source_by_name = {source['name']: source for source in catalog['sources']}
    chosen_name = st.selectbox('Hospital dataset', list(source_by_name))
    chosen = source_by_name[chosen_name]
    st.markdown(f"[Published dataset]({chosen['source_url']}) · [Licence]({chosen['license']['url']})")
    st.caption(chosen['target_meaning'])
    st.write(chosen['attribution'])
    prepared = DATA_DIR / 'regional' / summaries[chosen['id']]['file']
    st.download_button('Download clinical observations', prepared.read_bytes(), file_name=prepared.name,
                       mime='application/x-ndjson')
    st.download_button('Download hospital source catalog', json.dumps(catalog, indent=2),
                       file_name='hospital_sources.json', mime='application/json')
    with st.expander('Clinical dataset limitations and audit'):
        st.write(chosen['population'])
        for note in chosen['limitations']:
            st.write(note)
        for note in audit['warnings']:
            st.caption(note)


def evaluation_page(version):
    st.title('Model Evaluation')
    st.write('Measured results from the saved training run. The final test set is separate from model tuning.')
    try:
        metrics = model_metrics(version)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(str(exc))
        return
    sizes = metrics['split_sizes']
    st.caption(f"Train: {sizes['train']} · Validation: {sizes['validation']} · Test: {sizes['test']} · Seed: {metrics['seed']}")
    rows = []
    for key, name in EVAL_NAMES.items():
        item = metrics['models'][key]
        rows.append({
            'Model': name,
            'Accuracy': item['accuracy'],
            'Macro F1': item['report']['macro avg']['f1-score'],
            'Precision': item['report']['macro avg']['precision'],
            'Recall': item['report']['macro avg']['recall'],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.warning('This is a technical benchmark on 61 template-like held-out records, not a clinical validation. '
               'Perfect scores do not establish real-world diagnosis, triage or medicine safety.')
    if metrics.get('unevaluated_labels'):
        st.warning('No independent test examples for: ' + ', '.join(metrics['unevaluated_labels']) + '.')
    chosen = st.selectbox('Inspect model', list(EVAL_NAMES), format_func=lambda k: EVAL_NAMES[k])
    item = metrics['models'][chosen]
    st.metric('Test accuracy', f"{item['accuracy']:.2%}")
    with st.expander('Precision, recall and F1', expanded=True):
        st.dataframe(pd.DataFrame(item['report']).transpose(), use_container_width=True)
    with st.expander('Confusion matrix — rows actual, columns predicted'):
        st.dataframe(pd.DataFrame(item['confusion_matrix'], index=metrics['labels'], columns=metrics['labels']))
    st.subheader('Fine-tuning history')
    st.line_chart(pd.DataFrame(metrics['training']['history']).set_index('epoch')[['train_loss', 'validation_loss']])
    st.caption(f"Best epoch: {metrics['training']['best_epoch']} · All encoder layers fine-tuned · Batch size: 16")
    with st.expander('Hyperparameters and optimization'):
        st.json(metrics['optimization'])
        st.write('Head regularization selected on fixed original validation examples. Training gives each condition and training family equal total weight. Transformer checkpoint selected by validation accuracy, then validation loss. Test data is not used for model selection.')
    research_path = ARTIFACT_DIR.parent / 'research/metrics.json'
    if research_path.exists():
        st.subheader('Published hospital data benchmarks')
        research = json.loads(research_path.read_text())
        st.caption('Separate research tasks using original study labels. These models are not used for chat responses.')
        st.dataframe(pd.DataFrame([{
            'Dataset': row['name'], 'Country': row['country'], 'Study task': row['task'],
            'Train / validation / test': ' / '.join(str(row['split_sizes'][key]) for key in ['train', 'validation', 'test']),
            'Balanced accuracy': row['test']['balanced_accuracy'], 'ROC AUC': row['test']['roc_auc'],
            'Majority baseline': row['majority_baseline']['balanced_accuracy'],
        } for row in research['models']]), use_container_width=True, hide_index=True)
        with st.expander('Hospital benchmark methods and limitations'):
            st.write(research['selection_policy'])
            for row in research['models']:
                st.markdown('**' + row['name'] + '**')
                for note in row['limitations']:
                    st.caption(note)
            st.caption('The infant-referral cohort is excluded until its coded features and timing are reviewed.')
        st.download_button('Download hospital benchmark results', research_path.read_bytes(),
                           file_name='hospital_benchmarks.json', mime='application/json')
    st.download_button(
        'Download evaluation JSON',
        (ARTIFACT_DIR / 'metrics.json').read_bytes(),
        file_name='evaluation.json',
        mime='application/json',
    )


def history_page():
    st.title('History')
    page_header()
    st.write('Your conversations are saved in this browser so you can return to them later. '
             'Anyone using this browser can see them. Download a copy if you want to keep one elsewhere.')
    if st.session_state.get('browser_history_error'):
        st.warning('This browser is not allowing conversation saving. You can still download a copy below.')
    st.button('Clear saved conversations', icon=':material/delete_outline:',
              on_click=clear_saved_conversations)
    chats = list(st.session_state.chats.values())
    st.download_button(
        'Export history as JSON',
        json.dumps(chats, indent=2),
        file_name='mrs_ai_bot_history.json',
        mime='application/json',
    )
    for chat in reversed(chats):
        if not chat['messages']:
            continue
        with st.expander(chat['title']):
            st.caption(chat['created_at'])
            for msg in chat['messages']:
                st.markdown('**' + msg['role'].title() + '**')
                st.write(msg['content'])
            st.button('Continue this chat', key='resume_' + chat['id'], on_click=choose_chat, args=(chat['id'],))


def about_page(data):
    page_header()
    st.title('About & sources')
    st.write('MRS AI offers health information and symptom guidance. It is not a diagnosis or treatment recommendation service.')
    with st.container(border=True):
        st.subheader('What to expect')
        st.write('Describe your symptoms, answer brief follow-up questions, and read educational information. '
                 f"The classifier covers {len(data['conditions'])} condition labels, but its matches are not clinically validated diagnoses.")
        st.write('Health context can withhold medicine information. The system cannot verify doses, interactions or personal suitability.')
    st.subheader('Where the information comes from')
    if data.get('health_library', {}).get('manifest'):
        st.markdown('Information from the NHS website is licensed under the '
                    '[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). '
                    'The reference excerpts were collected on ' + data['health_library']['manifest']['collected_on'] + '.')
        st.markdown('[Medicine label source and limitations](https://open.fda.gov/apis/drug/label/)')
    st.markdown('[Original symptom and medicine dataset](https://github.com/dr-mushtaq/Medicine-Recommendation-System)')
    st.caption('Medicine mappings in this dataset have not been clinically reviewed.')
    for record in data['reference_conditions']:
        for source in record['sources']:
            date = ('page reviewed ' + source['page_last_reviewed'] if source.get('page_last_reviewed')
                    else 'source updated ' + source['updated_on'] if source.get('updated_on')
                    else 'review date unavailable')
            st.markdown(f"[{record['name']} — {source['publisher']}]({source['url']}) · {date}")
    st.caption('NHS, FDA, Aga Khan University Hospital, Shifa International Hospital and PKLI summaries provide educational references. Selected symptom inventories also support clearly labelled illustrative training examples; they are not patient records.')
    st.subheader('Your conversation')
    st.write('Your conversations are saved in this browser so you can return to them. Anyone using this browser '
             'can see them, and you can clear them from Conversation history. They are not medical records. '
             + model_privacy_notice() + ' '
             'Please leave out names and other identifying details. You can download your conversation before '
             'leaving. We keep anonymous activity counts, but not your messages or health context in those counts.')
    with st.expander('Project resources'):
        st.markdown('[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)')
        st.button('Dataset Preview', on_click=navigate, args=('Dataset Preview',))
        st.button('Model Evaluation', on_click=navigate, args=('Model Evaluation',))


def main():
    st.set_page_config(page_title='MRS AI · Symptom guidance', page_icon=str(ASSETS / 'logo.svg'),
                       layout='wide', initial_sidebar_state='auto')
    st.markdown('<style>' + (ASSETS / 'style.css').read_text() + '</style>', unsafe_allow_html=True)
    if 'analytics_session' not in st.session_state:
        st.session_state.analytics_session = uuid4().hex
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
        create_chat()
    if not st.session_state.get('browser_history_loaded'):
        stored = read_history()
        if isinstance(stored, dict) and stored.get('status') == 'loaded':
            restored = valid_history(stored.get('data'))
            if restored and not any(chat['messages'] for chat in st.session_state.chats.values()):
                chats, active = restored
                if chats:
                    st.session_state.chats = chats
                    st.session_state.active_chat = active
                    st.session_state.chat_landing = True
            st.session_state.browser_history_loaded = True
        elif isinstance(stored, dict) and stored.get('status') == 'error':
            st.session_state.browser_history_loaded = True
            st.session_state.browser_history_error = True
    chat = st.session_state.chats[st.session_state.active_chat]
    sidebar(chat)
    try:
        version = fingerprint()
        supplemental = tuple((DATA_DIR / name).stat().st_mtime_ns for name in ['reference_conditions.json', 'symptom_metadata.json', 'pakistan_hospital_symptoms.json'])
        library_manifest = DATA_DIR / 'health_library/manifest.json'
        supplemental += (library_manifest.stat().st_mtime_ns if library_manifest.exists() else 0,)
        data = cached_data((version, supplemental))
    except (FileNotFoundError, ValueError):
        logging.exception('Unable to load knowledge base')
        st.error('The information service is currently unavailable. Please try again later.')
        return
    page = st.session_state.navigation
    view = (page, chat['id'])
    requested_landing = st.session_state.pop('chat_landing', False)
    landing = st.session_state.get('last_view') != view or requested_landing
    st.session_state.last_view = view
    if page == 'Chat':
        chat_page(chat, data, version, landing=landing)
    elif page == 'Health context':
        health_context_page(chat)
    elif page == 'Dataset Preview':
        dataset_page(data)
    elif page == 'Model Evaluation':
        evaluation_page(version)
    elif page == 'History':
        history_page()
    elif page == 'Analytics':
        analytics_page()
    else:
        about_page(data)
    if st.session_state.get('browser_history_loaded'):
        result = write_history(st.session_state.chats, st.session_state.active_chat,
                               clear=st.session_state.pop('clear_browser_history', False))
        if isinstance(result, dict) and result.get('status') == 'error':
            st.session_state.browser_history_error = True
