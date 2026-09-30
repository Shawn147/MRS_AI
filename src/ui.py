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
from src.dialogue import new_state, respond, profile_has_context

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


def choose_chat(key):
    st.session_state.active_chat = key
    st.session_state.navigation = 'Chat'


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
    chat['editing_last'] = True


def cancel_edit(chat):
    chat['editing_last'] = False


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
        st.markdown('<div class="sidebar-note">For educational information.<br>Not a diagnosis or prescription.'
                    '<br><br>Your conversation stays here while this session is open. Some questions may be shared '
                    'with Groq to provide an answer. Please leave out names and other identifying details.</div>',
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
    st.caption('Privacy: some questions and relevant symptom details may be shared with Groq to provide an answer. '
               'Please leave out names and other identifying details. Your conversation stays here while this '
               'session is open. We keep anonymous activity counts.')


def welcome_details(chat):
    st.markdown('<div class="how-it-works"><h2>A conversation, one step at a time</h2><div class="steps">'
                '<div><span class="step-number">1</span><strong>Tell me what’s happening</strong>'
                '<p>Start with your symptoms, using your own words.</p></div>'
                '<div><span class="step-number">2</span><strong>Explore the details</strong>'
                '<p>Answer a few follow-up questions to clarify what you’re feeling.</p></div>'
                '<div><span class="step-number">3</span><strong>Understand the possibilities</strong>'
                '<p>Review possible matches and educational information.</p></div></div></div>', unsafe_allow_html=True)
    configured = profile_has_context(chat['profile'])
    st.markdown('<div class="context-note"><strong>' + ('Your health context is saved' if configured else 'Your health context matters') +
                '</strong>Allergies, current medicines or existing conditions can affect which information is shown. '
                'You can add these in Health context.</div>', unsafe_allow_html=True)


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


def render_message(message, chat, data, editable=False, latest=False):
    is_user = message['role'] == 'user'
    with st.chat_message(message['role'], avatar=':material/person:' if is_user else str(ASSETS / 'logo.svg')):
        author = 'You' if is_user else 'MRS AI'
        time = datetime.fromisoformat(message['timestamp']).astimezone().strftime('%H:%M') if message.get('timestamp') else ''
        role_class = ' user-author' if is_user else ''
        st.markdown(f'<div class="message-author{role_class}">{author}<span>{time}</span></div>', unsafe_allow_html=True)
        if is_user:
            st.markdown(message['content'])
            if editable:
                st.button('Edit last prompt', icon=':material/edit:', key='edit_last_' + chat['id'],
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
            st.caption('A possible match from your symptoms, not a diagnosis. A healthcare professional can assess the cause.')
            st.caption('Source: original symptom and medicine dataset · clinical review date unavailable.')
            context = {**chat['state']['context'], **chat['profile']}
            withheld = message.get('medicine_withheld') or profile_has_context(context) or chat['state']['urgent']
            if withheld:
                st.info('Medicine information is withheld. Your health context or an urgent symptom report means '
                        'these details need a clinician’s review.', icon=':material/info:')
            else:
                with st.expander('Medicine information · educational reference', icon=':material/medication:'):
                    st.caption('Unreviewed entries from the source dataset. Suitability and effectiveness have not been established.')
                    for medicine in record['medications'][:5]:
                        st.write('• ' + medicine)
                    st.caption('This source does not provide verified doses, allergy checks or interaction rules. '
                               'Discuss treatment with a qualified healthcare professional.')
            others = [p['condition'] for p in message.get('predictions', []) if p['condition'] != record['name']]
            if others:
                with st.expander('Other possible matches'):
                    st.write(' · '.join(others))
                    st.caption('These are other possibilities to discuss with a healthcare professional, not confirmed conditions.')
        elif message.get('uncertain'):
            st.info('**Let’s understand this a little better**\n\n' + message['content'], icon=':material/info:')
        else:
            st.markdown(message['content'])
            render_sources(message)
        if latest and message.get('retryable'):
            st.button('Retry answer', icon=':material/refresh:', key='retry_' + chat['id'],
                      on_click=queue_retry)


def chat_page(chat, data, version):
    page_header()
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
    if not chat['messages']:
        welcome(chat)
    else:
        st.markdown('<div class="conversation-heading"><h1>Let’s understand how you’re feeling</h1>'
                    '<p>You can add details or correct a symptom at any time.</p></div>', unsafe_allow_html=True)
    for index, message in enumerate(chat['messages']):
        editable = (index == len(chat['messages']) - 2 and message['role'] == 'user')
        render_message(message, chat, data, editable=editable,
                       latest=index == len(chat['messages']) - 1)
    edited_prompt = None
    if chat.get('editing_last'):
        with st.form('edit_form_' + chat['id']):
            revised = st.text_area('Edit your last message', value=chat['messages'][-2]['content'], max_chars=2000)
            save_edit = st.form_submit_button('Save and regenerate', type='primary')
        st.button('Cancel edit', on_click=cancel_edit, args=(chat,))
        if save_edit:
            if not revised.strip():
                st.warning('Enter a message before saving.')
            else:
                from src.chat_edit import replace_last_turn
                old_event_id = replace_last_turn(chat, data, predictor)
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
    elif not chat.get('editing_last') and chat['messages'] and not chat['state']['urgent']:
        with st.container(key='followup'):
            for col, label in zip(st.columns(2), ['Summarize my symptoms', 'Explain the previous result']):
                col.button(label, on_click=queue_prompt, args=(label,), use_container_width=True)
    if chat['state']['urgent']:
        st.warning('Medicine suggestions are paused for this conversation. Contact local emergency services or seek urgent care.')
    st.markdown('<div class="composer-note">For educational information. Not a diagnosis or prescription.</div>', unsafe_allow_html=True)
    if chat.get('editing_last'):
        entered = None
    elif chat['messages']:
        entered = st.chat_input('Describe what you’re experiencing…', max_chars=2000)
    else:
        # Inline on the welcome page avoids Streamlit scrolling past the greeting
        # to its pinned composer before the first message has been sent.
        with st.container(key='welcome_composer'):
            entered = st.chat_input('Describe what you’re experiencing…', max_chars=2000)
        welcome_details(chat)
    prompt = retry_prompt or edited_prompt or st.session_state.pop('queued_prompt', None) or entered
    if prompt:
        chat['messages'].append({'role': 'user', 'content': prompt,
                                 'state_before': deepcopy(chat['state']),
                                 'timestamp': datetime.now(timezone.utc).isoformat()})
        if chat['title'] == 'New conversation':
            chat['title'] = prompt[:64]
        try:
            with st.spinner('Reviewing your symptoms…'):
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
        chat['messages'].append(message)
        if 'text' in message:
            from src.analytics import record_event
            message['id'] = uuid4().hex
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
    st.write('Supervisor CSV bundle, converted to JSON with source-row references.')
    stats = data['manifest']
    labels = ['Source rows', 'Unique patterns', 'Conditions', 'Symptom features']
    values = [stats['raw_rows'], stats['unique_rows'], stats['condition_count'], stats['feature_count']]
    for col, label, value in zip(st.columns(4), labels, values):
        col.metric(label, value)
    st.info(
        f"{stats['duplicates_removed']:,} repeated patterns were removed before splitting. "
        'Repeated variants are not independent clinical cases.'
    )
    st.info(f"{len(data['reference_conditions'])} additional source-linked conditions are reference-only; "
            'they are not classifier predictions or clinically reviewed prescribing guidance.')
    table = st.selectbox('JSON table', ['training', 'conditions', 'symptoms', 'manifest',
                                      'reference_conditions', 'symptom_metadata'])
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
        st.write('Head regularization selected with 2-fold cross-validation on training data. Transformer checkpoint selected by validation accuracy, then validation loss. Test data is not used for model selection.')
    st.download_button(
        'Download evaluation JSON',
        (ARTIFACT_DIR / 'metrics.json').read_bytes(),
        file_name='evaluation.json',
        mime='application/json',
    )


def history_page():
    st.title('History')
    page_header()
    st.write('Your conversations from this browser session. Download a copy to keep for yourself.')
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
                 'The classifier covers 41 condition labels, but its matches are not clinically validated diagnoses.')
        st.write('Health context can withhold medicine information. The system cannot verify doses, interactions or personal suitability.')
    st.subheader('Where the information comes from')
    st.markdown('[Original symptom and medicine dataset](https://github.com/dr-mushtaq/Medicine-Recommendation-System)')
    st.caption('Medicine mappings in this dataset have not been clinically reviewed.')
    for record in data['reference_conditions']:
        for source in record['sources']:
            date = ('page reviewed ' + source['page_last_reviewed'] if source.get('page_last_reviewed')
                    else 'source updated ' + source['updated_on'] if source.get('updated_on')
                    else 'review date unavailable')
            st.markdown(f"[{record['name']} — {source['publisher']}]({source['url']}) · {date}")
    st.caption('These additional NHS and FDA records are reference-only. They are not included in classifier predictions.')
    st.subheader('Your conversation')
    st.write('Your conversation and health context stay here while this session is open; they are not saved as a '
             'medical record. Some questions and relevant symptoms may be shared with Groq to provide an answer. '
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
    chat = st.session_state.chats[st.session_state.active_chat]
    sidebar(chat)
    try:
        version = fingerprint()
        supplemental = tuple((DATA_DIR / name).stat().st_mtime_ns for name in ['reference_conditions.json', 'symptom_metadata.json'])
        data = cached_data((version, supplemental))
    except (FileNotFoundError, ValueError):
        logging.exception('Unable to load knowledge base')
        st.error('The information service is currently unavailable. Please try again later.')
        return
    page = st.session_state.navigation
    if page == 'Chat':
        chat_page(chat, data, version)
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
