"""Streamlit interface for MRS AI BOT."""
import json
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.data import ARTIFACT_DIR, DATA_DIR, fingerprint, load_data
from src.dialogue import new_state, respond

MODEL_NAMES = {
    'transformer': 'MiniLM · fine-tuned transformer',
    'ml': 'TF-IDF · Logistic Regression',
}
EVAL_NAMES = {
    'ml': 'TF-IDF + Logistic Regression',
    'frozen': 'Frozen MiniLM + trained head',
    'transformer': 'Fine-tuned MiniLM transformer',
}
PAGES = ['Chat', 'Dataset Preview', 'Model Evaluation', 'History', 'About']


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


def sidebar(chat):
    with st.sidebar:
        st.title('MRS AI BOT')
        st.caption('Medicine Recommendation System')
        st.button('＋ New chat', on_click=create_chat, use_container_width=True)
        st.subheader('Workspace')
        st.radio('Workspace', PAGES, key='navigation', label_visibility='collapsed')
        st.subheader('Recent chats')
        for key, item in list(st.session_state.chats.items())[::-1][:8]:
            st.button(item['title'][:44], key='open_' + key, on_click=choose_chat, args=(key,), use_container_width=True)
        with st.expander('Model & patient context'):
            model = st.selectbox(
                'Prediction model',
                list(MODEL_NAMES),
                format_func=lambda k: MODEL_NAMES[k],
                index=list(MODEL_NAMES).index(chat['model']),
                key='model_' + chat['id'],
            )
            if model != chat['model']:
                chat['model'] = model
            st.caption('Optional context stays in this browser session. No interaction or allergy database is available.')
            with st.form('context_' + chat['id']):
                profile = {}
                fields = [
                    ('allergies', 'Medicine allergies'),
                    ('history', 'Medical history / current conditions'),
                    ('medicines', 'Current medicines'),
                    ('preferences', 'Age / preferences'),
                ]
                for name, label in fields:
                    profile[name] = st.text_input(label, value=chat['profile'].get(name, ''))
                if st.form_submit_button('Save context'):
                    chat['profile'] = profile
                    st.success('Context saved for future replies.')
        st.caption('VU–BC190411036 · Educational prototype')


def render_message(message):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('predictions'):
            with st.expander('Other model matches'):
                frame = pd.DataFrame(message['predictions']).rename(
                    columns={'condition': 'Condition', 'probability': 'Model score'}
                )
                st.dataframe(
                    frame,
                    hide_index=True,
                    use_container_width=True,
                    column_config={'Model score': st.column_config.NumberColumn(format='%.3f')},
                )
        if message.get('model'):
            st.caption(MODEL_NAMES[message['model']])


def chat_page(chat, data, version):
    st.title('MRS AI BOT')
    st.caption('Medicine Recommendation System')
    if not chat['messages']:
        st.subheader('Describe your symptoms')
        st.write('Enter your symptoms below. You can add details and answer follow-up questions in the conversation.')
        examples = [
            'I have a runny nose, sneezing and cough',
            'I have itching and a skin rash',
            'I have a headache and nausea',
        ]
        for col, example in zip(st.columns(3), examples):
            if col.button(example, use_container_width=True):
                st.session_state.queued_prompt = example
        st.caption('Educational use only. This assistant cannot diagnose, prescribe, or replace a clinician.')
    for message in chat['messages']:
        render_message(message)
    entered = st.chat_input('Reply with symptoms or details…', max_chars=2000)
    prompt = st.session_state.pop('queued_prompt', None) or entered
    if prompt:
        chat['messages'].append({'role': 'user', 'content': prompt})
        if chat['title'] == 'New conversation':
            chat['title'] = prompt[:64]
        render_message(chat['messages'][-1])
        try:
            def predictor(symptoms):
                from src.models import predict
                model_metrics(version)
                bundle = cached_models(version, (ARTIFACT_DIR / 'metrics.json').stat().st_mtime_ns)
                return predict(bundle, symptoms, chat['model'])

            with st.spinner('Considering your symptoms…'):
                answer = respond(prompt, chat['state'], data, predictor, chat['profile'])
            message = {
                'role': 'assistant',
                'content': answer['text'],
                'predictions': answer.get('predictions', []),
                'model': chat['model'],
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
            message = {
                'role': 'assistant',
                'content': f'The local model is not ready: {exc}\n\nRun the setup steps in README.md, then try again.',
            }
        chat['messages'].append(message)
        st.rerun()
    st.caption('For medical concerns, consult a qualified healthcare professional. Chat history is session-only.')


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
    table = st.selectbox('JSON table', ['training', 'conditions', 'symptoms', 'manifest'])
    if table == 'manifest':
        st.json(stats)
    else:
        st.dataframe(pd.DataFrame(data[table]), use_container_width=True, hide_index=True)
        st.download_button(
            'Download ' + table + '.json',
            (DATA_DIR / (table + '.json')).read_bytes(),
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
    st.warning('These template-like symptom records are not clinical validation. High scores do not establish real-world diagnosis or medicine safety.')
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
    st.write('Conversations from this browser session. Export a copy if you need it for your demonstration.')
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


def about_page():
    st.title('About MRS AI BOT')
    st.write('A transformer-based conversational Medicine Recommendation System. The interface uses native Streamlit components.')
    st.markdown('**How it works:** message → explicit symptom aliases and conversation state → fine-tuned MiniLM → condition probabilities → JSON reference information.')
    st.write('MiniLM is a six-layer pretrained transformer. Its encoder and classification head are fine-tuned on the project training split. Replies and follow-up questions are controlled Python templates, not a generative LLM. The app runs locally after setup, without an API key.')
    st.write('The comparison includes a TF-IDF Logistic Regression baseline and a frozen transformer with a trained head. No hard-coded condition prediction or borrowed benchmark score is used.')
    st.markdown('[Supervisor repository](https://github.com/dr-mushtaq/Medicine-Recommendation-System) · [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)')
    st.subheader('Scope and limitations')
    st.write('41 dataset conditions; finite aliases; basic negation handling; no clinical validation; uncalibrated probabilities. Follow-up questions are selected from dataset symptoms. The source does not contain verified contraindications or medicine effectiveness rankings, so no such ranking is claimed. Patient context conservatively suppresses medicine names rather than claiming personalization the data cannot support.')
    st.write('Original sources and their license are in references/supervisor.')


def main():
    st.set_page_config(page_title='MRS AI BOT')
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
        create_chat()
    chat = st.session_state.chats[st.session_state.active_chat]
    sidebar(chat)
    try:
        version = fingerprint()
        data = cached_data(version)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f'JSON data is not ready: {exc}')
        st.code('python scripts/prepare_json.py')
        return
    page = st.session_state.navigation
    if page == 'Chat':
        chat_page(chat, data, version)
    elif page == 'Dataset Preview':
        dataset_page(data)
    elif page == 'Model Evaluation':
        evaluation_page(version)
    elif page == 'History':
        history_page()
    else:
        about_page()
