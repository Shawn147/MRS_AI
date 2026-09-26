"""Local, optional general-question answers using Ollama and reference search."""
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.reference_search import search_references

OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'
MODEL = 'qwen3:4b'
QUESTION_START = re.compile(
    r'^(?:what|why|how|when|where|who|which|can|could|does|do|is|are|tell me|explain)\b', re.I
)
PERSONAL_SYMPTOMS = re.compile(
    r'\b(?:i have|i feel|i am feeling|i\x27m feeling|i\x27ve had|my symptoms|my headache|my pain)\b', re.I
)
UNRECOGNIZED_HEALTH_REPORT = re.compile(
    r'\b(?:i feel|i felt|i am feeling|i\x27m feeling|i\x27ve been feeling|'
    r'(?:i am|i\x27m) (?:sleepy|dizzy|tired|weak|sick|unwell|nauseous)|'
    r'i get (?:sleepy|dizzy|tired|weak|sick)|my (?:body|head|stomach|skin) feels)\b', re.I
)
CONTEXTUAL_REPORT = re.compile(r'\b(?:after|during|when|while|following)\b', re.I)


def is_information_question(text):
    """Keep first-person symptom reports in the existing symptom flow."""
    message = text.strip()
    return bool(QUESTION_START.match(message) or message.endswith('?')) and not PERSONAL_SYMPTOMS.search(message)


def is_unrecognized_health_report(text):
    """Find personal health descriptions that the symptom aliases did not capture."""
    return bool(UNRECOGNIZED_HEALTH_REPORT.search(text) or
                (text.strip().endswith('?') and PERSONAL_SYMPTOMS.search(text)))


def is_contextual_symptom_report(text):
    """Recognize circumstances that a symptom-only classifier would discard."""
    return bool(CONTEXTUAL_REPORT.search(text))


def _matched_references(question, records):
    """Require an explicit reference topic before claiming a source supports an answer."""
    aliases = {'Influenza (flu)': ('flu', 'influenza'),
               'COVID-19': ('covid', 'coronavirus'),
               'Sinusitis': ('sinusitis', 'sinus infection')}
    selected = [record for record in records if any(
        re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', question, re.I)
        for alias in aliases.get(record['name'], (record['name'],))
    )]
    return selected


def answer_general_question(question, state, data):
    records = _matched_references(question, data['reference_conditions'])
    passages = search_references(question, records, limit=3) if records else []
    source_text = '\n'.join(f'- {item["text"]}' for item in passages)
    symptoms = ', '.join(data['by_symptom'][key]['label'] for key in state['symptoms']) or 'none reported'
    system = (
        'You are MRS AI, an educational health information assistant. Answer the user\'s actual question '
        'clearly and briefly. If the user describes a feeling or symptom, acknowledge it and ask one useful '
        'clarifying question about timing, severity, or other symptoms. '
        'Do not diagnose the user, infer a disease from symptoms, recommend a medicine, '
        'give doses, or claim to have checked personal safety. If urgent symptoms are described, advise urgent '
        'medical assessment. The reference notes below are the only verified source material supplied to you. '
        'Use them when relevant and do not claim they cover other conditions. Reply directly with the final answer; '
        'do not show reasoning, planning, or instructions. If there are no reference notes, '
        'give cautious general information and say that this app has no reviewed source for the topic. '
        'Do not invent citations or imply that a condition has been clinically confirmed. '
        f'Previously reported symptoms (context only): {symptoms}.\n'
        f'Reference notes:\n{source_text or "None for this topic."}'
    )
    payload = {
        'model': MODEL,
        'stream': False,
        'think': False,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': question + '\n/no_think'}],
        'options': {'temperature': 0.2, 'num_predict': 1200},
    }
    request = Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError):
        return {'text': 'I can’t reach the local Qwen model right now. Start Ollama and confirm qwen3:4b is installed.',
                'predictions': [], 'intent': 'general_question_unavailable'}
    answer = result.get('message', {}).get('content', '').strip()
    if '</think>' in answer:
        answer = answer.rsplit('</think>', 1)[1].strip()
    elif answer.startswith('<think>') or answer.startswith('Okay, the user'):
        answer = ''
    if not answer:
        return {'text': 'The local model did not return an answer. Please try again.',
                'predictions': [], 'intent': 'general_question_unavailable'}
    sources = list(dict.fromkeys(item['source'] for item in passages))
    return {'text': answer, 'predictions': [], 'intent': 'general_question', 'sources': sources}
