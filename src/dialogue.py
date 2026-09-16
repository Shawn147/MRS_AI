"""Conversation state around classifier inference and JSON retrieval.

Symptom matching uses explicit aliases, not a learned NER model.
"""
import re

from src.data import load_corrections

EMERGENCY_PHRASES = (
    'chest pain', 'trouble breathing', 'difficulty breathing', 'cannot breathe', "can't breathe",
    'shortness of breath', 'severe bleeding', 'one sided weakness', 'one-sided weakness',
    'fainted', 'seizure', 'suicidal', 'coughing blood', 'coughing up blood',
)
EMERGENCY_LABELS = {'Heart attack', 'Paralysis (brain hemorrhage)'}
NO_CONTEXT = {'', 'none', 'no', 'no known allergies', 'no allergies', 'n/a', 'not applicable'}
SKIP_WORDS = {'show results', 'results', 'continue', 'skip', 'show matches'}
YES = {'yes', 'yes i do', 'yes i have', 'yeah', 'yep'}
NO = {'no', 'no i do not', "no i don't", 'nope'}
MIN_PROBABILITY = 0.45
MIN_MARGIN = 0.10
INPUT_CORRECTIONS = load_corrections()


def correct_spelling(text):
    corrections = []

    def replace(match):
        word = match.group().lower()
        if word in INPUT_CORRECTIONS:
            corrected = INPUT_CORRECTIONS[word]
            corrections.append((word, corrected))
            return corrected
        return match.group()

    return re.sub(r'\b[a-zA-Z]+\b', replace, text), list(dict.fromkeys(corrections))


def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r"[^a-z0-9\s'-]", ' ', text.lower())).strip()


def negated(text, start):
    prefix = re.split(r'[.!?;,]|\bbut\b|\bhowever\b', text[:start].lower())[-1]
    words = prefix.split()[-6:]
    return any(w in {'no', 'not', 'without', 'deny', 'denies', 'denied', "don't", 'never'} for w in words)


def ulcer_mentions(text):
    pattern = r'\b(?:(?:stomach|peptic|gastric|mouth|oral|skin|leg|duodenal)\s+)?ulcers?\b'
    return [m for m in re.finditer(pattern, text, re.I) if not negated(text, m.start())]


def emergency(text):
    lower = text.lower()
    return any(
        not negated(lower, match.start())
        for phrase in EMERGENCY_PHRASES
        for match in re.finditer(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', lower)
    )


def extract_symptoms(text, symptoms):
    lower = text.lower().replace('_', ' ')
    matches = []
    for item in symptoms:
        for alias in item['aliases']:
            for match in re.finditer(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', lower):
                matches.append((match.start(), match.end(), item['id'], negated(lower, match.start())))
    matches.sort(key=lambda hit: -(hit[1] - hit[0]))
    accepted = []
    for hit in matches:
        if any(hit[0] >= old[0] and hit[1] <= old[1] for old in accepted):
            continue
        accepted.append(hit)
    positive, negative = set(), set()
    for _, _, key, is_negative in sorted(accepted):
        if is_negative:
            negative.add(key)
            positive.discard(key)
        else:
            positive.add(key)
            negative.discard(key)
    return positive, negative


def new_state():
    return {
        'symptoms': [], 'denied': [], 'pending': None, 'questions_asked': [],
        'context': {}, 'last_predictions': [], 'urgent': False,
    }


def profile_has_context(context):
    return any(normalize(str(value)) not in NO_CONTEXT for value in context.values())


def infer_profile(text):
    found = {}
    if re.search(r'\ballerg(?:ic|y|ies)\b', text, re.I):
        found['allergies_in_chat'] = text
    if re.search(r'\b(pregnant|pregnancy|kidney disease|liver disease|blood thinner|medical history)\b', text, re.I):
        found['history_in_chat'] = text
    if ulcer_mentions(text):
        found['ulcer_in_chat'] = text
    return found


def next_question(state, predictions, data):
    seen = state['symptoms'] + state['denied'] + state['questions_asked']
    candidates = []
    for rank, pred in enumerate(predictions):
        for position, symptom in enumerate(data['by_condition'][pred['condition']]['symptoms']):
            if symptom not in seen:
                candidates.append((rank * 20 + position, symptom))
    return min(candidates)[1] if candidates else None


def respond(text, state, data, predictor, profile=None):
    corrected, corrections = correct_spelling(text)
    answer = _respond(corrected, state, data, predictor, profile)
    if corrections and not answer.get('urgent'):
        note = '; '.join(f'“{original}” as “{replacement}”' for original, replacement in corrections)
        answer['text'] = f'I understood {note}.\n\n' + answer['text']
    return answer


def _reply(text, predictions=None, **extra):
    payload = {'text': text, 'predictions': predictions or []}
    payload.update(extra)
    return payload


def _respond(text, state, data, predictor, profile=None):
    if not text.strip():
        return _reply('Please describe a symptom to begin.')
    if len(text) > 2000:
        return _reply('Please keep each message under 2,000 characters.')
    state['context'].update(infer_profile(text))
    context = {**state['context'], **(profile or {})}
    if emergency(text):
        state['urgent'] = True
        state['pending'] = None
        state['last_predictions'] = []
        return _reply(
            'These symptoms may need urgent assessment. Contact local emergency services or seek urgent medical care. I will not suggest medicines for this conversation.',
            urgent=True,
        )
    if state['urgent']:
        return _reply(
            'This conversation contains an urgent symptom report. Please seek urgent medical assessment; medicine suggestions remain paused.',
            urgent=True,
        )
    norm = normalize(text)
    if norm in {'hi', 'hello', 'hey', 'good morning', 'salam', 'assalam o alaikum'}:
        return _reply('Hello! Tell me which symptoms you have. You can add details over several messages, and correct a symptom by saying, for example, “no cough”.')
    if norm in {'thanks', 'thank you', 'ok', 'okay'}:
        return _reply('You’re welcome. You can add another symptom or start a new chat for a different concern.')
    positive, negative = extract_symptoms(text, data['symptoms'])
    pending = state['pending']
    if pending and norm in YES:
        positive.add(pending)
    elif pending and norm in NO:
        negative.add(pending)
    if positive or negative:
        state['pending'] = None
    for symptom in sorted(negative):
        if symptom in state['symptoms']:
            state['symptoms'].remove(symptom)
        if symptom not in state['denied']:
            state['denied'].append(symptom)
    for symptom in sorted(positive):
        if symptom in state['denied']:
            state['denied'].remove(symptom)
        if symptom not in state['symptoms']:
            state['symptoms'].append(symptom)
    ulcers = ulcer_mentions(text)
    if any(match.group().lower() in {'ulcer', 'ulcers'} for match in ulcers):
        state['pending'] = None
        state['last_predictions'] = []
        added = ', '.join(data['by_symptom'][s]['label'] for s in sorted(positive))
        note = f'I’ve added **{added}** to this conversation. ' if added else ''
        return _reply(
            note + 'You also mentioned an **ulcer**. Is it a stomach ulcer, a mouth ulcer, or another type, and has it been diagnosed? I’ve recorded it as context without assuming its location. Medicine suggestions are paused while we clarify this.'
        )
    if not state['symptoms'] and infer_profile(text):
        return _reply('I’ve noted that medical context. What symptoms are you experiencing now?')
    if not state['symptoms']:
        state['last_predictions'] = []
        return _reply('I don’t have an active symptom to check yet. Try describing it directly, such as “cough”, “fever”, “itching”, or “stomach pain”.')
    if not (positive or negative):
        if infer_profile(text):
            return _reply('I’ve noted that context. Medicine suitability requires a clinician’s review. Add any remaining symptoms, or update your patient context in the sidebar.')
        if norm not in SKIP_WORDS:
            return _reply('I couldn’t identify a new symptom in that message. Your earlier symptoms are retained. Name another symptom, answer the pending question, or type “show results”.')
    predictions = predictor(state['symptoms'])
    state['last_predictions'] = predictions
    best = predictions[0]
    if best['condition'] in EMERGENCY_LABELS:
        state['urgent'] = True
        return _reply(
            'The model matched a condition that can require urgent assessment. This is not a diagnosis. Please seek urgent medical care; no medicine suggestions will be shown.',
            urgent=True,
        )
    missing = next_question(state, predictions, data)
    low = best['probability'] < MIN_PROBABILITY or best['probability'] - predictions[1]['probability'] < MIN_MARGIN
    labels = ', '.join(data['by_symptom'][s]['label'] for s in state['symptoms'])
    if len(state['symptoms']) < 2 and missing and norm not in SKIP_WORDS:
        state['pending'] = missing
        state['questions_asked'].append(missing)
        return _reply(
            f'I’ve noted **{labels}**. One symptom alone is not enough to narrow this down.\n\nDo you also have **{data["by_symptom"][missing]["label"]}**? You can answer yes/no, name another symptom, or type “show results”.'
        )
    if low:
        answer = f'I’ve considered **{labels}**. The model cannot distinguish the possible conditions confidently enough to show medicine information.'
        if missing and len(state['questions_asked']) < 3:
            state['pending'] = missing
            state['questions_asked'].append(missing)
            answer += f'\n\nDo you also have **{data["by_symptom"][missing]["label"]}**?'
        else:
            answer += ' Please seek a qualified clinician’s assessment.'
        return _reply(answer, predictions, uncertain=True)
    record = data['by_condition'][best['condition']]
    answer = (
        f'Based on **{labels}**, the model’s top possible condition is **{best["condition"]}**.\n\n'
        f'Model score: **{best["probability"]:.1%}** — an uncalibrated class probability, not diagnostic certainty.\n\n'
        f'{record["description"]}\n\n'
    )
    if profile_has_context(context):
        answer += '**Patient context noted.** Medicine names are withheld because the source data cannot establish suitability for your history, allergies, current medicines, age, or preferences. Please discuss these details with a clinician.\n\n'
    else:
        meds = '\n'.join('- ' + name for name in record['medications'][:5])
        answer += (
            '**Educational medicine information from the supplied dataset**\n\n'
            f'{meds}\n\n'
            'These entries are in source order; they are not ranked by safety or effectiveness. The dataset does not provide verified dosage, interaction, or allergy rules.\n\n'
        )
    answer += 'This is an educational prototype, not a diagnosis or prescription. A qualified healthcare professional should confirm the condition and treatment.'
    return _reply(answer, predictions, condition=best['condition'])
