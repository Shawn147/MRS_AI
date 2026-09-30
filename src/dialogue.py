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
EMERGENCY_SYMPTOMS = {'chest_pain', 'breathlessness', 'weakness_of_one_body_side'}
SEX_HEADACHE = re.compile(
    r'(?=.*\b(?:headache|head pain|pain in (?:my |the )?head)\b)'
    r'(?=.*\b(?:sex|sexual activity|intercourse|orgasm)\b)', re.I,
)
SUDDEN_SEVERE = re.compile(
    r'\b(?:thunderclap|worst headache|sudden(?:ly)?(?:\s+\w+){0,4}\s+(?:severe|intense|explosive)|'
    r'(?:severe|intense|explosive)(?:\s+\w+){0,4}\s+sudden(?:ly)?)\b', re.I,
)
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
        'symptoms': [], 'denied': [], 'pending': None, 'pending_detail': None,
        'questions_asked': [],
        'context': {}, 'last_predictions': [], 'urgent': False,
        'details': {}, 'last_condition': None,
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


def respond(text, state, data, predictor, profile=None, general_answer=None):
    corrected, corrections = correct_spelling(text)
    answer = _respond(corrected, state, data, predictor, profile, general_answer)
    if corrections and not answer.get('urgent'):
        note = '; '.join(f'“{original}” as “{replacement}”' for original, replacement in corrections)
        answer['text'] = f'I understood {note}.\n\n' + answer['text']
        answer['correction_note'] = f'I understood {note}.'
    return answer


def _reply(text, predictions=None, **extra):
    payload = {'text': text, 'predictions': predictions or []}
    payload.update(extra)
    return payload


def natural_list(items):
    """Join short symptom labels the way they are spoken in a conversation."""
    if len(items) < 2:
        return ''.join(items)
    if len(items) == 2:
        return ' and '.join(items)
    return ', '.join(items[:-1]) + ', and ' + items[-1]


def _respond(text, state, data, predictor, profile=None, general_answer=None):
    if not text.strip():
        return _reply('Please describe a symptom to begin.')
    if len(text) > 2000:
        return _reply('Please keep each message under 2,000 characters.')
    state['context'].update(infer_profile(text))
    context = {**state['context'], **(profile or {})}
    if emergency(text):
        state['urgent'] = True
        state['pending'] = None
        state['pending_detail'] = None
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
    headache_mention = re.search(r'\b(?:headache|head pain|pain in (?:my |the )?head)\b', text, re.I)
    if SEX_HEADACHE.search(text) and headache_mention and not negated(text, headache_mention.start()):
        if 'headache' not in state['symptoms']:
            state['symptoms'].append('headache')
        state['pending'] = None
        state['last_condition'] = None
        state['last_predictions'] = []
        if SUDDEN_SEVERE.search(text):
            state['urgent'] = True
            return _reply(
                'A sudden, very severe headache during or after sex needs emergency assessment now. '
                'Contact local emergency services or go to an emergency department. '
                'I cannot determine the cause here and will not suggest medicines.',
                urgent=True, intent='sex_headache_urgent',
                sources=['https://www.nhs.uk/conditions/subarachnoid-haemorrhage/'],
            )
        return _reply(
            'You mentioned a headache during or after sex. If it came on suddenly and is very severe, '
            'seek emergency care now. If this is your first headache of this type, contact a clinician promptly. '
            'Did it start suddenly and become very severe?',
            intent='sex_headache',
            sources=['https://www.mayoclinic.org/diseases-conditions/sex-headaches/symptoms-causes/syc-20377477',
                     'https://www.nhs.uk/conditions/subarachnoid-haemorrhage/'],
        )
    from src.conversation import remember_details, context_reply
    remember_details(text, state)
    norm = normalize(text)
    detail_answered = False
    if state.get('pending_detail') == 'duration':
        if (re.search(r'\b(?:\d+|one|two|three|four|five|few|several|a)\s+'
                      r'(?:hours?|days?|weeks?|months?|years?)\b|'
                      r'\b(?:today|yesterday|this morning|this week|last night|suddenly|gradually)\b', norm)
                or norm in {'not sure', 'unsure', "don't know", 'i do not know', 'a while'}):
            state['details']['duration'] = text.strip()[:100]
            state['pending_detail'] = None
            detail_answered = True
    if norm in {'hi', 'hello', 'hey', 'hola', 'holla', 'good morning', 'salam', 'assalam o alaikum'}:
        return _reply('Hello! Tell me which symptoms you have. You can add details over several messages, and correct a symptom by saying, for example, “no cough”.')
    if norm in {'thanks', 'thank you', 'ok', 'okay'}:
        return _reply('You’re welcome. You can add another symptom or start a new chat for a different concern.')
    from src.general_qa import (answer_general_question, is_information_question,
                                is_unrecognized_health_report, is_contextual_symptom_report)
    answer_general = general_answer or answer_general_question
    # Informational questions must not be entered as symptoms or consume a pending answer.
    # Keep explicit medicine and dose requests in the existing guarded reply path.
    medicine_question = re.search(r'\b(dose|dosage|dosing|tablet|pill|medicine|medication|milligrams|mg|paracetamol|ibuprofen)\w*\b', norm)
    if is_information_question(text) and not medicine_question:
        existing = context_reply(text, state, data, profile_has_context(context))
        if existing and existing.get('intent') in {'summary', 'explain_match'}:
            return existing
        return answer_general(text, state, data)
    positive, negative = extract_symptoms(text, data['symptoms'])
    pending = state['pending']
    if pending and (norm in YES or re.match(r'^(yes|yeah|yep)\b', norm)):
        positive.add(pending)
    elif pending and (norm in NO or norm in {'not really', 'not at all', 'i do not', "i don't"}):
        negative.add(pending)
    if positive or negative:
        state['pending'] = None
        state['last_condition'] = None
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
    if positive & EMERGENCY_SYMPTOMS:
        state['urgent'] = True
        state['pending'] = None
        state['pending_detail'] = None
        state['last_predictions'] = []
        return _reply('These symptoms may need urgent assessment. Contact local emergency services '
                      'or seek urgent medical care. I will not suggest medicines for this conversation.',
                      urgent=True)
    ulcers = ulcer_mentions(text)
    if any(match.group().lower() in {'ulcer', 'ulcers'} for match in ulcers):
        state['pending'] = None
        state['last_predictions'] = []
        added = ', '.join(data['by_symptom'][s]['label'] for s in sorted(positive))
        note = f'I’ve added **{added}** to this conversation. ' if added else ''
        return _reply(
            note + 'You also mentioned an **ulcer**. Is it a stomach ulcer, a mouth ulcer, or another type, and has it been diagnosed? I’ve recorded it as context without assuming its location. Medicine suggestions are paused while we clarify this.'
        )
    if positive and not medicine_question and is_contextual_symptom_report(text):
        state['last_predictions'] = []
        return answer_general(text, state, data)
    if not state['symptoms'] and infer_profile(text):
        return _reply('I’ve noted that medical context. What symptoms are you experiencing now?')
    medication_request = re.search(r'\b(dose|dosage|dosing|tablet|pill|medicine|medication|milligrams|mg|paracetamol|ibuprofen)\w*\b', norm)
    if (not (positive or negative) or medication_request) and not detail_answered and not infer_profile(text) and norm not in SKIP_WORDS | YES | NO:
        followup = context_reply(text, state, data, profile_has_context(context))
        if followup:
            return followup
    if (not (positive or negative) and not medication_request and not infer_profile(text)
            and norm not in SKIP_WORDS | YES | NO and is_unrecognized_health_report(text)):
        return answer_general(text, state, data)
    if not state['symptoms']:
        state['last_predictions'] = []
        return _reply('I don’t have an active symptom to check yet. Try describing it directly, such as “cough”, “fever”, “itching”, or “stomach pain”.')
    if not (positive or negative) and not detail_answered:
        if infer_profile(text):
            return _reply('I’ve noted that context. Medicine suitability requires a clinician’s review. Add any remaining symptoms, or update your Health context.')
        if norm not in SKIP_WORDS:
            return _reply('I couldn’t identify a new symptom in that message. Your earlier symptoms are retained. Name another symptom, answer the pending question, or type “show results”.')
    predictions = predictor(state['symptoms'])
    state['last_condition'] = None
    state['last_predictions'] = predictions
    best = predictions[0]
    if best['condition'] in EMERGENCY_LABELS:
        state['urgent'] = True
        return _reply(
            'Your symptoms could indicate a condition that needs urgent assessment. Please seek urgent medical care. This is not a diagnosis, and I will not show medicine information for this conversation.',
            urgent=True,
        )
    missing = next_question(state, predictions, data)
    low = best['probability'] < MIN_PROBABILITY or best['probability'] - predictions[1]['probability'] < MIN_MARGIN
    labels = natural_list([data['by_symptom'][s]['label'] for s in state['symptoms']])
    if (len(state['symptoms']) == 1 and missing and norm not in SKIP_WORDS
            and len(state['questions_asked']) < 2):
        state['pending'] = missing
        state['questions_asked'].append(missing)
        return _reply(
            f'You mentioned **{labels}**. Several conditions can share this sign. '
            f'To narrow the possibilities, do you also have **{data["by_symptom"][missing]["label"]}**?\n\n'
            'You can answer yes or no, or tell me about another symptom.',
            uncertain=True,
        )
    if not state['details'].get('duration') and norm not in SKIP_WORDS:
        state['pending'] = None
        state['pending_detail'] = 'duration'
        return _reply(
            f'You mentioned **{labels}**. These can have several causes, and I need a little context '
            'before discussing a possible condition. **When did they start, and how long have you had them?** '
            'You can answer briefly, such as “two days” or “since this morning”.',
            uncertain=True,
        )
    if len(state['symptoms']) < 3 and missing and norm not in SKIP_WORDS and len(state['questions_asked']) < 2:
        state['pending'] = missing
        state['questions_asked'].append(missing)
        return _reply(
            f'You mentioned **{labels}**. Several conditions can share these signs. '
            f'To narrow the possibilities, do you also have **{data["by_symptom"][missing]["label"]}**?\n\n'
            'You can answer yes or no, or tell me about another symptom.',
            uncertain=True,
        )
    if len(state['symptoms']) < 3:
        state['pending'] = None
        return _reply(
            f'You mentioned **{labels}**. I still do not have enough information to name a reliable '
            'condition match or show medicine information. If symptoms persist or worry you, '
            'a healthcare professional can assess them.',
            uncertain=True,
        )
    if low:
        answer = (f'You mentioned **{labels}**. These symptoms can happen for different reasons, '
                  'so I need a little more detail before discussing possible conditions or medicine information.')
        if state.get('details', {}).get('duration'):
            answer += ' I’ve also noted how long this has been going on.'
        if missing and len(state['questions_asked']) < 3:
            state['pending'] = missing
            state['questions_asked'].append(missing)
            answer += f'\n\nHave you also had **{data["by_symptom"][missing]["label"]}**?'
        else:
            answer += ' Please add any other symptoms or speak with a healthcare professional if you remain concerned.'
        return _reply(answer, predictions, uncertain=True)
    record = data['by_condition'][best['condition']]
    state['last_condition'] = best['condition']
    answer = (
        f'Based on **{labels}**, one possible condition is **{best["condition"]}**. This is not a diagnosis.\n\n'
        f'{record["description"]}\n\n'
    )
    withheld = profile_has_context(context)
    if withheld:
        answer += '**Patient context noted.** Medicine names are withheld because the source data cannot establish suitability for your history, allergies, current medicines, age, or preferences. Please discuss these details with a clinician.\n\n'
    else:
        meds = '\n'.join('- ' + name for name in record['medications'][:5])
        answer += (
            '**Educational medicine information from the supplied dataset**\n\n'
            f'{meds}\n\n'
            'These entries are in source order; they are not ranked by safety or effectiveness. The dataset does not provide verified dosage, interaction, or allergy rules.\n\n'
        )
    answer += 'This is educational information, not a diagnosis or prescription. A qualified healthcare professional should confirm the condition and treatment.'
    return _reply(answer, predictions, condition=best['condition'], medicine_withheld=withheld)
