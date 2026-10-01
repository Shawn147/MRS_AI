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
EMERGENCY_LABELS = {'Heart attack', 'Paralysis (brain hemorrhage)', 'Appendicitis'}
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
SKIP_WORDS = {'show result', 'show results', 'results', 'continue', 'skip', 'show matches'}
YES = {'yes', 'yes i do', 'yes i have', 'yeah', 'yep'}
NO = {'no', 'no i do not', "no i don't", 'nope'}
QUALIFIED_YES = {'a little', 'a little bit', 'little bit', 'a bit', 'somewhat',
                 'slightly', 'kind of', 'sort of', 'mildly'}
QUESTION_EXCLUDE = {'family_history'}
COMMON_HEADACHE_SYMPTOMS = {'headache', 'fatigue', 'loss_of_appetite', 'nausea', 'dizziness'}
MAX_FOLLOWUP_QUESTIONS = 2
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
            if symptom not in seen and symptom not in QUESTION_EXCLUDE:
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
    pregnancy_mention = re.search(r'\b(?:pregnant|pregnancy)\b', text, re.I)
    pregnancy_context = bool(profile and profile.get('pregnancy') == 'Yes')
    if (re.search(r'\b(?:ibuprofen|nsaids?)\b', text, re.I)
            and ((pregnancy_mention and not negated(text, pregnancy_mention.start())) or pregnancy_context)
            and re.search(r'\b(?:can|could|should|safe|take|use)\b', text, re.I)):
        source = next(record['sources'][0] for record in data['reference_conditions']
                      if record['name'] == 'NSAIDs in pregnancy')
        return _reply(
            'Ibuprofen is an NSAID. The FDA advises avoiding NSAIDs at around 20 weeks of pregnancy '
            'or later unless a healthcare professional specifically advises them. I cannot tell whether '
            'ibuprofen is suitable for you. Please ask your pregnancy clinician or pharmacist before taking it. '
            'How many weeks pregnant are you?',
            intent='medicine_safety', medicine_withheld=True,
            sources=[source['url']], source_details=[source],
        )
    if state.get('medical_files'):
        # Reports never bypass urgent-headache, pregnancy-medicine or dose boundaries.
        if re.search(r'\b(?:dosage|dose|dosing)\b|how (?:many|much|often|frequently).*(?:tablet|pill|medicin|take)', text, re.I):
            return _reply('I can help you understand the information printed in your report, but I can’t choose a dose or dosing schedule for you. '
                          'Please confirm the medicine and instructions with your pharmacist or clinician.', intent='dose_question', medicine_withheld=True)
        from src.general_qa import answer_general_question
        return (general_answer or answer_general_question)(text, state, data)
    from src.conversation import remember_details, context_reply
    remember_details(text, state)
    norm = normalize(text)
    if norm in SKIP_WORDS:
        state['pending'] = None
        state['pending_detail'] = None
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
        if existing and existing.get('intent') in {'summary', 'explain_match', 'greeting', 'gratitude', 'report_help'}:
            return existing
        return answer_general(text, state, data)
    positive, negative = extract_symptoms(text, data['symptoms'])
    pending = state['pending']
    qualified_answer = pending and norm in QUALIFIED_YES
    if pending and (norm in YES or re.match(r'^(yes|yeah|yep)\b', norm) or qualified_answer):
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
    if qualified_answer:
        state['details'].setdefault('qualified_symptoms', {})[pending] = 'a little'
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
        return _reply('I haven’t identified a symptom from that message yet. Could you describe what you’re feeling, '
                      'such as a cough, headache, itching or stomach pain?')
    if not (positive or negative) and not detail_answered:
        if infer_profile(text):
            return _reply('I’ve noted that context. Medicine suitability requires a clinician’s review. Add any remaining symptoms, or update your Health context.')
        if norm not in SKIP_WORDS:
            if state['details'].get('duration') and len(state['questions_asked']) >= MAX_FOLLOWUP_QUESTIONS:
                state['pending'] = None
                labels = natural_list([data['by_symptom'][s]['label'] for s in state['symptoms']])
                return _reply(f'You mentioned **{labels}** for **{state["details"]["duration"]}**. '
                              'These symptoms can have several causes, so I cannot name a reliable condition '
                              'from this information alone. If they persist, worsen, or concern you, '
                              'a healthcare professional can assess them.', uncertain=True)
            return _reply('Thank you for adding that detail. Could you clarify how it relates to your symptoms? '
                          'You can also answer the previous question or describe anything that has changed.')
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
            and len(state['questions_asked']) < MAX_FOLLOWUP_QUESTIONS):
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
    if len(state['symptoms']) < 3 and missing and norm not in SKIP_WORDS and len(state['questions_asked']) < MAX_FOLLOWUP_QUESTIONS:
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
            'condition match. Here are the available dataset matches for your symptoms; '
            'they are uncertain possibilities, not a diagnosis. Medicine information is withheld. '
            'If symptoms persist or worry you, a healthcare professional can assess them.',
            predictions, uncertain=True, result_ready=True, medicine_withheld=True,
        )
    if set(state['symptoms']) <= COMMON_HEADACHE_SYMPTOMS:
        source = next((record['sources'][0] for record in data['reference_conditions']
                       if record['name'] == 'Headaches'), None)
        duration = state['details'].get('duration', 'some time')
        if re.fullmatch(r'1 week (?:before|ago)', normalize(duration)):
            duration = 'about a week'
        described = natural_list([
            ('a little ' if state['details'].get('qualified_symptoms', {}).get(symptom) else '') +
            data['by_symptom'][symptom]['label'] for symptom in state['symptoms']
        ])
        return _reply(
            f'You’ve had **{described}** for **{duration}**. These symptoms '
            'can have several causes; there is not enough information here to identify a specific condition. '
            'Rest, regular meals and enough fluids may help. Because the symptoms are continuing, '
            'consider speaking with a healthcare professional, especially if they are worsening or unusual for you. '
            'Seek urgent help for a sudden, extremely painful headache or new weakness, confusion, or vision loss.',
            predictions, uncertain=True, result_ready=True, medicine_withheld=True,
            sources=[source['url']] if source else [],
            source_details=[source] if source else [],
        )
    if low:
        answer = (f'You mentioned **{labels}**. These symptoms can happen for different reasons, '
                  'so I need a little more detail before discussing possible conditions or medicine information.')
        if state.get('details', {}).get('duration'):
            answer += ' I’ve also noted how long this has been going on.'
        if missing and len(state['questions_asked']) < MAX_FOLLOWUP_QUESTIONS:
            state['pending'] = missing
            state['questions_asked'].append(missing)
            answer += f'\n\nHave you also had **{data["by_symptom"][missing]["label"]}**?'
        else:
            answer += ' Please add any other symptoms or speak with a healthcare professional if you remain concerned.'
        return _reply(answer, predictions, uncertain=True,
                      result_ready=state['pending'] is None, medicine_withheld=True)
    record = data['by_condition'][best['condition']]
    state['last_condition'] = best['condition']
    answer = (
        f'Based on **{labels}**, one possible condition is **{best["condition"]}**. This is not a diagnosis.\n\n'
        f'{record["description"]}\n\n'
    )
    if record.get('data_type') == 'illustrative_condition_profile':
        answer += 'This label is supported by illustrative training patterns and has no independent clinical evaluation.\n\n'
    withheld = profile_has_context(context) or not record['medications']
    if withheld:
        if not record['medications']:
            answer += 'This illustrative condition profile has no verified medicine mapping. Medicine information is withheld; a clinician should assess the cause and treatment.\n\n'
        else:
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
