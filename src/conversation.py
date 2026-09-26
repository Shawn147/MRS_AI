"""Context memory and constrained follow-up replies, separate from diagnosis labels."""
import logging
import re


def remember_details(text, state):
    details = state.setdefault('details', {})
    duration = re.search(r'\b(?:for\s+(?:the\s+)?(?:last\s+)?|since\s+|started\s+|began\s+)([^.!?;,]{1,55})', text, re.I)
    if duration and re.search(r'\b(hour|day|week|month|year|morning|night|afternoon|evening|yesterday|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|fortnight)', duration.group(), re.I):
        details['duration'] = duration.group().strip()
    severity = re.search(r'\b(?:\d{1,2}\s*(?:/|out of)\s*10|mild|moderate|severe)\b', text, re.I)
    if severity and not re.search(r'\b(?:not|no|never)\s+(?:very\s+)?$', text[:severity.start()], re.I):
        details['severity'] = severity.group().lower()


def context_reply(text, state, data, blocked):
    from src.context_model import classify_intent
    norm = text.lower()
    # Explicit medication boundaries are handled even when model confidence is low.
    if re.search(r'\b(dosage|dose|dosing|milligrams|mg)\b|how (?:many|much|often|frequently).*(?:tablet|pill|medicin|take)', norm):
        intent = 'dose_question'
    elif re.search(r'\b(medicine|medication|tablet|painkiller|antibiotic|paracetamol|ibuprofen)\w*\b', norm) and re.search(r'\b(what|which|can|could|should|show|tell|take)\b', norm):
        intent = 'medicine_question'
    else:
        try:
            intent = classify_intent(text)
        except (FileNotFoundError, ValueError, OSError, ImportError):
            logging.warning('Context model unavailable; using the symptom dialogue fallback.')
            return None
    def reply(content, **extra):
        return {'text': content, 'predictions': [], 'intent': intent, **extra}
    details = state.setdefault('details', {})
    labels = [data['by_symptom'][key]['label'] for key in state['symptoms']]
    if intent == 'dose_question':
        return reply('I can’t calculate a dose or dosing schedule from this conversation. '
                     'Please check the medicine’s leaflet and ask a pharmacist or clinician about the correct dose for you.', medicine_withheld=True)
    if intent == 'summary':
        parts = ['**Your conversation so far**', '**Symptoms:** ' + (', '.join(labels) if labels else 'No active symptoms recorded yet.')]
        if state['denied']:
            parts.append('**Symptoms you denied:** ' + ', '.join(data['by_symptom'][s]['label'] for s in state['denied']))
        parts.extend(f'**{key.title()}:** {value}' for key, value in details.items())
        if state.get('last_condition'):
            parts.append('**Previous possible match:** ' + state['last_condition'] + ' (not a diagnosis).')
        if blocked:
            parts.append('**Health context noted:** medicine suitability needs a clinician’s review.')
        if state['pending']:
            parts.append('**Still to clarify:** whether you also have ' + data['by_symptom'][state['pending']]['label'] + '.')
        return reply('\n\n'.join(parts))
    if intent == 'explain_match':
        condition = state.get('last_condition')
        if not condition:
            return reply('There isn’t a clear condition match to explain yet. ' +
                         ('I’m keeping track of ' + ', '.join(labels) + '. ' if labels else '') +
                         'Tell me more about your symptoms or answer the follow-up question.')
        record = data['by_condition'][condition]
        overlap = [data['by_symptom'][s]['label'] for s in state['symptoms'] if s in record['symptoms']]
        return reply(f'**{condition}** came up as one possible match for the symptoms you described.\n\n' + record['description'] +
                     '\n\nSymptoms shared with its source record: ' + (', '.join(overlap) or 'No direct overlap recorded') +
                     '. This overlap alone cannot confirm the cause. I’ve kept any duration or severity details you shared '
                     'in our conversation, but they do not change how the condition match is calculated. A clinician can assess the cause.')
    if intent == 'medicine_question':
        condition = state.get('last_condition')
        if blocked:
            return reply('Your health context is noted. Medicine names remain withheld because this dataset cannot '
                         'check suitability, allergies or interactions. Please discuss your context with a pharmacist or clinician.', medicine_withheld=True)
        if not condition:
            return reply('I don’t have a clear symptom-based match to attach medicine information to. '
                         'Describe your symptoms first. I can show educational references, but cannot choose a medicine for you.', medicine_withheld=True)
        record = data['by_condition'][condition]
        return reply('**Educational medicine information for the previous possible match: ' + condition + '**\n\n' +
                     '\n'.join('- ' + name for name in record['medications'][:5]) +
                     '\n\nThese are unreviewed source references, not prescriptions or a suitability check.',
                     condition=condition, medicine_withheld=False)
    if intent in {'duration', 'severity', 'improving', 'worsening'}:
        if not labels:
            return reply('I can note that detail once I know which symptoms you mean. What symptoms are you experiencing?')
        if intent == 'duration':
            details['duration'] = text.strip()[:100]
        elif intent == 'severity':
            details['severity'] = text.strip()[:100]
        elif intent in {'improving', 'worsening'}:
            details['trend'] = intent
        lead = {'duration': 'I’ve noted how long this has been happening.',
                'severity': 'I’ve noted the severity you described.',
                'improving': 'I’ve noted that you’re feeling better. This doesn’t confirm the cause.',
                'worsening': 'I’ve noted that your symptoms are worsening or not improving. Please seek a clinician’s assessment.'}[intent]
        content = lead + '\n\nI’m still keeping track of **' + ', '.join(labels) + '**.'
        if state['pending']:
            content += '\n\nDo you also have **' + data['by_symptom'][state['pending']]['label'] + '**?'
        else:
            content += ' Have any symptoms changed or have you noticed anything new?'
        return reply(content)
    return None
