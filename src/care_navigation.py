"""Keep hospital-location conversations out of symptom classification.

Maps search links do not claim verified facilities, distance or opening hours.
"""
import re
from urllib.parse import urlencode

FACILITY = re.compile(r'\b(?:hospitals?|clinics?|urgent care|emergency department)\b', re.I)
FIND = re.compile(r'\b(?:near|nearby|nearest|closest|where|which|find|looking for|recommend|suggest)\b', re.I)


def hospital_request(text):
    return bool(FACILITY.search(text) and FIND.search(text))


def location_reply(text, awaiting=False):
    value = text.strip().rstrip('.!?')
    match = re.match(r"^(?:i live in|i live at|i am in|i'm in|im in|i am from|i'm from|my location is|my area is|located in)\s+(.+)$", value, re.I)
    if match:
        value = match[1].strip()
    elif not awaiting:
        return None
    elif re.search(r'\?|\b(?:pain|hurt|cough|fever|numbness|weakness|symptom|medicine|yes|no)\b|^(?:i|my|what|why|how|which|can|tell|explain|hi|hello|thanks|okay|cancel|stop)\b', value, re.I):
        return None
    if re.search(r'\b(?:pain|hurt|cough|fever|numbness|weakness|symptom|medicine)\b|'
                 r'^(?:\d+|one|two|three|four|five|six|seven)\s+(?:days?|weeks?|hours?)$', value, re.I):
        return None
    if not 2 <= len(value) <= 150 or not re.search(r'[a-zA-Z\u0600-\u06ff]', value):
        return None
    return value


def navigation_reply(text, state):
    requested = hospital_request(text)
    if (state.get('location') and state.get('active_general_topic') == 'Finding a nearby hospital'
            and re.search(r'\b(?:directions?|address|names?|closest|nearest)\b', text, re.I)):
        requested = True
    awaiting = state.get('pending_location', False) or (
        not state.get('location') and hospital_request(state.get('active_general_topic', '')))
    # Recover a pending hospital search in chats saved before this field existed.
    if not awaiting:
        for turn in reversed(state.get('recent_turns', [])):
            if turn.get('role') == 'user':
                awaiting = hospital_request(turn.get('content', ''))
                break
    location = location_reply(text, awaiting=awaiting) if not requested else None
    if awaiting and not requested and location is None and re.search(r'\b(?:pain|hurt|cough|fever|numbness|weakness)\b', text, re.I):
        state['pending_location'] = False
    if requested:
        match = re.search(r'\b(?:near|in|around)\s+(.+)$', text, re.I)
        if match and not re.fullmatch(r'(?:to )?me[?.!]*', match[1].strip(), re.I):
            location = location_reply(match[1], awaiting=True)
    if not requested and not (awaiting and location):
        return None
    state['active_general_topic'] = 'Finding a nearby hospital'
    state['pending'] = None
    state['pending_detail'] = None
    state['last_condition'] = None
    state['last_predictions'] = []
    if location:
        state['location'] = location
    place = state.get('location')
    if not place:
        state['pending_location'] = True
        return {'text': 'Which area and city are you in?', 'predictions': [], 'intent': 'hospital_location'}
    state['pending_location'] = False
    url = 'https://www.google.com/maps/search/?' + urlencode({'api': '1', 'query': 'hospitals near ' + place})
    display = re.sub(r'[\\`*_\[\]<>]', '', place)
    if display.islower():
        display = display.title()
    return {'text': 'For **' + display + '**: [Find nearby hospitals on Maps](' + url + ').\n\n'
                    'Compare distances, directions and opening hours there.',
            'predictions': [], 'intent': 'hospital_search', 'location': place}
