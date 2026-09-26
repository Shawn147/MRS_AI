"""Local anonymous event counts; no messages, names or profile values are stored."""
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'artifacts/analytics/events.sqlite3'


def connect(path=None):
    path = Path(path or os.environ.get('MRS_ANALYTICS_DB', DEFAULT_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.execute('''CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL,
        session_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
        condition TEXT, symptoms TEXT NOT NULL, medicines TEXT NOT NULL,
        urgent INTEGER NOT NULL, withheld INTEGER NOT NULL,
        uncertain INTEGER NOT NULL, intent TEXT NOT NULL)''')
    connection.execute('CREATE INDEX IF NOT EXISTS events_date ON events(occurred_at)')
    return connection


def record_event(session_id, conversation_id, message, data, path=None):
    """One event per completed assistant turn; reruns cannot double-count a turn."""
    condition = message.get('condition')
    eligible = bool(condition and not any(message.get(k) for k in ['urgent', 'uncertain', 'medicine_withheld']))
    medicines = data['by_condition'].get(condition, {}).get('medications', [])[:5] if eligible else []
    connection = connect(path)
    try:
        with connection:
            connection.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
                message['id'], message.get('timestamp') or datetime.now(timezone.utc).isoformat(),
                session_id, conversation_id, condition,
                json.dumps(sorted(set(message.get('symptoms', [])))), json.dumps(sorted(set(medicines))),
                int(bool(message.get('urgent'))), int(bool(message.get('medicine_withheld'))),
                int(bool(message.get('uncertain'))), message.get('intent', 'symptom_assessment')))
    finally:
        connection.close()


def delete_event(event_id, path=None):
    """Remove an assistant turn superseded by an edited user message."""
    if not event_id:
        return
    connection = connect(path)
    try:
        with connection:
            connection.execute('DELETE FROM events WHERE event_id = ?', (event_id,))
    finally:
        connection.close()


def summarize(since=None, path=None):
    connection = connect(path)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute('SELECT * FROM events WHERE occurred_at >= ? ORDER BY occurred_at',
                                  (since or '',)).fetchall()
    finally:
        connection.close()
    medicines, conditions, symptoms, activity, days, intents = (Counter() for _ in range(6))
    medicine_conversations = {}
    for row in rows:
        activity[row['conversation_id']] += 1
        days[row['occurred_at'][:10]] += 1
        intents[row['intent']] += 1
        if row['condition']:
            conditions[row['condition']] += 1
        symptoms.update(json.loads(row['symptoms']))
        for medicine in json.loads(row['medicines']):
            medicines[medicine] += 1
            medicine_conversations.setdefault(medicine, set()).add(row['conversation_id'])
    return {
        'sessions': len({r['session_id'] for r in rows}), 'conversations': len(activity), 'responses': len(rows),
        'urgent_conversations': len({r['conversation_id'] for r in rows if r['urgent']}),
        'withheld_responses': sum(r['withheld'] for r in rows),
        'uncertain_responses': sum(r['uncertain'] for r in rows),
        'medicines': [{'Medicine reference': name, 'Responses': count,
                       'Conversations': len(medicine_conversations[name])} for name, count in medicines.most_common()],
        'conditions': [{'Condition match': k, 'Responses': v} for k, v in conditions.most_common()],
        'symptoms': [{'Symptom': k.replace('_', ' '), 'Responses': v} for k, v in symptoms.most_common()],
        'activity': [{'Conversation': k, 'Responses': v} for k, v in activity.most_common()],
        'daily': [{'Date (UTC)': k, 'Responses': v} for k, v in sorted(days.items())],
        'intents': [{'Reply type': k.replace('_', ' '), 'Responses': v} for k, v in intents.most_common()],
    }
