"""Local semantic search over the project's source-linked reference summaries."""
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / 'artifacts/pretrained/medembed-small'


@lru_cache(maxsize=1)
def _model():
    if not (MODEL_DIR / 'model.safetensors').exists():
        raise FileNotFoundError('MedEmbed is missing. Run python scripts/download_medembed.py.')
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(MODEL_DIR), local_files_only=True)


def _passages(records):
    for record in records:
        if not record.get('sources'):
            continue
        name = record['name']
        yield {'condition': name, 'text': f'{name}: {record["description"]}',
               'source': record['sources'][0]['url']}
        for field in ('symptom_terms', 'care_notes', 'seek_help_notes'):
            if record.get(field):
                text = ', '.join(record[field]) if field == 'symptom_terms' else ' '.join(record[field])
                yield {'condition': name, 'text': f'{name}: {text}',
                       'source': record['sources'][0]['url']}


def search_references(query, records, limit=3):
    """Rank reviewed-source summaries; scores are similarity, not clinical confidence."""
    if not query.strip() or limit < 1:
        return []
    passages = list(_passages(records))
    if not passages:
        return []
    embeddings = _model().encode(
        [query] + [item['text'] for item in passages], normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = embeddings[1:] @ embeddings[0]
    order = np.argsort(scores)[::-1][:limit]
    return [{**passages[index], 'similarity': float(scores[index])} for index in order]
