"""Collect bounded patient-education excerpts from AKUH Pakistan's public sitemap.

No patient records, treatment mappings, diagnoses or generated clinical facts.
"""
import concurrent.futures
from datetime import date
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from collect_health_library import fetch

DATA = ROOT / 'data'
DATE = date.today().isoformat()


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def text(node):
    return re.sub(r'\s+', ' ', ' '.join(n['text'] for n in walk(node)
                  if n.get('type') == 'text' and isinstance(n.get('text'), str))).strip(' \u200b')


def parse_page(html, url):
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not match:
        raise ValueError('Missing published page data')
    story = json.loads(match[1])['props']['pageProps']['story']
    if story['full_slug'].strip('/') != urlparse(url).path.strip('/'):
        raise ValueError('Source redirected to a different topic')
    content = story['content']
    blocks = content.get('blocks', [])
    sections = [item for block in blocks if block.get('component') == 'accordion'
                for item in block.get('item', [])]
    terms = []
    words = 0
    for section in sections:
        if section.get('title', '').strip().lower() != 'symptoms':
            continue
        for item in walk(section.get('content', {})):
            if item.get('type') != 'list_item':
                continue
            value = text(item)
            # Retain complete short items; do not split or invent symptom aliases.
            if value and len(value.split()) <= 25 and words + len(value.split()) <= 115 and value not in terms:
                terms.append(value)
                words += len(value.split())
    if not terms:
        raise ValueError('No explicit short symptom list')
    description = ''
    for block in blocks:
        if block.get('component') == 'paragraph':
            for node in walk(block.get('text', {})):
                value = text(node)
                if node.get('type') == 'paragraph' and 5 <= len(value.split()) <= 45:
                    description = value
                    break
        if description:
            break
    notes = []
    for block in blocks:
        if block.get('component') == 'paragraph':
            for node in walk(block.get('text', {})):
                value = text(node)
                if node.get('type') == 'paragraph' and value != description and 5 <= len(value.split()) <= 30:
                    if re.search(r'consult|seek|doctor|emergency', value, re.I):
                        notes.append(value)
                        break
        if notes:
            break
    source = {'publisher': 'Aga Khan University Hospital, Pakistan', 'url': url,
              'accessed_on': DATE, 'updated_on': story['updated_at'][:10]}
    return {'name': content.get('title') or story['name'], 'description': description,
            'symptom_terms': terms, 'care_notes': [], 'seek_help_notes': notes,
            'sources': [source], 'eligible_for_training': False, 'clinically_reviewed': False,
            'data_type': 'public_source_excerpt',
            'scope': 'Bounded Pakistani hospital patient-education excerpts; not patient observations or a complete clinical guideline.'}


def main():
    sitemap, _ = fetch('https://hospitals.aku.edu/pakistan/sitemap.xml')
    urls = sorted({u for u in re.findall(r'<loc>(.*?)</loc>', sitemap)
                   if u.startswith('https://hospitals.aku.edu/pakistan/encyclopedia/')})
    path = DATA / 'pakistan_hospital_topics.json'
    cache = {r['sources'][0]['url']: r for r in json.loads(path.read_text())} if path.exists() else {}
    failures = []
    def collect(url):
        html, canonical = fetch(url)
        if canonical.rstrip('/') != url.rstrip('/'):
            raise ValueError('Unexpected redirect')
        return parse_page(html, url)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(collect, u): u for u in urls if u not in cache}
        for i, future in enumerate(concurrent.futures.as_completed(pending), 1):
            try:
                row = future.result()
                cache[pending[future]] = row
            except Exception as exc:
                failures.append({'url': pending[future], 'error': str(exc)})
            if i % 25 == 0:
                path.write_text(json.dumps(list(cache.values()), indent=2, ensure_ascii=False) + '\n')
                print('published symptom topics', len(cache), 'skipped', len(failures), flush=True)
    rows = sorted(cache.values(), key=lambda r: r['name'].casefold())
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    (DATA / 'pakistan_hospital_collection_failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    links_path = DATA / 'pakistan_hospital_symptoms.json'
    links = json.loads(links_path.read_text())
    features = json.loads((DATA / 'symptoms.json').read_text())
    lookup = {a.casefold(): s['id'] for s in features for a in [s['label'], *s.get('aliases', [])]}
    seen = {(r['topic'].casefold(), r['term'].casefold()) for r in links}
    for row in rows:
        for term in row['symptom_terms']:
            key = (row['name'].casefold(), term.casefold())
            if key in seen:
                continue
            seen.add(key)
            links.append({'topic': row['name'], 'term': term,
                          'feature_id': lookup.get(term.casefold()), 'source': row['sources'][0],
                          'scope': row['scope'], 'data_type': 'hospital_published_symptom_reference',
                          'eligible_for_training': False, 'patient_record': False})
    links_path.write_text(json.dumps(links, indent=2, ensure_ascii=False) + '\n')
    print('hospital symptom links', len(links), 'source topics', len(rows), flush=True)


if __name__ == '__main__':
    main()
