"""Import pinned, public Asian clinical cohorts without changing chatbot training.

python scripts/collect_regional_data.py --raw-dir /path/to/downloads
python scripts/collect_regional_data.py --download

Only allowlisted clinical columns are retained. No synthetic cases, imputations,
diagnosis translations, or clinical interpretation of coded values are added.
"""
import argparse
import csv
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data/regional'
MAX_BYTES = 10 * 1024 * 1024
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ('', '?', 'NaN', 'nan') else text


def read_xlsx(raw, sheet_path):
    """Read cached scalar cells only. Never run formulas or workbook code."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(f.file_size for f in archive.infolist()) > 50 * 1024 * 1024:
            raise ValueError('Workbook decompressed size exceeds limit')
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            strings = [''.join(t.text or '' for t in node.findall('.//s:t', NS))
                       for node in root.findall('s:si', NS)]
        root = ET.fromstring(archive.read(sheet_path))
        rows = []
        for row in root.findall('s:sheetData/s:row', NS):
            if int(row.attrib['r']) != len(rows) + 1:
                raise ValueError('Non-contiguous worksheet rows would lose source-row provenance')
            values = {}
            for cell in row.findall('s:c', NS):
                letters = ''.join(ch for ch in cell.attrib['r'] if ch.isalpha())
                index = 0
                for letter in letters:
                    index = index * 26 + ord(letter.upper()) - 64
                node = cell.find('s:v', NS)
                value = node.text if node is not None else None
                kind = cell.attrib.get('t')
                if kind == 's' and value is not None:
                    value = strings[int(value)]
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                elif kind == 'e':
                    raise ValueError('Spreadsheet error cell in clinical source')
                values[index - 1] = value
            rows.append([values.get(i) for i in range(max(values, default=-1) + 1)])
        return rows


def read_table(raw, source):
    if source['format'] == 'xlsx':
        table = read_xlsx(raw, source['sheet_path'])
    else:
        table = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    if not table:
        raise ValueError('Empty source')
    header = [clean(x) for x in table[0]]
    if header != source['source_columns'] or len(set(header)) != len(header):
        raise ValueError(f"Unexpected source schema: {source['id']}")
    rows = []
    for line, values in enumerate(table[1:], 2):
        if not any(clean(x) is not None for x in values):
            raise ValueError(f'Empty source row {line}')
        if len(values) > len(header):
            raise ValueError(f'Extra cells at source row {line}')
        values = values + [None] * (len(header) - len(values))
        rows.append((line, dict(zip(header, map(clean, values)))))
    if len(rows) != source['expected_rows']:
        raise ValueError(f"Source row count changed: {source['id']}")
    return rows


def prepare_records(source, rows):
    groups = defaultdict(list)
    allowed = set(source['features'] + source['targets'])
    if not allowed <= set(source['source_columns']):
        raise ValueError('Allowlisted fields missing from source')
    for line, row in rows:
        features = {k: row[k] for k in source['features']}
        targets = {k: row[k] for k in source['targets']}
        if any(v is None for v in targets.values()):
            raise ValueError(f'Missing target at row {line}')
        for key, values in source['target_values'].items():
            if targets[key] not in values:
                raise ValueError(f'Unexpected target code at row {line}')
        groups[canonical(features)].append((line, features, targets))
    records, conflicts = [], []
    for group, items in groups.items():
        targets = {canonical(item[2]) for item in items}
        group_id = digest((source['id'] + ':' + group).encode())
        if len(targets) > 1:
            # Keep all contradictory rows for review, outside the prepared data.
            for line, features, target in items:
                conflicts.append({'source_id': source['id'], 'feature_group_id': group_id,
                                  'features': features, 'target': target, 'source_rows': [line],
                                  'eligible_for_classifier_training': False})
            continue
        line, features, target = items[0]
        record_id = digest((source['id'] + ':' + canonical([features, target])).encode())
        records.append({'id': record_id, 'source_id': source['id'],
                        'feature_group_id': group_id, 'features': features, 'target': target,
                        'source_rows': sorted(item[0] for item in items),
                        'eligible_for_classifier_training': False})
    summary = {'raw_rows': len(rows), 'unique_rows': len(records),
               'duplicates_removed': sum(len(r['source_rows']) - 1 for r in records),
               'conflicting_rows': len(conflicts),
               'target_counts': dict(Counter(canonical(r['target']) for r in records)),
               'missing_feature_values': {k: sum(r['features'][k] is None for r in records)
                                          for k in source['features']},
               'excluded_columns': [k for k in source['source_columns'] if k not in allowed]}
    return records, conflicts, summary


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(rows):
    return ''.join(canonical(row) + '\n' for row in rows).encode('utf-8')


def fetch(file, raw_dir):
    path = raw_dir / file['local_name']
    request = urllib.request.Request(file['download_url'], headers={'User-Agent': 'MRS-AI-data-import/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or digest(raw) != file['sha256']:
        raise ValueError(f"Downloaded file changed or exceeded size limit: {path.name}")
    path.write_bytes(raw)


def collect(raw_dir, download=False, directory=DIRECTORY):
    catalog = json.loads((directory / 'sources.json').read_text())
    outputs, summaries = {}, []
    for source in catalog['sources']:
        file = source['file']
        if Path(file['local_name']).name != file['local_name']:
            raise ValueError('Invalid local filename')
        if download:
            fetch(file, raw_dir)
        raw = (raw_dir / file['local_name']).read_bytes()
        if len(raw) > MAX_BYTES or digest(raw) != file['sha256']:
            raise ValueError(f"Source checksum mismatch: {source['id']}")
        records, conflicts, summary = prepare_records(source, read_table(raw, source))
        name = source['id'] + '.jsonl'
        outputs[name] = jsonl(records)
        outputs[source['id'] + '.conflicts.jsonl'] = jsonl(conflicts)
        summary.update(source_id=source['id'], file=name, sha256=digest(outputs[name]),
                       conflicts_file=source['id'] + '.conflicts.jsonl',
                       conflicts_sha256=digest(outputs[source['id'] + '.conflicts.jsonl']))
        summaries.append(summary)
    total = sum(s['unique_rows'] for s in summaries)
    core = json.loads((ROOT / 'data/manifest.json').read_text())
    legacy = core.get('legacy_unique_rows', core['unique_rows'])
    manifest = {'schema_version': 1, 'accessed_on': catalog['accessed_on'],
                'catalog_sha256': digest((directory / 'sources.json').read_bytes()),
                'status': 'research_only; requires task-specific preparation and clinical validation',
                'eligible_for_classifier_training': False,
                'legacy_classifier_patterns': legacy, 'collection_target_unique_rows': legacy * 10,
                'target_met_for_research_collection': total >= legacy * 10,
                'classifier_10x_target_met_by_collection': False,
                'raw_rows': sum(s['raw_rows'] for s in summaries), 'unique_rows': total,
                'duplicates_removed': sum(s['duplicates_removed'] for s in summaries),
                'conflicting_rows': sum(s['conflicting_rows'] for s in summaries),
                'sources': summaries,
                'count_definition': 'Distinct allowlisted feature observations within each cohort, not verified independent patients or symptom-classifier examples.'}
    # Validate every input before publishing any outputs. Publish manifest last.
    directory.mkdir(parents=True, exist_ok=True)
    for name, raw in outputs.items():
        path = directory / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
    write_json(directory / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not args.download and args.raw_dir is None:
        parser.error('Use --download or provide --raw-dir')
    if args.raw_dir:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        manifest = collect(args.raw_dir, args.download)
    else:
        # Raw identifiers in the published infant workbook never enter the repo.
        with tempfile.TemporaryDirectory(prefix='mrs-regional-') as directory:
            manifest = collect(Path(directory), download=True)
    print(json.dumps({k: manifest[k] for k in ['raw_rows', 'unique_rows', 'duplicates_removed',
                     'conflicting_rows', 'target_met_for_research_collection', 'classifier_10x_target_met_by_collection']}, indent=2))


if __name__ == '__main__':
    main()
