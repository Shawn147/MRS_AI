"""Collect bounded public-source excerpts; never generate patient or training rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/health_library'
TODAY = date.today().isoformat()

class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []
    def text(self):
        return re.sub(r'\s+', ' ', ' '.join(c.text() if isinstance(c, Node) else c for c in self.children)).strip()
    def walk(self, tag=None):
        if tag is None or self.tag == tag:
            yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk(tag)

class HTML(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.root=Node(); self.stack=[self.root]; self.feed(text)
    def handle_starttag(self, tag, attrs):
        node=Node(tag, attrs); self.stack[-1].children.append(node)
        if tag not in {'br','img','meta','input','link','hr','source','wbr','area','embed'}:
            self.stack.append(node)
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack=self.stack[:i]; break
    def handle_data(self, data):
        self.stack[-1].children.append(data)

def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent':'MRS-AI educational source collector/1.0'}), timeout=35) as r:
                return r.read().decode('utf-8'), r.geturl()
        except Exception:
            if attempt == 2: raise
            time.sleep(1 + attempt)

def save(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path=OUT / name; tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n'); tmp.replace(path)

def nhs_page(url):
    text, canonical=fetch(url)
    if urlparse(canonical).hostname != 'www.nhs.uk':
        raise ValueError('Unexpected source redirect')
    tree=HTML(text).root
    main=next(tree.walk('main'), None)
    if main is None: raise ValueError('Missing main content')
    title=next(main.walk('h1'), None)
    if not title: raise ValueError('Missing title')
    terms=[]; notes=[]; description=''; heading=''; total=0
    # Whole paragraphs/list items only: no cut-off medical instructions.
    for n in main.walk():
        if n.tag in {'h2','h3'}: heading=n.text()
        if n.tag not in {'p','li'}: continue
        value=n.text(); words=len(value.split())
        if not value or words>55 or total+words>180: continue
        if any(x in value.lower() for x in ['page last reviewed','page last updated','media last reviewed','video transcript','subscribe','cookies','nhs 111','999','call 111','call 999']): continue
        if n.tag=='p' and not description and words>=8:
            description=value; total+=words
        elif n.tag=='li' and 'symptom' in heading.lower() and words<=30:
            if value not in terms: terms.append(value); total+=words
        elif n.tag=='p' and re.search(r'treat|help|manage|medicine',heading,re.I) and len(notes)<2:
            notes.append(value); total+=words
    if not description: raise ValueError('No suitable introductory paragraph')
    child_symptom_urls=[]
    for a in main.walk('a'):
        dest=urljoin(canonical,a.attrs.get('href',''))
        if urlparse(dest).hostname=='www.nhs.uk' and '/symptoms/' in urlparse(dest).path and dest.startswith(canonical):
            child_symptom_urls.append(dest.split('#')[0])
    return {'name': title.text(), 'description':description, 'symptom_terms':terms,
            'care_notes':notes, 'seek_help_notes':[],
            'sources':[{'publisher':'NHS','url':canonical,'accessed_on':TODAY,'date_not_published':True,
                        'attribution':f'Information from the NHS website, as at {TODAY}. Licensed under the Open Government Licence v3.0.',
                        'license_url':'https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/'}],
            'eligible_for_training':False,'clinically_reviewed':False,'data_type':'public_source_excerpt',
            'scope':'Bounded excerpts, not a complete symptom list or medical guideline; UK source, local applicability not assessed.',
            'symptom_page_urls':sorted(set(child_symptom_urls))}

def collect_conditions():
    text,_=fetch('https://www.nhs.uk/conditions/')
    urls=set()
    for a in HTML(text).root.walk('a'):
        dest=urljoin('https://www.nhs.uk',a.attrs.get('href','')).split('#')[0]
        path=urlparse(dest).path
        if urlparse(dest).hostname=='www.nhs.uk' and re.fullmatch(r'/conditions/[^/]+/',path): urls.add(dest)
    cache={r['sources'][0]['url']:r for r in json.loads((OUT/'conditions.json').read_text())} if (OUT/'conditions.json').exists() else {}
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(nhs_page,u):u for u in sorted(urls) if u not in cache}
        for i,future in enumerate(as_completed(pending),1):
            try:
                row=future.result(); cache[row['sources'][0]['url']]=row
            except Exception as exc: failures.append({'url':pending[future],'error':str(exc)})
            if i%25==0:
                save('conditions.json',list(cache.values())); print('condition pages',len(cache),'failures',len(failures),flush=True)
    # Some NHS topics place symptom lists on a dedicated child page.
    children={u for r in cache.values() if not r['symptom_terms'] for u in r['symptom_page_urls']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(nhs_page,u):u for u in sorted(children)}
        for future in as_completed(pending):
            try:
                child=future.result()
                for row in cache.values():
                    if pending[future] in row['symptom_page_urls']:
                        row['symptom_terms']=child['symptom_terms']
                        row['symptom_source']=child['sources'][0]
            except Exception as exc: failures.append({'url':pending[future],'error':str(exc)})
    rows=sorted(cache.values(),key=lambda r:r['name'].casefold())
    save('conditions.json',rows); save('collection_failures.json',failures)
    # Scoped phrases, deduplicated by text rather than counting aliases as new symptoms.
    symptoms={}
    for row in rows:
        for term in row['symptom_terms']:
            key=re.sub(r'\W+',' ',term.casefold()).strip()
            entry=symptoms.setdefault(key,{'term':term,'topics':[], 'sources':[], 'data_type':'source_symptom_phrase', 'eligible_for_training':False})
            if row['name'] not in entry['topics']: entry['topics'].append(row['name'])
            src=row.get('symptom_source',row['sources'][0])
            if src not in entry['sources']: entry['sources'].append(src)
    save('symptoms.json',list(symptoms.values()))
    print('condition topics',len(rows),'distinct scoped symptom phrases',len(symptoms),flush=True)

def excerpt(values, max_words):
    text=' '.join(values or [])
    text=re.sub(r'<[^>]+>',' ',text); text=re.sub(r'\s+',' ',text).strip()
    # Retain whole sentences only, never crop a warning mid-sentence.
    result=[]
    for sentence in re.split(r'(?<=[.!?])\s+',text):
        if sum(len(s.split()) for s in result)+len(sentence.split())>max_words: break
        result.append(sentence)
    return ' '.join(result)

def medicine_key(name):
    parts = re.split(r'\s+(?:and|with)\s+|[,;/+]', name.casefold())
    normalized = {re.sub(r'\W+', ' ', part).strip() for part in parts}
    return ' / '.join(sorted(part for part in normalized if part))

def medicine_batches(query, start):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for base in range(start, 25000, 2000):
            urls = ['https://api.fda.gov/drug/label.json?'+urlencode({'search':query,'limit':500,'skip':skip,'sort':'effective_time:desc'}) for skip in range(base, min(base+2000,25000), 500)]
            futures = [(url, pool.submit(fetch, url)) for url in urls]
            for i, (url, future) in enumerate(futures):
                text, _ = future.result()
                yield base+i*500, url, json.loads(text)

def collect_medicines(target=1210, start=0):
    rows={medicine_key(r['name']):r for r in json.loads((OUT/'medicines.json').read_text())} if (OUT/'medicines.json').exists() else {}
    query='_exists_:openfda.generic_name AND _exists_:openfda.application_number AND _exists_:indications_and_usage'
    for skip, url, payload in medicine_batches(query, start):
        if len(rows)>=target: break
        for label in payload['results']:
            meta=label.get('openfda',{}); apps=meta.get('application_number',[])
            if not any(a.startswith(('NDA','ANDA','BLA')) for a in apps): continue
            names=meta.get('generic_name',[])
            # A combination product is one name, not one new medicine per brand/formulation.
            name=' / '.join(sorted(set(n.strip().lower() for n in names if n.strip())))
            key=medicine_key(name)
            if not key or key in rows: continue
            indication=excerpt(label.get('indications_and_usage'),65)
            if not indication: continue
            sid=label['set_id']
            rows[key]={'name':name,'brand_names':meta.get('brand_name',[]),'route':meta.get('route',[]),
                        'indications_excerpt':indication,'warnings_excerpt':excerpt(label.get('boxed_warning') or label.get('warnings_and_cautions') or label.get('warnings'),65),
                        'contraindications_excerpt':excerpt(label.get('contraindications'),40),
                        'source':{'publisher':'US drug label submitted to FDA','url':'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid='+sid,'api_url':url,'accessed_on':TODAY,'label_effective_date':label.get('effective_time'),'set_id':sid,'application_numbers':apps},
                        'data_type':'drug_label_excerpt','eligible_for_training':False,'prescribing_approved':False,
                        'pakistan_availability_verified':False,'scope':'US label excerpt; not complete safety information or a personal treatment recommendation.'}
        save('medicines.json',sorted(rows.values(),key=lambda r:r['name'])); print('distinct medicine names',len(rows),'labels scanned',skip+500,flush=True)
        save('medicine_progress.json', {'next_skip':skip+500,'distinct_names':len(rows)})
    print('medicine collection finished',len(rows),flush=True)

def manifest():
    files={n:json.loads((OUT/n).read_text()) for n in ['conditions.json','symptoms.json','medicines.json']}
    result={'collected_on':TODAY,'baseline':{'classifier_condition_labels':51,'classifier_symptom_features':156,'legacy_medicine_entries':121},'requested_targets':{'conditions':510,'symptom_phrases':1560,'medicine_names':1210},'collected':{n[:-5]:len(v) for n,v in files.items()},'sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files},'scope':'Reference library only. Counts do not represent classifier labels/features or patient cases. Symptom phrases are scoped source descriptions, not medically distinct normalized features.'}
    save('manifest.json',result); print(json.dumps(result['collected']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['conditions','medicines','manifest','all'],default='all');parser.add_argument('--medicine-start',type=int,default=0);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.part in {'conditions','all'}: collect_conditions()
    if args.part in {'medicines','all'}: collect_medicines(start=args.medicine_start)
    if args.part in {'manifest','all'}: manifest()
