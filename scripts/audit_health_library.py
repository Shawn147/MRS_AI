"""Audit provenance and scope of the expanded public reference library."""
import json
import re
from urllib.parse import urlparse

from src.health_library import DIRECTORY, load_health_library
from scripts.collect_health_library import medicine_key


def validate(library):
    errors=[]
    for category in ('conditions','symptoms','medicines'):
        identities=set()
        for row in library[category]:
            key=re.sub(r'\W+',' ',row.get('name',row.get('term','')).casefold()).strip()
            if category=='medicines':
                key=medicine_key(row['name'])
            if not key or key in identities: errors.append('Duplicate or empty '+category+' identity: '+key)
            identities.add(key)
            if row.get('eligible_for_training') is not False:
                errors.append('Reference presented as training evidence: '+key)
            sources=[row['source']] if category=='medicines' else row['sources']
            for source in sources:
                url=urlparse(source['url'])
                expected='dailymed.nlm.nih.gov' if category=='medicines' else 'www.nhs.uk'
                if url.scheme!='https' or url.hostname!=expected or not source.get('accessed_on'):
                    errors.append('Invalid source: '+key)
            if category=='medicines':
                if (row.get('prescribing_approved') is not False or row.get('pakistan_availability_verified') is not False
                    or not row['indications_excerpt'] or not any(a.startswith(('NDA','ANDA','BLA')) for a in row['source']['application_numbers'])):
                    errors.append('Invalid medicine scope: '+key)
                if sum(len(row[f].split()) for f in ('indications_excerpt','warnings_excerpt','contraindications_excerpt'))>170:
                    errors.append('Medicine excerpt too long: '+key)
            elif category=='conditions':
                if not row['description'] or row.get('clinically_reviewed') is not False:
                    errors.append('Invalid condition scope: '+key)
    return errors


def main():
    library=load_health_library(); errors=validate(library)
    report={'errors':errors,'counts':library['manifest']['collected'],
            'topics_with_symptom_lists':sum(bool(r['symptom_terms']) for r in library['conditions']),
            'targets_met':{k:library['manifest']['collected'][category]>=target for k,category,target in
                           [('condition_topics','conditions',510),('symptom_phrases','symptoms',1560),('medicine_names','medicines',1210)]},
            'note':'Reference counts do not measure independent clinical evidence or classifier coverage.'}
    (DIRECTORY/'quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
