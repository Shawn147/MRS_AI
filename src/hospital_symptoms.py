"""Published Pakistani hospital symptom references, separate from patient data."""
from datetime import date
from urllib.parse import urlparse

HOSPITAL_HOSTS = {'hospitals.aku.edu', 'www.shifa.com.pk', 'pkli.org.pk'}


def validate_hospital_symptoms(rows, symptoms):
    seen = set()
    for row in rows:
        source = row['source']
        url = urlparse(source['url'])
        identity = (row['topic'], row['term'], source['url'])
        if identity in seen or not row['topic'] or not row['term'] or not row['scope']:
            raise ValueError('Duplicate or incomplete hospital symptom reference')
        seen.add(identity)
        if (url.scheme != 'https' or url.hostname not in HOSPITAL_HOSTS or
                (url.hostname == 'hospitals.aku.edu' and not url.path.startswith('/pakistan/'))):
            raise ValueError('Unapproved Pakistani hospital source')
        date.fromisoformat(source['accessed_on'])
        if not source.get('publisher') or not (source.get('page_last_reviewed') or
                source.get('updated_on') or source.get('date_not_published') is True):
            raise ValueError('Missing hospital source provenance')
        if (row['patient_record'] is not False or row['eligible_for_training'] is not False or
                row['data_type'] != 'hospital_published_symptom_reference'):
            raise ValueError('Hospital reference cannot claim patient or training evidence')
        if row['feature_id'] is not None and row['feature_id'] not in symptoms:
            raise ValueError('Unknown hospital symptom feature mapping')
