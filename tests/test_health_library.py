import copy
import unittest
from unittest.mock import patch
from urllib.error import URLError

from scripts.collect_health_library import HTML, excerpt, medicine_key
from scripts.audit_health_library import validate
from src.health_library import load_health_library, answer_records
from src.general_qa import _matched_references, answer_general_question
from src.data import load_data
from src.dialogue import new_state


class HealthLibraryTests(unittest.TestCase):
    def test_excerpts_do_not_crop_medical_sentences(self):
        self.assertEqual(excerpt(['Short sentence. Do not take this product when a dangerous condition exists.'],5),'Short sentence.')
        self.assertEqual(excerpt(['One enormous incomplete sentence with a safety instruction'],3),'')

    def test_parser_keeps_nested_list_text(self):
        tree=HTML('<main><h1>Topic</h1><ul><li>pain <strong>at night</strong></li></ul></main>').root
        self.assertEqual(next(tree.walk('li')).text(),'pain at night')

    def test_combination_name_order_and_punctuation_are_not_new_medicines(self):
        self.assertEqual(medicine_key('Aspirin and Caffeine'),medicine_key('caffeine / aspirin'))
        self.assertEqual(medicine_key('Famotidine, calcium carbonate and magnesium hydroxide'),
                         medicine_key('famotidine / magnesium hydroxide / calcium carbonate'))

    def test_real_library_is_valid_and_not_training_evidence(self):
        library=load_health_library()
        if not library['manifest']:
            self.skipTest('Collection still in progress')
        self.assertEqual(validate(library),[])
        self.assertGreaterEqual(len(library['conditions']),510)
        self.assertGreater(len(library['symptoms']),156)
        records=list(answer_records(library))
        chosen=library['medicines'][0]['name']
        matches=_matched_references('Explain '+chosen,records)
        self.assertTrue(any(r['name']==chosen for r in matches))
        self.assertTrue(all(r['eligible_for_training'] is False for r in records))

    def test_scope_escalation_and_source_spoof_rejected(self):
        library=load_health_library()
        if not library['manifest']:
            self.skipTest('Collection still in progress')
        library=copy.deepcopy(library)
        library['medicines'][0]['prescribing_approved']=True
        library['conditions'][0]['sources'][0]['url']='https://www.nhs.uk.fake.example/page'
        self.assertGreaterEqual(len(validate(library)),2)

    def test_new_topic_is_used_when_answer_provider_is_unavailable(self):
        data=load_data()
        if not data['health_library']['manifest']:
            self.skipTest('Collection still in progress')
        with patch('src.general_qa.urlopen', side_effect=URLError('offline')):
            result=answer_general_question('What is shingles?',new_state(),data)
        self.assertEqual(result['intent'],'reference_fallback')
        self.assertTrue(any('/shingles/' in s for s in result['sources']))
        self.assertIn('Shingles',result['text'])

    def test_condition_short_name_is_matched(self):
        library={'conditions':[{'name':'Underactive thyroid (hypothyroidism)'}], 'medicines':[]}
        records=list(answer_records(library))
        self.assertEqual(len(_matched_references('What is hypothyroidism?',records)),1)
        self.assertEqual(len(_matched_references('What is underactive thyroid?',records)),1)
