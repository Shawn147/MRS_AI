import json
import unittest
from unittest.mock import Mock

from src.data import ARTIFACT_DIR, fingerprint, load_data
from src.dialogue import emergency, extract_symptoms, new_state, respond


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_data()

    def setUp(self):
        self.state=new_state()
        self.predict=Mock(return_value=[{'condition':'Common Cold','probability':.9},
                                       {'condition':'Allergy','probability':.06},
                                       {'condition':'Pneumonia','probability':.02}])

    def test_unsupported_input_does_not_predict(self):
        answer=respond('write my homework',self.state,self.data,self.predict)
        self.assertFalse(answer['predictions'])
        self.predict.assert_not_called()

    def test_negation_and_longer_alias(self):
        positive,negative=extract_symptoms('No cough, but mild fever and nausea',self.data['symptoms'])
        self.assertIn('cough',negative)
        self.assertIn('mild_fever',positive)
        self.assertNotIn('high_fever',positive)
        self.assertIn('nausea',positive)

    def test_followup_yes_and_memory(self):
        respond('I have cough',self.state,self.data,self.predict)
        pending=self.state['pending']
        self.assertIsNotNone(pending)
        answer=respond('yes',self.state,self.data,self.predict)
        self.assertIn('cough',self.state['symptoms'])
        self.assertIn(pending,self.state['symptoms'])
        self.assertIn('Common Cold',answer['text'])

    def test_correction_removes_symptom(self):
        respond('cough and fever',self.state,self.data,self.predict)
        respond('no cough',self.state,self.data,self.predict)
        self.assertNotIn('cough',self.state['symptoms'])
        self.assertIn('cough',self.state['denied'])

    def test_no_does_not_enter_symptoms(self):
        respond('cough',self.state,self.data,self.predict)
        pending=self.state['pending']
        respond('no',self.state,self.data,self.predict)
        self.assertIn(pending,self.state['denied'])
        self.assertNotIn(pending,self.state['symptoms'])

    def test_emergency_persists(self):
        first=respond('chest pain',self.state,self.data,self.predict)
        later=respond('cough and fever',self.state,self.data,self.predict)
        self.assertTrue(first['urgent'])
        self.assertTrue(later['urgent'])
        self.predict.assert_not_called()
        self.assertFalse(emergency('no chest pain, but cough'))

    def test_low_confidence_has_no_medicine_list(self):
        self.predict.return_value[0]['probability']=.3
        answer=respond('cough and fever',self.state,self.data,self.predict)
        self.assertTrue(answer['uncertain'])
        self.assertNotIn('Educational medicine information',answer['text'])

    def test_context_withholds_medicine_names(self):
        answer=respond('cough and fever',self.state,self.data,self.predict,{'allergies':'penicillin'})
        self.assertIn('withheld',answer['text'])
        self.assertNotIn('Educational medicine information',answer['text'])

    def test_chat_allergy_is_retained(self):
        respond('I am allergic to penicillin',self.state,self.data,self.predict)
        answer=respond('cough and fever',self.state,self.data,self.predict)
        self.assertIn('withheld',answer['text'])

    def test_context_does_not_leak_between_chats(self):
        respond('I am pregnant',self.state,self.data,self.predict)
        other=new_state()
        self.assertFalse(other['context'])
        self.assertFalse(other['symptoms'])

    def test_split_has_no_pattern_overlap(self):
        split=json.loads((ARTIFACT_DIR/'splits.json').read_text())
        a,b,c=[set(split[k]) for k in ['train','validation','test']]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(len(a|b|c),304)
        self.assertEqual(self.data['manifest']['duplicates_removed'],4616)

    def test_duplicate_csv_feature_is_merged(self):
        from scripts.prepare_json import read
        rows=read('Training.csv')
        self.assertEqual(len(rows[0]),132)  # 131 unique features plus prognosis
        self.assertEqual(sum(int(r['fluid_overload']) for r in rows),114)

    def test_reported_typo_and_ulcer_followup(self):
        respond('I have a runny nose, sneezing and cough',self.state,self.data,self.predict)
        self.predict.reset_mock()
        answer=respond('have dirhea and ulcer',self.state,self.data,self.predict)
        self.assertIn('diarrhoea',self.state['symptoms'])
        self.assertIn('cough',self.state['symptoms'])
        self.assertIn('ulcer_in_chat',self.state['context'])
        self.assertIn('“dirhea” as “diarrhea”',answer['text'])
        self.assertIn('Is it a stomach ulcer',answer['text'])
        self.assertFalse(answer['predictions'])
        self.predict.assert_not_called()
        clarified=respond('stomach ulcer',self.state,self.data,self.predict)
        self.assertIn('noted that context',clarified['text'])
        self.assertNotIn('Is it a stomach ulcer',clarified['text'])

    def test_negated_typo_and_ulcer(self):
        respond('diarrhea and cough',self.state,self.data,self.predict)
        answer=respond('no dirhea and no ulcer',self.state,self.data,self.predict)
        self.assertNotIn('diarrhoea',self.state['symptoms'])
        self.assertNotIn('ulcer_in_chat',self.state['context'])
        self.assertNotIn('Is it a stomach ulcer',answer['text'])

    def test_specific_ulcer_context_on_first_turn(self):
        answer=respond('I have a stomach ulcer',self.state,self.data,self.predict)
        self.assertIn('noted that medical context',answer['text'])
        self.assertIn('ulcer_in_chat',self.state['context'])
        self.predict.assert_not_called()

    def test_spelling_is_bounded(self):
        from src.dialogue import correct_spelling
        self.assertEqual(correct_spelling('dirhea researchdirhea')[0],'diarrhea researchdirhea')

    def test_metrics_match_json(self):
        m=json.loads((ARTIFACT_DIR/'metrics.json').read_text())
        self.assertEqual(m['data_fingerprint'],fingerprint())
        self.assertTrue(m['training']['encoder_fine_tuned'])
        self.assertEqual(m['split_sizes'],{'train':182,'validation':61,'test':61})
        self.assertEqual(len(self.data['by_condition']),41)


if __name__=='__main__':unittest.main()
