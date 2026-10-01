import json
import unittest
from unittest.mock import Mock

from src.data import ARTIFACT_DIR, fingerprint, load_data
from src.dialogue import emergency, extract_symptoms, new_state, next_question, respond


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
        self.state['pending'] = 'runny_nose'
        pending=self.state['pending']
        self.assertIsNotNone(pending)
        answer=respond('yes',self.state,self.data,self.predict)
        self.assertIn('cough',self.state['symptoms'])
        self.assertIn(pending,self.state['symptoms'])
        self.assertNotIn('Common Cold',answer['text'])
        self.assertNotIn('Educational medicine information',answer['text'])

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

    def test_headache_after_sex_uses_context_specific_reply(self):
        answer = respond('headache after sex', self.state, self.data, self.predict)
        self.assertEqual(answer['intent'], 'sex_headache')
        self.assertIn('Did it start suddenly', answer['text'])
        self.assertIn('headache', self.state['symptoms'])
        self.assertIsNone(self.state['pending'])
        self.predict.assert_not_called()

    def test_sudden_severe_headache_after_sex_is_urgent(self):
        answer = respond('sudden severe headache during sex', self.state, self.data, self.predict)
        self.assertTrue(answer['urgent'])
        self.assertTrue(self.state['urgent'])
        self.assertIn('emergency assessment', answer['text'])
        self.predict.assert_not_called()

    def test_negated_headache_after_sex_is_not_recorded(self):
        respond('no headache after sex', self.state, self.data, self.predict)
        self.assertNotIn('headache', self.state['symptoms'])

    def test_low_confidence_has_no_medicine_list(self):
        self.predict.return_value[0]['probability']=.3
        answer=respond('cough and fever for two days',self.state,self.data,self.predict)
        self.assertTrue(answer['uncertain'])
        self.assertIn('You mentioned **cough and high fever**', answer['text'])
        self.assertIn('do you also have', answer['text'])
        self.assertNotIn('model', answer['text'].lower())
        self.assertNotIn('Educational medicine information',answer['text'])

    def test_uncertain_reply_uses_plain_language_without_a_followup(self):
        self.predict.return_value[0]['probability'] = .3
        self.state['questions_asked'] = ['cough', 'high_fever', 'headache']
        answer = respond('nausea and stomach pain', self.state, self.data, self.predict)
        self.assertTrue(answer['uncertain'])
        self.assertIn('nausea and stomach pain', answer['text'])
        self.assertNotIn('model', answer['text'].lower())
        self.assertNotIn('medicine information from', answer['text'].lower())

    def test_context_withholds_medicine_names(self):
        answer=respond('cough, fever and runny nose for two days',self.state,self.data,self.predict,{'allergies':'penicillin'})
        self.assertIn('withheld',answer['text'])
        self.assertNotIn('Educational medicine information',answer['text'])

    def test_chat_allergy_is_retained(self):
        respond('I am allergic to penicillin',self.state,self.data,self.predict)
        answer=respond('cough, fever and runny nose for two days',self.state,self.data,self.predict)
        self.assertIn('withheld',answer['text'])

    def test_context_does_not_leak_between_chats(self):
        respond('I am pregnant',self.state,self.data,self.predict)
        other=new_state()
        self.assertFalse(other['context'])
        self.assertFalse(other['symptoms'])

    def test_common_symptoms_require_context_before_naming_condition(self):
        first = respond('headache and fatigue', self.state, self.data, self.predict)
        self.assertTrue(first['uncertain'])
        self.assertIn('When did they start', first['text'])
        self.assertNotIn('Common Cold', first['text'])
        second = respond('two days', self.state, self.data, self.predict)
        self.assertEqual(self.state['details']['duration'], 'two days')
        self.assertNotIn('Common Cold', second['text'])
        self.assertNotIn('Educational medicine information', second['text'])

    def test_partial_yes_completes_headache_followup(self):
        self.state['symptoms'] = ['headache', 'fatigue']
        self.state['details']['duration'] = '1 week before'
        self.state['pending'] = 'loss_of_appetite'
        self.state['questions_asked'] = ['loss_of_appetite']
        answer = respond('little bit', self.state, self.data, self.predict)
        self.assertIn('loss_of_appetite', self.state['symptoms'])
        self.assertIsNone(self.state['pending'])
        self.assertTrue(answer['uncertain'])
        self.assertIn('a little loss of appetite', answer['text'])
        self.assertIn('about a week', answer['text'])
        self.assertNotIn('do you also have', answer['text'])
        self.assertNotIn('Common Cold', answer['text'])

    def test_family_history_is_not_asked_as_symptom(self):
        predictions = [{'condition': 'Heart attack', 'probability': .6}]
        self.assertNotEqual(next_question(self.state, predictions, self.data), 'family_history')

    def test_two_followups_end_with_guidance_not_a_named_condition(self):
        self.predict.return_value = [
            {'condition': 'Hepatitis C', 'probability': .75},
            {'condition': 'Chicken pox', 'probability': .12},
            {'condition': 'Hypertension', 'probability': .03},
        ]
        respond('I have a headache and feel tired', self.state, self.data, self.predict)
        respond('1 week before', self.state, self.data, self.predict)
        respond('no', self.state, self.data, self.predict)
        answer = respond('little bit', self.state, self.data, self.predict)
        self.assertTrue(answer['uncertain'])
        self.assertIn('about a week', answer['text'])
        self.assertIn('a little nausea', answer['text'])
        self.assertNotIn('Hepatitis C', answer['text'])
        self.assertIsNone(self.state['pending'])

    def test_two_denied_followups_show_uncertain_results(self):
        respond('I have a headache and feel tired', self.state, self.data, self.predict)
        respond('around 1 week', self.state, self.data, self.predict)
        respond('no', self.state, self.data, self.predict)
        answer = respond('no', self.state, self.data, self.predict)
        self.assertTrue(answer['result_ready'])
        self.assertTrue(answer['uncertain'])
        self.assertTrue(answer['medicine_withheld'])
        self.assertEqual(answer['predictions'], self.state['last_predictions'])
        self.assertIn('available dataset matches', answer['text'])
        self.assertIsNone(self.state['pending'])
        self.assertIsNone(self.state['last_condition'])

    def test_show_results_uses_same_uncertain_result_path(self):
        respond('headache and fatigue', self.state, self.data, self.predict)
        answer = respond('show results', self.state, self.data, self.predict)
        self.assertTrue(answer['result_ready'])
        self.assertEqual(answer['predictions'], self.predict.return_value)
        self.assertTrue(answer['medicine_withheld'])
        self.assertNotIn('condition', answer)

    def test_affirming_urgent_followup_triggers_urgent_response(self):
        respond('cough', self.state, self.data, self.predict)
        self.state['pending'] = 'chest_pain'
        answer = respond('yes', self.state, self.data, self.predict)
        self.assertTrue(answer['urgent'])
        self.assertTrue(self.state['urgent'])
        self.assertNotIn('condition', answer)

    def test_not_sure_does_not_trap_user_in_duration_question(self):
        respond('headache and fatigue', self.state, self.data, self.predict)
        answer = respond('not sure', self.state, self.data, self.predict)
        self.assertIsNone(self.state['pending_detail'])
        self.assertIn('do you also have', answer['text'])
        self.assertNotIn('Common Cold', answer['text'])

    def test_pregnancy_ibuprofen_question_uses_curated_safety_source(self):
        answer = respond('Can I take ibuprofen while pregnant?', self.state, self.data, self.predict)
        self.assertEqual(answer['intent'], 'medicine_safety')
        self.assertTrue(answer['medicine_withheld'])
        self.assertIn('20 weeks', answer['text'])
        self.assertEqual(answer['source_details'][0]['publisher'], 'FDA')
        self.predict.assert_not_called()

    def test_split_has_no_pattern_overlap(self):
        split=json.loads((ARTIFACT_DIR/'splits.json').read_text())
        a,b,c=[set(split[k]) for k in ['train','validation','test']]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(len(a|b|c),self.data['manifest']['unique_rows'])
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
        self.assertEqual(m['split_sizes'],{'train':2918,'validation':61,'test':61})
        self.assertEqual(len(self.data['by_condition']),51)


if __name__=='__main__':unittest.main()
