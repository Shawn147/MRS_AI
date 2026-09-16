import unittest
from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
    def test_chat_navigation_and_isolated_history(self):
        at=AppTest.from_file('app.py',default_timeout=60).run()
        self.assertFalse(at.exception)
        at.chat_input[0].set_value('I have a runny nose, sneezing and cough').run()
        self.assertFalse(at.exception)
        first=at.session_state['active_chat']
        messages=at.session_state['chats'][first]['messages']
        self.assertEqual(len(messages),2)
        self.assertIn('top possible condition',messages[-1]['content'])
        self.assertEqual(messages[-1]['model'],'transformer')
        for page in ['Dataset Preview','Model Evaluation','History','About','Chat']:
            at.radio(key='navigation').set_value(page).run()
            self.assertFalse(at.exception,page)
        next(b for b in at.button if b.label=='＋ New chat').click().run()
        self.assertFalse(at.exception)
        second=at.session_state['active_chat']
        self.assertNotEqual(first,second)
        self.assertFalse(at.session_state['chats'][second]['state']['symptoms'])
        at.chat_input[0].set_value('chest pain').run()
        self.assertFalse(at.exception)
        self.assertTrue(at.session_state['chats'][second]['state']['urgent'])
        self.assertFalse(at.session_state['chats'][first]['state']['urgent'])


if __name__=='__main__':unittest.main()
