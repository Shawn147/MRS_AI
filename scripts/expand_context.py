"""Add authored follow-up examples with fixed splits, not patient observations."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data/context_intents.json'
# Each line is a separately authored utterance. Existing evaluation examples stay fixed.
PHRASES = {
'summary': '''Give me the main points from our chat
Can I see my symptom list so far
Write down everything I have reported
Recap the details I have shared with you
What information have you saved about me
Put my complaints together in one summary
Show the symptoms I said I do not have
Include how long I have felt unwell in a recap
Can you organize my answers into a short note
I need a summary to discuss with my doctor
Review what we have discussed about my health
Make a brief record of my current complaints
What have you learned from my answers
Collect my symptoms and timing in one place
Please outline the information from this conversation
Can you remind me of my earlier answers
Give me an overview of the symptoms I reported
Write a short recap before we continue
Which symptoms have I already told you about
Summarise our conversation so far''',
'explain_match': '''Why is that illness on the list
What led to the match you showed me
How does the suggested condition relate to my symptoms
Please walk me through the previous finding
What does the last condition match mean
Tell me why this possible cause appeared
Which of my symptoms fit that suggestion
Explain the reasoning behind the displayed possibility
I do not understand the result you gave earlier
Help me interpret the previous match
Does that result confirm what I have
Why did you list that particular illness
What is the meaning of the condition you mentioned
Explain what the earlier result can and cannot tell me
What do my symptoms share with the previous condition
Can you clarify the last possibility you showed
I want to understand that suggested match better
How should I interpret the earlier condition label
Tell me more about the condition from the last reply
What made you suggest the previous possible cause''',
'medicine_question': '''Are there medicine references for the last match
Which medicines does the dataset list for this condition
Show me the treatment names in your source
Can you display medicine information for that possibility
What drug names are mentioned in the educational record
Is there any medication information in the previous result
Tell me about medicines associated with the match
Which medicines are in the condition entry
Can I see the listed medicine names
What does your dataset say about medicine for it
Do you have drug information about this illness
Show the educational medicine references again
I want information on the medicines in that record
Which tablets are associated with the earlier match
Can you list the medications from the source dataset
What medication references belong to this result
Please show medicine information for my previous match
Are any treatments named in that condition record
Tell me which medicines your source mentions
Can you retrieve the drug list for the last condition''',
'dose_question': '''What dose of that tablet should I use
How many pills can I take in a day
Tell me the correct number of milligrams
How often do I need to take the medicine
Can you calculate the dose based on my weight
What is the dosing schedule for this drug
Should I take one tablet or two
How many times per day should I take it
What amount of medication is appropriate for a child
Can you tell me a safe dose during pregnancy
How much of the syrup should I give
What is the maximum daily dose
Is it time to take the next pill
What interval should I leave between doses
How many millilitres of medicine should I use
What dosage applies to someone my age
Should I increase the dose if it does not help
Can I take another tablet this evening
Tell me how much medicine to take each time
Can you work out the dosing frequency''',
'duration': '''It has been happening for nearly a fortnight
This started sometime last weekend
I have felt this way since Tuesday evening
The symptoms began about five hours ago
It has lasted a little over a month
I first noticed it yesterday after lunch
I have had this problem for several weeks
It began roughly ten days back
This has been going on since early September
I started feeling unwell late last night
The problem has persisted for three mornings
I have been dealing with this for half a year
It has continued since the weekend before last
I noticed the symptoms about forty minutes ago
This began during the afternoon yesterday
I cannot remember the exact date but about two weeks
It has happened on and off for a few months
The symptoms started four days before today
I have felt like this since waking up today
It first began around the beginning of last month''',
'severity': '''The discomfort is barely noticeable
I would rate the intensity about three out of ten
It is quite intense but I can still move around
The level of pain is roughly seven out of ten
It feels moderate rather than extreme
The symptoms are mild most of the time
I can manage it but the discomfort is significant
The pain is a two on a ten point scale
It feels severe enough to interrupt my work
It is uncomfortable but only a little
I would describe the symptoms as fairly mild
The intensity is about six on a scale of ten
It is stronger than a minor annoyance
I would call this moderate discomfort
The symptom intensity is around four out of ten
It feels very mild when I am resting
The discomfort is substantial throughout the day
It is not severe but it is noticeable
I feel a moderate level of discomfort
I would rate the symptom severity as eight out of ten''',
'improving': '''I feel a little better than yesterday
The symptoms are beginning to settle down
It seems to be easing over time
I am starting to recover now
Things are slowly getting better
The discomfort is less than it was before
I have noticed some improvement this afternoon
I feel more like myself today
The problem is gradually fading
It has eased since we last spoke
I am doing better compared with earlier
The symptoms are less noticeable now
I think I am recovering a bit
It is improving with each passing day
The discomfort is starting to subside
I feel much better this morning
The symptoms seem to be clearing up
I am feeling some relief now
It has become easier to manage today
I notice it less often than yesterday''',
'worsening': '''I am feeling worse than I did earlier
The symptoms have become more difficult to manage
It is not getting any better
Things seem to be deteriorating today
The discomfort keeps increasing
I have noticed the problem getting worse
It is more intense than it was yesterday
I am struggling more with it now
There has been no improvement at all
It has worsened since we last spoke
I feel increasingly unwell
The symptoms are becoming more frequent
It seems to be progressing rather than easing
I am doing worse this morning
The problem has continued without any relief
It is harder to cope with than before
I feel worse with each passing day
The symptoms are getting stronger
I have not recovered as I expected
The discomfort has escalated today''',
'other': '''What is the capital city of Nepal
Can you write a birthday invitation
Tell me a joke about computers
What will the weather be like tomorrow
Help me choose a name for my shop
How do I fix a bicycle tyre
Write a poem about the sea
Who won the football game
Can you solve this algebra problem
Suggest a movie for this weekend
How do I cook lentils
What is the tallest mountain
Explain how solar panels work
Help me draft a job application
Translate a sentence into French
Can you plan a weekend trip
What is a good password manager
Tell me about ancient history
How do I change my phone wallpaper
Suggest some ideas for a school project''',
'greeting': '''Hello I am ready for guidance
Good morning to you
Good evening assistant
Hi I would like to start a conversation
Hello can we talk
Hey are you available
Greetings I am here for some help
Hi there how are you
Good afternoon
Hello I am new here
Hey I have just opened this chat
Hello I would like some guidance
Hi can we get started
Greetings assistant
Good day to you
Hey there I am ready to talk
Hi I need a little help today
Hello is anyone here
Morning can you help me
Hi this is my first time using the chat
Hello I want to begin
Hey can we start now
Good evening I have a question
Hi nice to meet you
Greetings can you assist me
Hello I am checking in
Hey I am back again
Good afternoon can we chat
Hi I would like to speak with you
Hello let us get started''',
'gratitude': '''Thank you for explaining that
Thanks for your help today
I appreciate the information
That was helpful thank you
Many thanks for clarifying
Thank you I understand now
I am grateful for your help
Thanks that makes sense
I appreciate you taking the time
Thank you for the clear explanation
Thanks for listening to me
Much appreciated
Thank you that answers my question
I appreciate your guidance
Thanks for the useful details
That helped a lot thank you
Thank you for walking me through it
Thanks I will keep that in mind
I am thankful for the explanation
Thank you for clearing that up
Thanks for answering my follow-up
I appreciate the summary you gave
Thank you for your support
Thanks for making it easier to understand
That is useful information thanks
I appreciate your response
Thank you for helping me understand
Thanks for explaining the limitations
Thank you for the information you shared
I am glad you clarified it thanks''',
'report_help': '''How do I upload my medical report
Can I send a file in this chat
Where do I attach my lab results
How can I share a blood test document
Can you explain a report that I upload
Where is the file upload option
How do I remove a file before sending
Can I send a report without typing a message
I want to attach a hospital document
How do I add an image of my test results
Can I ask a question along with a file
How do I share my medical file with you
What button do I use to upload a report
Can I attach a scanned report here
How do I cancel a selected attachment
I would like help uploading my results
Where should I put my PDF report
Can I send a photograph of a report
Explain how to attach a file to my message
How can I add a medical document
Is there a way to upload a test report
Can I remove an attachment from the input
I want to send my report and a question together
How do I select a file to share
Can you read a document after I send it
Where do uploaded files appear before sending
How do I discard the file I picked
Can I upload a report on its own
Tell me how the attachment option works
I need help sharing my laboratory report'''
}

def main():
    dataset = json.loads(DATA.read_text())
    rows = [r for r in dataset['examples'] if not r['id'].startswith('expanded_')]
    # This original training utterance was a fallback before greeting became a supported intent.
    for row in rows:
        if row['id'] == 'other_06' and row['split'] == 'train':
            row.update(intent='greeting', original_intent='other', relabel_reason='Greeting is now a supported route')
        if row['id'] == 'other_07' and row['split'] == 'train':
            row.update(intent='gratitude', original_intent='other', relabel_reason='Gratitude is now a supported route')
    existing = {r['text'].strip().lower() for r in rows}
    new_intents = {'greeting', 'gratitude', 'report_help'}
    for intent, phrases in PHRASES.items():
        for number, text in enumerate(phrases.splitlines(), 1):
            if text.strip().lower() in existing:
                raise ValueError('Duplicate authored utterance: ' + text)
            existing.add(text.strip().lower())
            part = ('train' if number <= 18 else 'validation' if number <= 24 else 'test') if intent in new_intents else 'train'
            rows.append(dict(id=f'expanded_{intent}_{number:02}', text=text, intent=intent, split=part,
                             data_type='authored_illustrative_language', clinically_reviewed=False))
    dataset['examples'] = rows
    dataset['provenance'] = '450 authored illustrative language examples, not patient cases. Original held-out examples preserved; new intents have separate authored holdouts. Not clinical validation.'
    DATA.write_text(json.dumps(dataset, indent=2, ensure_ascii=False)+'\n')
    print(len(rows), len({r['intent'] for r in rows}))

if __name__ == '__main__':
    main()
