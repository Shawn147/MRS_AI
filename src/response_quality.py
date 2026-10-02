"""Grounded replies when the optional generative provider cannot answer."""
import re
from src.medical_files import is_account_file

LAB_TERMS = re.compile(r'\b(?:hemoglobin|haemoglobin|hgb|hb|glucose|hba1c|creatinine|urea|egfr|cholesterol|'
                       r'triglycerides|hdl|ldl|platelets?|wbc|rbc|white blood|red blood|sodium|potassium|'
                       r'calcium|bilirubin|alt|ast|tsh|thyroxine|ferritin|vitamin|hematocrit|haematocrit|mcv|mch|crp)\b', re.I)


def protected_attachment_reply(reports):
    if reports and all(is_account_file(r.get('name', '')) for r in reports):
        return {'text': 'This attachment appears to contain account recovery information rather than a medical report. '
                'Please choose a test result, medical report or prescription. You can also describe your symptoms here.',
                'predictions': [], 'intent': 'attachment_not_medical', 'medicine_withheld': True}


def fallback_reply(records, reports, reason='connection', retryable=True):
    reports = [r for r in reports if not is_account_file(r.get('name', ''))]
    if reports:
        sections = ['I can read your uploaded text. Here are the report details available while the full explanation is unavailable:']
        found = False
        for report in reports:
            lines, page = [], None
            for line in report.get('text', '').splitlines():
                marker = re.fullmatch(r'\[Page (\d+)\]', line.strip())
                if marker:
                    page = marker.group(1)
                if LAB_TERMS.search(line) and re.search(r'\d', line) and len(line.strip()) <= 180:
                    # Verbatim report values only. No reference ranges, units or conclusions are invented.
                    plain = re.sub(r'[`*<>\[\]]', '', line.strip())
                    lines.append('• ' + plain + (f' (page {page})' if page else ''))
                if len(lines) == 6:
                    break
            if lines:
                found = True
                sections.append('**' + re.sub(r'[`*<>\[\]]', '', report['name']) + '**\n\n' + '\n\n'.join(lines))
        if found:
            sections.append('Here are the values I could read from your file. '
                            'Please check them against the original report. Select **Retry answer** for a detailed explanation, '
                            'or tell me which result you would like to discuss.')
        else:
            sections = ['Your file is available in this conversation, but I couldn’t complete its explanation right now. '
                        'Please select **Retry answer**, or paste the test name, result, unit and reference range you want to understand.']
        return {'text': '\n\n'.join(sections), 'predictions': [], 'intent': 'report_summary_fallback',
                'retryable': True, 'medicine_withheld': True, 'generation_status': reason}
    if records:
        sections = ['Here is the source-backed information I can share:']
        for record in records[:2]:
            sections.append('**' + record['name'] + '**\n\n' + record['description'])
            if record.get('care_notes'):
                sections.append('**General care**\n\n' + ' '.join(record['care_notes']))
            if record.get('seek_help_notes'):
                title = 'Label safety notes' if record.get('data_type') == 'drug_label_excerpt' else 'When to seek help'
                sections.append('**' + title + '**\n\n' + ' '.join(record['seek_help_notes']))
            if record.get('data_type') == 'drug_label_excerpt':
                sections.append('This is a brief extract from the US product label. Read the full label for complete instructions and safety information.')
        sources = [s for r in records[:2] for s in r['sources']]
        return {'text': '\n\n'.join(sections), 'predictions': [], 'intent': 'reference_fallback',
                'sources': list(dict.fromkeys(s['url'] for s in sources)), 'source_details': sources,
                'generation_status': reason, 'medicine_withheld': True}
    lead = ('I couldn’t complete your answer at the moment. Please try again shortly.' if reason == 'busy'
            else 'I couldn’t complete your answer right now.')
    action = (' Please retry using **Retry answer**. Your message is still here.' if retryable
              else ' General answers are temporarily unavailable. You can describe your symptoms here and continue with guided questions.')
    return {'text': lead + action, 'predictions': [], 'intent': 'general_question_unavailable',
            'retryable': retryable, 'generation_status': reason}
