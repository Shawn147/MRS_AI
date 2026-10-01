# MRS AI BOT

Streamlit chatbot for health information and symptom guidance. A MiniLM classifier
supports symptom follow-ups; Qwen answers informational questions with optional
source-linked reference notes. Condition matches are not clinically validated diagnoses.

```bash
source .venv/bin/activate
streamlit run app.py
```

On Windows: `.venv\Scripts\activate`. No API key is needed for local Ollama use.
Chat history is saved in the visitor's browser local storage. With the hosted Groq configuration, general
questions and contextual descriptions are sent to Groq; avoid entering identifying details.

## Live deployment and Qwen

The default chat endpoint is `http://127.0.0.1:11434/api/chat`. That works when
Streamlit and Ollama run on the same computer. On Streamlit Community Cloud,
`127.0.0.1` refers to the cloud container, so a model downloaded to a personal
Mac is not available to the public app.

For a free, rate-limited live demo, [Groq's Free plan](https://console.groq.com/docs/rate-limits)
currently serves [Qwen](https://console.groq.com/docs/models).
Create a [Groq API key](https://console.groq.com/keys) and set these root-level
Streamlit app secrets:

```toml
MRS_LLM_BACKEND = "openai"
MRS_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
MRS_LLM_MODEL = "qwen/qwen3.8-27b"
MRS_LLM_API_KEY = "YOUR_GROQ_API_KEY"
```

Save the secrets, restart the app, and test a general question. The free plan has
request and token limits; check Groq's current model list and rate-limit page
before relying on it for public traffic. Keep the key in Streamlit secrets,
never in GitHub. This hosted model differs from the locally installed `qwen3:4b`.

To answer general questions in a cloud deployment, run Qwen on a hosted inference
service with an OpenAI-compatible chat-completions endpoint. One option is
[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index),
which currently serves [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
Create a fine-grained Hugging Face token with **Make calls to Inference Providers**
permission. Add these **root-level secrets** in the Streamlit app's settings,
replacing only the token placeholder:

```toml
MRS_LLM_BACKEND = "openai"
MRS_LLM_URL = "https://router.huggingface.co/v1/chat/completions"
MRS_LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MRS_LLM_API_KEY = "hf_YOUR_TOKEN"
```

Save the secrets in the deployed app's settings, then restart the app and test a
general question. Inference Providers may require available credits or billing;
check the account's usage before making the app public. Model availability can
change, so select another supported chat model if this model is no longer served.

The endpoint must use HTTPS. The provider receives the general question and any
symptom labels included as context. Review its data handling before enabling it
for real users. Do not commit `.streamlit/secrets.toml`; it is git-ignored.
The app can also connect to a reachable hosted Ollama endpoint by setting
`MRS_LLM_BACKEND = "ollama"`, `MRS_LLM_URL` to its full `/api/chat` URL,
`MRS_LLM_MODEL` to the hosted model tag, and `MRS_LLM_API_KEY` if required.
Without a hosted endpoint, the live app cannot generate general answers. It will
show a setup message while local Ollama continues to work for local development.

## Layout

```
app.py                 Streamlit entry
src/                   chat, data, models, UI
data/                  JSON knowledge base
scripts/               import CSVs, download MiniLM, train
artifacts/chatbot/     saved models and metrics
references/supervisor/ original CSVs and documents
tests/
```

The main experience is a styled Streamlit conversation with recent chats, optional **Health context**, **Conversation history**, and **About & sources**. Dataset Preview and Model Evaluation remain available under **About & sources → Technical details & project resources**.

## Rebuild

Tested on Python 3.9.6 with the pinned packages in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_json.py
python scripts/download_model.py
python scripts/train_chatbot.py
streamlit run app.py
```

The one-time download fetches public `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Training and inference are local. Only locally generated `baseline.joblib` should be loaded.

Restart Streamlit after Python edits; file watching is off to avoid PyTorch watcher issues.

## Data

Source: https://github.com/dr-mushtaq/Medicine-Recommendation-System  
Pinned revision: `aa6b145d4838ac1dd6fd21b4db7a43c3098e19f9`

`scripts/prepare_json.py` reads `references/supervisor/dataset` and writes `data/*.json`. The app reads JSON only.

- 4,920 original rows → 304 original patterns; 4,616 repeats removed
- 2,736 explicitly illustrative additions → **3,040 unique patterns (10× the original)**
- 7,656 total input rows, 51 conditions, 156 symptom features and 17 reference records
- Original rows are preserved; generated examples are not counted as hospital cases
- Duplicate `fluid_overload` columns merged by OR; `Peptic ulcer diseae` corrected so tables join
- Severity weights stored as metadata, not used as triage
- Medicine/diet/exercise/precaution rows are unreviewed educational source data

`data/input_corrections.json` is spelling help only (`dirhea` → `diarrhea`). Editing it does not require retraining.

`data/regional/` adds **3,900 distinct public clinical observations** from hospital
and primary-care studies in Pakistan, Bangladesh and India (4,169 source rows,
269 repeats removed). Sources include Aga Khan University Hospital primary-care
sites, Faisalabad Institute of Cardiology/Allied Hospital, Sylhet Diabetes Hospital,
and an Indian CKD cohort reported as Apollo Hospitals. Licences, original study
targets, variable definitions, source links and checksums accompany the prepared
JSONL files. See [the regional data guide](data/regional/README.md).

These cohorts are research-only because their outcomes and features differ from
the general symptom-classification task. They are not relabelled as disease
examples. The classifier expansion uses labelled illustrative patterns separately;
Dataset Preview shows hospital-data counts and downloads separately.
Run `python3 scripts/collect_regional_data.py --download` to reproduce the pinned
imports and `python3 scripts/audit_regional_data.py` to validate the collection.

## Models

Fixed original holdouts: **2,918 train, 61 validation, 61 test**. All generated
subsets stay in training, with their original training parent or source profile.
Each label and training family receives equal total training weight. Head
regularization and transformer checkpoint selection use the original validation
set; the test set is untouched during selection. Ten added labels have no
independent evaluation. See [the expansion guide](data/EXPANSION.md).

| Model | Correct / test | Accuracy |
|---|---:|---:|
| TF-IDF Logistic Regression | 61/61 | 100% |
| Frozen MiniLM + learned head | 60/61 | 98.36% |
| Fine-tuned MiniLM | 60/61 | 98.36% |

These repetitive educational patterns do not establish clinical generalization. Class probabilities are not calibrated.

Three separate research models also benchmark original published hospital tasks.
They use 950 distinct observations across Pakistan heart failure, Bangladesh
diabetes and India CKD cohorts. Exact duplicates are removed before stratified
60/20/20 splits. Imputation, scaling and categorical encoding are fitted on train
only; model selection uses validation balanced accuracy. Test results are
evaluated once. Follow-up `time` is excluded from the heart-failure features.

```bash
.venv/bin/python scripts/benchmark_hospital_models.py
```

`artifacts/research/metrics.json` preserves source checksums, split IDs, validation
choices, per-class metrics, ROC AUC, majority baselines and limitations. The Model
Evaluation page displays these results and offers the full JSON download.
These are internal single-source research benchmarks, not external clinical
validation. Their models are separate from chatbot symptom inference. The infant
cohort remains excluded until coded features and timing have been reviewed.

## Conversation

Explicit aliases detect symptoms. A small negation rule handles phrases such as “no cough”. Unknown-only input is rejected. Emergency keywords pause medicine output. Replies are templates over JSON, not a generative chatbot.

Patient context in the sidebar withholds medicine names rather than pretending the source can check suitability.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Data quality and reference expansion

Run `python scripts/audit_data.py` to validate dataset relationships, duplicate
patterns, source-row coverage, required condition fields, manifest counts and
reference-only status. It writes `data/quality_report.json` and exits nonzero on
validation errors.

`data/symptom_metadata.json` repairs three missing severity joins using explicit
source-name mappings and the original CSV checksum. The importer regenerates this
file; `load_data()` applies it after checking the original JSON hashes. Original
symptom IDs, training patterns and model fingerprints are preserved. Severity is
still provenance metadata, not a triage score. Dataset Preview and downloads show
the corrected values.

`data/reference_conditions.json` contains source-linked records for flu, sinusitis,
COVID-19, headaches, sore throat, hepatitis B and NSAID safety in pregnancy,
summarized from linked NHS and FDA pages. Each record contains symptom terms,
care notes, help-seeking notes, source review dates and review status. These are
manually maintained reference records, visible and downloadable in Dataset
Preview; the CSV importer does not overwrite them. COVID-19's source page has a
past review-due date, so recheck current guidance before clinical use.

Optional local semantic search uses `abhinand/MedEmbed-small-v0.1` at pinned revision
`40a5850d046cfdb56154e332b4d7099b63e8d50e`. Run
`python scripts/download_medembed.py` once, then call
`search_references(question, data['reference_conditions'])` from
`src.reference_search`. It ranks passages from the source-linked summaries
and returns their URLs; similarity is not diagnostic confidence. It does not
extend the condition classifier.

For informational questions and personal health descriptions with contextual details,
the chat calls local Ollama `qwen3:4b` by default, or the hosted provider configured
above on the live app. Directly named reference topics supply their complete curated
notes, including self-care and when-to-seek-help guidance. Run Ollama locally and
download the model with `ollama pull qwen3:4b` when using the local configuration.
Questions about other topics receive a clearly unverified general answer, with no
invented source link. The symptom flow, emergency handling and medicine/dose guards
remain separate. These generated answers have not been clinically validated.

Coverage is **51 classifier labels and 17 source-linked reference records**.
Ten new condition inventories and 25 symptom terms are summarized from NHS and
Aga Khan University Hospital Pakistan sources. Generated symptom subsets are
explicitly illustrative and not observed patient cases. The ten new labels have
no independent test cohort and cannot establish clinical accuracy. Their
medicine mappings are empty and medicine output is withheld. Existing medicine
mappings remain unreviewed. Some legacy classes still have only five patterns;
augmentation cannot create independent clinical evidence.

## Interface

The interface follows the supplied Stitch design using native Streamlit components
and local CSS in `assets/style.css`; no separate frontend or JavaScript framework
is needed. The local SVG logo is in `assets/logo.svg`.

Starter prompts and yes/no follow-up buttons submit real chatbot messages.
The More options control has been removed; summary and explanation requests can
still be typed in chat. Voice to text uses the microphone immediately left of
Send. Recording shows a timer, live transcript, Stop and Cancel. Stopping adds an
editable transcript inside the composer; its close button removes it. The user
can send a transcript alone or with typed text and files. Nothing is sent to the
chat model until Send is pressed. Recordings stop automatically after 60 seconds;
MRS AI does not store audio.

Voice input requires microphone permission and HTTPS or localhost. Browser
support varies, and browsers may use an external speech service; see the
[Web Speech API documentation](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition).
Permission failures and unsupported browsers show a clear typed-input fallback.
Run `node tests/test_voice_component.cjs` for simulated recording lifecycle checks
that never access a microphone.

General answers use a professional plain-language prompt. If generation fails,
named topics can use the bundled source summaries, and readable reports can show
verbatim lab values with filenames/page references. These fallbacks do not
interpret missing data or invent ranges. A full report explanation still needs
the configured local/hosted model. Account-recovery and credential filenames are
rejected as medical uploads, including legacy attachments before provider calls.
The latest user message has an **Edit last prompt** control. Saving an edit restores
conversation state from before that turn, regenerates its reply, and replaces its
analytics event. Chats made before turn snapshots were added rebuild their state
from earlier user messages when edited.
Headaches reported during or after sex receive a specific safety follow-up rather
than an unrelated symptom question; sudden severe onset directs users to emergency
assessment. Other symptom reports with timing or activity context can use Qwen.
Condition cards show captured symptoms and keep medicine information collapsed.
Health context is scoped to each conversation and also withholds medicine details
in earlier cards. Urgent notices persist through subsequent messages.

The design's fictional patient details, clinical verification claims and sample
diagnoses are not included. Predictions come from the existing saved model.
Model scores and model selection are not shown in the main conversation.

Use `streamlit run app.py` to start the interface. Restart the process after Python
changes because file watching is disabled. Conversations are saved in browser local
storage and can be cleared or downloaded from Conversation history. Browser storage
may be unavailable in private browsing or restrictive settings.

## Analytics

The **Analytics** page reports anonymous sessions, conversations, assistant replies,
urgent conversations, common condition matches and symptoms, medicine references
offered, failed general-answer calls, and the most active conversations. Filter by 7 days, 30 days or all time;
export medicine counts as CSV or the aggregate summary as JSON.

Events persist locally in `artifacts/analytics/events.sqlite3` (excluded from git).
This means local storage on the **app server**, not browser `localStorage`. Streamlit
Community Cloud may reset this file when the app restarts; use a durable database
for long-term metrics. The dashboard is enabled, but has no authentication for a
public deployment.
Each assistant turn has a unique event ID, so Streamlit reruns do not add counts.
No message text, patient names or profile values are stored. Session identifiers
are random and reset with the browser session: they do **not** identify unique
patients. Medicine counts measure lists offered, not prescriptions, use,
effectiveness or whether a collapsed section was opened. Historical counts do not
change when later health context hides a previous medicine card. Earlier chats
are not automatically imported. Dates use UTC. Set `MRS_ANALYTICS_DB` to override
the database path; tests use isolated temporary databases. Keep this local dashboard
private; authentication for shared or public deployments is not implemented.

## Context-aware follow-ups

`data/context_intents.json` contains **450 authored illustrative language examples
across 12 intents**: summary, match explanation, medicine information, dose
questions, duration, severity, improving, worsening, other, greeting, gratitude
and report-upload help. They are not patient cases. Two original training
utterances were relabelled to the newly supported greeting and gratitude routes;
original holdout labels were preserved.

```bash
.venv/bin/python scripts/expand_context.py
.venv/bin/python scripts/train_context.py
```

A separate logistic-regression intent head uses frozen MiniLM embeddings.
Fixed splits: **342 train / 54 validation / 54 test**. Regularization is selected
on validation accuracy, with correct accepted routes minus incorrect routes as
the tie-breaker. Test accuracy is **52/54 (96.3%)**. Fixed score and margin gates
accepted 45/54 examples, 44 of them correct.

These results are language-routing checks, not clinical validation or evidence of
real-world generalization. Scores are not calibrated. Rejected intents fall back
to the existing symptom dialogue.

Follow-ups retain symptom lists, denials, pending questions, duration, severity and
reported improvement/worsening. “Summarize my symptoms” and “Explain the previous
result” use that conversation's state. Duration and severity are remembered for
replies, but are **not** inputs to the condition classifier. Medicine/dose requests
have explicit guards; emergency handling takes precedence over intent routing.
Replies remain constrained templates; no generative medical model was added.
Training metrics and split IDs are in `artifacts/context/metrics.json`.


### Medical report attachments

Drop PDF, TXT, PNG or JPEG reports directly onto the chat input, or use its **+** menu.
The input highlights while files are dragged over it. Dropped files use the same
validation as selected uploads; dropping a file does not submit a chat message.
Selected files appear automatically as square cards inside the composer. Use **×**
to remove a draft, then press the send arrow with optional message text. Sending a
file alone asks for a report explanation. Text-only messages work normally. Sent
report text remains available for follow-up questions; drafts are not sent until
you submit them. The three most recently sent reports are retained as context.
Sent reports remain visible as file cards. Editing or retrying the last prompt
preserves that turn's attached reports and their readable text.
In the edit view, use **×** on a file card to exclude it from the regenerated
message. **Cancel edit** keeps the original attachments unchanged.

Limits: three files per message, 10 MB per file, 30 PDF pages and 18,000 extracted
characters per report. PDFs must contain readable text on every page; scanned
pages should be uploaded as clear photos instead. Encrypted and unreadable files
are rejected rather than silently summarized.

Photo OCR uses Apple Vision on macOS (requires the Swift developer tools), or
`tesseract` on Linux. `packages.txt` installs Tesseract on Streamlit Community
Cloud. OCR can misread values; the assistant asks users to verify unclear numbers.

Readable report contents are sent to the configured general-answer AI provider
when a question is submitted, and stored with browser conversation history.
Original uploaded files are not deliberately saved to the project. OCR temporary
files are deleted after extraction. Reports are treated as untrusted source data,
not instructions, and do not feed the symptom classifier. Replies explain the
report with filename/page references; they do not diagnose or prescribe.
