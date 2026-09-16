# MRS AI BOT

Local Streamlit chatbot: symptoms → MiniLM classifier → educational medicine notes from the supervisor dataset.

```bash
source .venv/bin/activate
streamlit run app.py
```

On Windows: `.venv\Scripts\activate`. No API key is needed. Chat history stays in the browser session.

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

Sidebar pages: **Chat**, **Dataset Preview**, **Model Evaluation**, **History**, **About**.

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

- 4,920 source rows → 304 unique symptom patterns (4,616 repeats removed before splitting)
- 41 conditions, 131 symptom features
- Duplicate `fluid_overload` columns merged by OR; `Peptic ulcer diseae` corrected so tables join
- Severity weights stored as metadata, not used as triage
- Medicine/diet/exercise/precaution rows are unreviewed educational source data

`data/input_corrections.json` is spelling help only (`dirhea` → `diarrhea`). Editing it does not require retraining.

## Models

60/20/20 stratified split: 182 train, 61 validation, 61 test.

| Model | Correct / test | Accuracy |
|---|---:|---:|
| TF-IDF Logistic Regression | 60/61 | 98.36% |
| Frozen MiniLM + learned head | 61/61 | 100% |
| Fine-tuned MiniLM | 61/61 | 100% |

These repetitive educational patterns do not establish clinical generalization. Class probabilities are not calibrated.

## Conversation

Explicit aliases detect symptoms. A small negation rule handles phrases such as “no cough”. Unknown-only input is rejected. Emergency keywords pause medicine output. Replies are templates over JSON, not a generative chatbot.

Patient context in the sidebar withholds medicine names rather than pretending the source can check suitability.

## Tests

```bash
python -m unittest discover -s tests -v
```
