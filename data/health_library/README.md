# Expanded health reference library

This collection increases public-source coverage for condition and medicine questions. It is **separate from the trained symptom classifier** and contains no new patient observations or fabricated hospital records.

Updated 7 October 2026: **6,390 reference topics, 2,465 NHS symptom phrases and 1,238 medicine label names**. Of the topics, **874 are guidance excerpts and 5,516 are Mondo disease dictionary definitions**. Dictionary entries supply definitions, never inferred symptom or treatment rules. 730 topics have symptom lists. The source audit found no errors.

## What the counts mean

The latest topic target is 6,390, ten times the previous 639 topics. Earlier targets were 1,560 reference symptom phrases and 1,210 medicine names. `manifest.json` records actual counts, record types and checksums. `quality_report.json` records source and duplicate checks.

These are different units from classifier features. Condition topics can include broader health problems and procedures. Symptom phrases preserve source wording and context, and several phrases may describe the same symptom. Medicine names can include salts and combinations; they are not a count of unique drug molecules. Combination-name ordering, punctuation and case do not count as extra records. More reference records do not demonstrate better diagnostic accuracy.

## Sources and fields

- NHS Conditions A–Z: whole introductory paragraphs, short symptom list items, and selected whole care paragraphs. Each excerpt links to its source, records the collection date and retains its scope. Some topics put symptoms on a linked child page; those symptoms retain the child-page URL. Excerpts omit information and must not be presented as complete guidelines.
- FDA Structured Product Label API: generic label names, indication excerpts, selected warning/contraindication excerpts, label date, application identifiers and DailyMed label links. Only records containing NDA/ANDA/BLA identifiers and an indication excerpt are retained. These are submitted US label records, not verification of current FDA approval or availability in Pakistan. No personal dose or condition-to-drug recommendation is generated.
- AKUH Pakistan's current public encyclopedia: bounded excerpts from explicit Symptoms sections, collected via the public sitemap. 352 source topics yielded 2,935 combined hospital links in `../pakistan_hospital_symptoms.json`, with 1,453 links mapped to current features. Pages without suitable published lists are logged rather than filled with generated facts. Hospital links are deduplicated by topic and phrase when adding records.
- Mondo release `2026-10-06`, CC BY 4.0: active, defined leaf disease concepts, deduplicated by exact names and exact synonyms against existing references. The import excludes obsolete concepts, broad parent groupings and undefined terms. `disease_dictionary_provenance.json` records the complete downloaded release checksum and source URL. Each record links to its ontology entry. Definitions support named-topic answers only, not classification or treatment.

Information from the NHS website is licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/), subject to [NHS terms](https://www.nhs.uk/our-policies/terms-and-conditions/). See each excerpt's attribution and date. This app is not endorsed by NHS. FDA source documentation and limitations: <https://open.fda.gov/apis/drug/label/>.

## Use and refresh

Dataset Preview displays and downloads the reference tables, with dictionary and guidance counts separated. Direct topic questions can use reference material when the answer provider is offline. Medicine material remains general information; no prescribing mappings are generated. A separately labelled illustrative training set now has 510 labels and 1,560 phrase features. The original held-out examples remain fixed; new classes have no independent clinical test data.

Collect using `.venv/bin/python -m scripts.collect_health_library`. For interrupted medicine collection, `medicine_progress.json` records a restart offset; pass it as `--medicine-start N`. Collecting from page indices is not guaranteed to preserve counts as websites change. A subsequent run should re-check dated content rather than claiming that cached excerpts are current. After collection, run `--part manifest` and `.venv/bin/python -m scripts.audit_health_library`. Do not edit data without regenerating its checksums. Run `.venv/bin/python -m unittest tests.test_health_library` for routing, source scope, duplicate handling and excerpt boundaries.

For the hospital collection, run `.venv/bin/python scripts/collect_pakistan_hospital.py`. Import a complete official Mondo release with `.venv/bin/python scripts/import_disease_dictionary.py /path/to/mondo.obo`, then regenerate the library manifest and audit it. To rebuild the illustrative expansion, run `scripts/build_source_expansion.py`, `scripts/prepare_json.py`, `scripts/audit_data.py`, and retrain with `scripts/train_chatbot.py`. Expanding phrase coverage does not demonstrate improved clinical accuracy; the added labels require independent evaluation.
