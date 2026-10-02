# Expanded health reference library

This collection increases public-source coverage for condition and medicine questions. It is **separate from the trained symptom classifier** and contains no new patient observations or fabricated hospital records.

As collected on 2 October 2026: **639 condition topics, 2,465 scoped symptom phrases and 1,238 medicine label names**. The audit found no errors. 495 topics contain extracted symptom lists; one index page could not supply a suitable excerpt and is listed in `collection_failures.json`.

## What the counts mean

The request used the existing 51 classifier labels, 156 classifier features and 121 legacy medicine entries as baselines. The collection targets at least 510 condition topics, 1,560 source symptom phrases and 1,210 distinct medicine label names. `manifest.json` records the actual counts and checksums; `quality_report.json` records duplicates/provenance checks and whether each reference target was met.

These are different units from classifier features. Condition topics can include broader health problems and procedures. Symptom phrases preserve source wording and context, and several phrases may describe the same symptom. Medicine names can include salts and combinations; they are not a count of unique drug molecules. Combination-name ordering, punctuation and case do not count as extra records. More reference records do not demonstrate better diagnostic accuracy.

## Sources and fields

- NHS Conditions A–Z: whole introductory paragraphs, short symptom list items, and selected whole care paragraphs. Each excerpt links to its source, records the collection date and retains its scope. Some topics put symptoms on a linked child page; those symptoms retain the child-page URL. Excerpts omit information and must not be presented as complete guidelines.
- FDA Structured Product Label API: generic label names, indication excerpts, selected warning/contraindication excerpts, label date, application identifiers and DailyMed label links. Only records containing NDA/ANDA/BLA identifiers and an indication excerpt are retained. These are submitted US label records, not verification of current FDA approval or availability in Pakistan. No personal dose or condition-to-drug recommendation is generated.
- Existing Pakistani hospital references remain in `../pakistan_hospital_symptoms.json` with their original provenance. No NHS/FDA entry is represented as a Pakistani hospital case.

Information from the NHS website is licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/), subject to [NHS terms](https://www.nhs.uk/our-policies/terms-and-conditions/). See each excerpt's attribution and date. This app is not endorsed by NHS. FDA source documentation and limitations: <https://open.fda.gov/apis/drug/label/>.

## Use and refresh

Dataset Preview displays and downloads all three reference tables. Direct topic questions can use this library as grounded answer material, including when the answer provider is offline. Medicine label material remains general information; no new prescribing mappings are added to legacy classifier conditions. Classifier labels, weights and held-out evaluations remain unchanged.

Collect using `.venv/bin/python -m scripts.collect_health_library`. For interrupted medicine collection, `medicine_progress.json` records a restart offset; pass it as `--medicine-start N`. Collecting from page indices is not guaranteed to preserve counts as websites change. A subsequent run should re-check dated content rather than claiming that cached excerpts are current. After collection, run `--part manifest` and `.venv/bin/python -m scripts.audit_health_library`. Do not edit data without regenerating its checksums. Run `.venv/bin/python -m unittest tests.test_health_library` for routing, source scope, duplicate handling and excerpt boundaries.

To expand trained classification to hundreds of labels/features, the project still needs appropriately labelled examples and independent evaluation for those labels. This reference library does not provide that evidence. The sparse fatigue/headache example can remain uncertain even with broader references; increasing records is not a reason to display a definite cause or select a medicine.
