# Public hospital research collection

Collected on 2026-10-01 from published CC BY 4.0 research datasets. These files
contain **3,900 distinct clinical observations**, from **4,169 published rows**,
after **269 repeats** were removed. No synthetic examples were generated.

| Cohort | Country | Published rows | Distinct observations | Study target |
| --- | --- | ---: | ---: | --- |
| [YICSS Pakistan, v2](https://data.mendeley.com/datasets/3pgb37wck4/2) | Pakistan | 2,950 | 2,950 | Infant referral for urgent hospital management, excluding jaundice |
| [Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) | Pakistan | 299 | 299 | Death during follow-up in diagnosed heart-failure patients |
| [Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) | Bangladesh | 520 | 251 | Positive/negative diabetes questionnaire label |
| [Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | India | 400 | 400 | CKD/non-CKD clinical label |

The Pakistan infant cohort was collected at Aga Khan University Hospital's
community/primary healthcare sites in Karachi, with hospital referral outcomes.
It is not a cohort of tertiary-hospital inpatients. The Faisalabad cohort came
from Faisalabad Institute of Cardiology and Allied Hospital. Bangladesh's
questionnaire cohort came from Sylhet Diabetes Hospital. The Indian cohort is
reported as Apollo Hospitals, Tamil Nadu in the linked study. Source URLs,
provenance evidence, full credits, variable metadata and licences are recorded
in `sources.json`. This collection does not imply endorsement by those hospitals.

## Relationship to chatbot training

The collection exceeds a 3,040-observation research target based on 10 times
the previous 304-pattern dataset. These observations do not expand the symptom
classifier because their tasks differ. A separate expansion now provides 3,040
educational symptom patterns across 51 labels using clearly marked illustrative
subsets. See [the expansion guide](../EXPANSION.md). Clinical outcomes, infant
referral labels and laboratory values are preserved in their original cohorts.

All prepared records have `eligible_for_classifier_training: false`. The chat and
its symptom-classifier training script do not read these files. A separate
`scripts/benchmark_hospital_models.py` trains research models for the three adult
cohorts, retaining their original targets. These are internal benchmarks, not
external clinical validation, and are never used in chat inference. Dataset Preview provides
counts, credits and downloads. No individual records are sent to a chat provider.

Each JSONL record contains a content hash `id`, `source_id`, `feature_group_id`,
`features`, the original `target`, and original spreadsheet/CSV `source_rows`
(header is row 1). Feature/target values are source strings or null. Whitespace
is trimmed and empty, `?`, `NaN`, and `nan` values become null. Source codes such
as `-9` remain intact. No value imputation, symptom inference, diagnosis mapping,
or treatment recommendations have been added.

Original identifiers, administrative fields, location fields, free text and
later clinical outcomes from YICSS are excluded. Its 42 selected features and
referral target are retained. Four source records have computed age 60 days
despite the published 0–59-day inclusion range; the audit flags this discrepancy.
Some categorical encodings still need verification against the original study
forms. The heart-failure `time` column is observed follow-up, so it must not be
treated as a baseline predictor of subsequent death.

Identical allowlisted feature observations within a cohort are grouped. Matching
targets are deduplicated while retaining every source-row reference. Contradictory
targets are quarantined in `*.conflicts.jsonl` rather than selecting one label.
There are currently no conflicting groups. Distinct observations do not prove
independent patients, and cohorts with different targets are not merged for
model fitting. Future splits should keep feature groups and any independently
verified participant groups together.

## Reproduce and audit

The importer uses Python's standard library. It reads workbook cached values
without executing formulas. Published downloads are pinned by SHA-256 and exact
source schema/row counts. A changed source is rejected for review.

```bash
python3 scripts/collect_regional_data.py --download
python3 scripts/audit_regional_data.py
python3 -m unittest tests.test_regional_data
```

Downloads normally use a temporary directory, which is removed afterwards. For
existing downloads use `--raw-dir /path/to/downloads`, using the `local_name`
filenames in the catalog. Do not commit the original YICSS workbook with its
study identifiers. `sources.json` retains only relevant clinical codebook labels
and its download URL/checksum, rather than republishing the complete codebook.

The audit checks prepared file hashes, allowlisted schemas, research status,
target values, duplicate feature groups, record hashes, source-row coverage,
counts, missingness and the collection target. `quality_report.json` is the saved
result. To expand the production classifier, obtain compatible symptom records
with confirmed disease labels, verify participant independence, define clinically
reviewed mappings and evaluate an independent held-out cohort before retraining.

All four datasets are used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain the source catalog and original credits when redistributing prepared data.
