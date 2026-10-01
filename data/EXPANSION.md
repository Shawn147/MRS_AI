# Dataset expansion, 2026-10-01

| Content | Before | After |
| --- | ---: | ---: |
| Original CSV rows | 4,920 | 4,920 preserved |
| Total classifier input rows | 4,920 | 7,656 |
| Unique classifier patterns | 304 | 3,040 |
| Conditions | 41 | 51 |
| Symptom features | 131 | 156 |
| Source-linked reference records | 7 | 17 |
| Illustrative conversation examples | 180 / 9 intents | 450 / 12 intents |
| Separate public hospital observations | 0 | 3,900 |

The classifier includes 304 original educational patterns and 2,736 generated
subsets. They are **illustrative, not observed patient cases**. Original duplicate
counts and CSV provenance remain intact. The public hospital collection is
separate because its original labels and features cover different tasks.

`expansion_profiles.json` records sourced symptom inventories and aliases. It
does not invent severity weights or medication mappings. `training.json` marks
each generated row with its training-only split, parent family, provenance and
review status. Ambiguous generated patterns spanning multiple condition labels
are excluded. Original validation/test IDs remain in `legacy_splits.json`.

Training assigns equal total weight to each condition and each training family.
There are 2,918 training examples, 61 validation examples and 61 test examples.
The fixed test set covers only the original labels. **The ten added conditions
have no independent evaluation**, and neither illustrative subsets nor the
existing template holdouts establish clinical accuracy. New condition medicine
mappings are empty, with medicine information withheld in the app.

The source summaries are educational and await clinical/local review. UK NHS
guidance does not establish local prescribing rules. The COVID source's last
published review date is 2023-03-21; its scheduled review date has passed. Some
hospital pages publish no review date; this is recorded rather than guessed.

Condition references include [NHS anaemia](https://www.nhs.uk/conditions/iron-deficiency-anaemia/),
[ear infections](https://www.nhs.uk/conditions/ear-infections/),
[appendicitis](https://www.nhs.uk/conditions/appendicitis/),
[tonsillitis](https://www.nhs.uk/conditions/tonsillitis/),
[norovirus](https://www.nhs.uk/conditions/norovirus/) and
[dehydration](https://www.nhs.uk/conditions/dehydration/).
Pakistan references include Aga Khan University Hospital's
[kidney disease](https://hospitals.aku.edu/pakistan/diseases-and-conditions/Pages/chronic-kidney-disease.aspx),
[tuberculosis](https://hospitals.aku.edu/pakistan/diseases-and-conditions/Pages/tuberculosis.aspx),
[dengue](https://hospitals.aku.edu/pakistan/encyclopedia/dengue) and
[typhoid](https://hospitals.aku.edu/pakistan/encyclopedia/typhoid) pages.
All source links/access dates accompany the relevant JSON records.

## Reproduce

```bash
.venv/bin/python scripts/prepare_json.py
.venv/bin/python scripts/audit_data.py
.venv/bin/python scripts/train_chatbot.py --epochs 6
.venv/bin/python scripts/expand_context.py
.venv/bin/python scripts/train_context.py
.venv/bin/python scripts/audit_regional_data.py
.venv/bin/python -m unittest discover -s tests
```

The clinical collection has its own [source and preparation guide](regional/README.md).
