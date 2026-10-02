# Pakistani hospital symptom collection

Collected/checked on 1 October 2026. `pakistan_hospital_symptoms.json` contains 71 topic–symptom links, 58 distinct symptom phrases and nine published pages from three hospitals:

- Aga Khan University Hospital, Karachi: malaria, asthma, childhood ear infections and existing kidney disease, dengue, typhoid and tuberculosis references.
- Pakistan Kidney and Liver Institute and Research Centre, Lahore: gallstones.
- Shifa International Hospital, Islamabad: emergency warning signs.

Each link retains the hospital, page URL, access date, population/scope, and an optional existing classifier feature ID. Different topics can share a symptom; these links are not independent patient cases. Lists are curated summaries of public hospital guidance, not private records. Missing page revision dates are marked as unpublished rather than guessed. Source content is paraphrased; no full hospital pages are redistributed.

## Use in the project

45 links map to existing model features. Nine additional phrases are added to runtime symptom aliases, including `upper abdominal pain`, `yellow eyes` and `fluid draining from the ear`. Novel signs such as wheezing, uncontrolled bleeding and sudden loss of vision remain reference-only because the classifier has no trained feature for them. They must not be silently assigned to unrelated features.

Five new source-linked answer topics increase the educational reference collection from 17 to 22. Dataset Preview offers counts and a JSON download. The core classifier remains at 156 features, 51 labels and 3,040 unique training patterns. Hospital guidance does not establish new patient observations or validate that classifier.

Run `.venv/bin/python -m scripts.audit_data` to validate provenance, mappings and duplicates and update `quality_report.json`, including the collection's checksum. Run `.venv/bin/python -m unittest tests.test_hospital_symptoms tests.test_data_quality` for mapping/negation and provenance checks. Future additions should cite a verified official page, preserve its age/severity scope and add no medication doses. Re-evaluate and retrain before changing core feature IDs.
