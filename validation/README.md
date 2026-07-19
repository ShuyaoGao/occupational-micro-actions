# Independent human validation

Materials and complete records of the blind three-annotator validation study
(construct reliability, encoder-vs-human agreement, and decomposition fidelity),
as reported in the Data Descriptor's Technical Validation.

| File | Content |
|---|---|
| `annotation_guide_en.md` | The guide given to annotators (English translation; annotators worked blind to model labels and study hypotheses; the Chinese original is retained by the authors) |
| `sampleA_fidelity_annotations.csv` | 79-DWA fidelity audit, 3 annotators × 79 items (237 rows); ratings Q1 completeness / Q2 faithfulness / Q3 granularity (1–5); 102 free-text notes in author-verified English translation (Chinese originals retained, available to editors on request) |
| `sampleB_intelligence_type_annotations.csv` | 150-action four-way intelligence-type blind annotation, 3 annotators × 150 items (450 rows) |
| `sampleB_sampling_key.csv` | Join key for sample B: `item` → action text → (`DWA_ID`, `step_order`), encoder label and confidence, macro class. One text with an ambiguous dataset match is flagged (`ambiguous_match = 1`). Stratification variables are reconstructed from the dataset |
| `intelligence_type_prototypes.csv` | The 32 prototype sentences (8 per class) that operationally define the four intelligence types for the nearest-centroid classifier |
| `compute_agreement.py`, `deep_analysis.py` | Agreement statistics (Fleiss/Cohen κ, confusion analysis) as reported |
| `build_bilingual_release.py` | Builds the release CSVs from the returned annotation spreadsheets (includes the full zh→en note translation table; the released sample-A file carries the English column) |

Headline figures reproduced by these scripts: human–human Fleiss κ = 0.90;
encoder-vs-independent-human κ = 0.63 (systematic confusion directions documented);
fidelity means 4.98 / 4.63 / 4.78. Annotators were compensated professionals
recruited with comprehension pre-screening; annotations are released with consent,
anonymised by role. See the Data Descriptor for the full ethics statement.
