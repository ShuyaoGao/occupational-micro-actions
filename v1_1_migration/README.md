# Zenodo v1.1 migration scripts

Scripts that produce the v1.1 revision of the Zenodo dataset from v1.0.x.

| Script | What it does |
|---|---|
| `normalize_structured_fields.py` | `cognitive_or_physical`: 39 raw LLM-emitted variants → {Cognitive, Physical, Mixed, Other} by a deterministic rule (raw preserved in `*_raw`); `mapped_stage`: 10 compound tags → first-listed canonical stage (raw preserved). Also renames the v1.0 label string `LLM` → `Linguistic` (incl. confidence columns) in file 04 |
| `finalize_v11_files.py` | File 01: `Status` Chinese local-vote tags → English codes (`local_vote_4_0` etc., original kept in `Status_zh`); file 07: `C0…C34` → integer micro-cluster IDs aligned with files 04/08; sample-B header hygiene; builds `sampleB_sampling_key.csv` |

The full v1.1 change list (file-03 withdrawal, README overhaul, added validation
records) is documented in the Data Descriptor's Usage Notes.
