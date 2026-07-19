# Provenance of the occupation ω table (dataset file 09)

`09_occupation_omega_table.csv` in the Zenodo release is produced by the
importance-weighted chain implemented here:

1. **`phase1_build_profiles.py`** — builds per-occupation profiles.
   For occupation *j*, each task *t* carries its O*NET importance rating
   IM(*t*, *j*); each (task, DWA) link receives weight IM(*t*, *j*)/N(*t*),
   where N(*t*) is the number of DWAs linked to *t* globally; every micro-action
   of a linked, decomposed DWA inherits that weight.
   ω_k = weighted type-*k* action count / weighted all-action count
   (four-way intelligence-type partition; sums to 1).
   `substrate_share` = weighted noise-action share under the eight-way
   macro-class partition (same weights, different partition — overlaps the ω's).
2. **`omega_table_export_provenance.py`** — the export step (section B4):
   selects and renames the profile columns into the released file-09 schema.

**Verification:** the output of this chain reproduces the released file 09 with
zero deviation on every column (all 923 occupations).

Note on paths: these are the original scripts, preserved verbatim as provenance;
input paths reference the authors' original workspace layout and the O*NET 30.2
task-ratings files. Adjust paths to re-run.
