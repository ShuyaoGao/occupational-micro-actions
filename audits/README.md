# Structural audits

Two audits reported in the Data Descriptor's Technical Validation.

## `filler_audit.py` — generic-substrate integrity
Tests whether the 35.6% HDBSCAN-noise layer ("generic substrate") is enriched in
the ceremonial/navigational filler steps that the blind annotators identified as
a generation failure mode. Two lexicons: NARROW (strictly annotator-derived) and
BROAD (deliberately widened in favour of the artefact hypothesis; the reported
figures). Result: filler is UNDER-represented in the substrate
(5.2% vs 6.7%, OR 0.76, Fisher P = 1.4e-4; per-DWA Spearman rho = -0.099) —
the substrate is not a deposit of generation filler.

## `test1_task_statements_pipeline.py` — positioning against O*NET's task layer
Runs the identical typology pipeline (MPNet → UMAP 5-d, seed 42 → HDBSCAN
mcs=100/ms=20 → Ward) on the 18,796 task records of O*NET 30.2.
Result: 3 coarse clusters, noise share 3.3% (vs 35 clusters / 35.6% at the
micro-action layer) — the generic substrate is a property of step resolution,
not of O*NET's whole-activity layers. Hyperparameters are held at the
micro-action settings; see the Descriptor for the scoping caveat.

Requires: `sentence-transformers`, `umap-learn`, `hdbscan`, `scipy`, `pandas`;
O*NET 30.2 text files (Task Statements, Tasks to DWAs) from onetcenter.org.
