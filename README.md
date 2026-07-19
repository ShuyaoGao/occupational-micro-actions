# Bipolar Action Substrate of Occupational Tasks

Reproducibility package for the micro-action research programme (Gao & Huang, 2026):
- **Preprint** (arXiv:2606.07939, v2): *The atomic structure of work: a micro-action instrument reveals two-pole AI occupational exposure and its decade-scale polar inversion* (v1 title: "Stable Geometry, Reversing Poles")
- **Dataset**: Zenodo, concept DOI [10.5281/zenodo.21395792](https://doi.org/10.5281/zenodo.21395792) (15,817 ordered micro-actions; a Scientific Data descriptor is in preparation)

The pipeline decomposes 1,961 O\*NET Detailed Work Activities (DWAs) into **15,817 micro-actions** via a multi-agent LLM pipeline (4 local + 3 frontier models, consensus voting with frontier-model arbitration), then clusters those micro-actions in two stages:

1. **HDBSCAN micro-clustering** in a UMAP-reduced sentence-embedding space (35 micro-clusters + 5,637-action "Generic Substrate" noise group),
2. **Hierarchical Ward macro-clustering** of the 35 micro-clusters into a 7-macro typology (M1-M7),

and then projects the DWA-level **GPT-4 task-exposure score** (built from Eloundou et al.'s public task ratings) onto this typology to test two questions:

- **(spatial axis)** Is the occupational-substitutability distribution continuous or polarised?
- **(temporal axis)** Is the polar identity of the typology stable across capability eras (Frey-Osborne 2013 vs Eloundou 2023)?

The headline empirical findings are a **bipolar structure** (Cohen's d = 2.98, Cliff's δ = 0.902 between M2 "Tool-Mediated Physical" and M7 "Planning & Design") and a **polarity inversion** (macro-level Spearman ρ = -0.750, p = 0.020) over a decade.


## Validation, audits, and dataset-revision scripts (added 2026-07)

| Folder | Content |
|---|---|
| `validation/` | Complete blind three-annotator validation records (construct κ = 0.90; encoder-vs-human κ = 0.63), bilingual fidelity notes, sampling key, the 32 intelligence-type prototype sentences, and agreement scripts |
| `audits/` | Substrate filler-integrity audit (OR 0.76, depletion) and the same-pipeline comparison against O*NET's 18,796-record task layer (noise 3.3% vs 35.6%) |
| `omega_provenance/` | The importance-weighted aggregation chain that produces the occupation ω table (dataset file 09), verified to reproduce it with zero deviation |
| `v1_1_migration/` | Scripts producing the Zenodo v1.1 dataset revision (field normalisation, label renaming, English status codes, integer cluster IDs) |

Each folder carries its own README.

## Repository layout

```
bipolar-action-substrate/
├── README.md
├── LICENSE                                          (MIT)
├── requirements.txt
├── .gitignore
│
├── scripts/                                         (16 sequential pipeline steps + 4 robustness scripts)
│   ├── step_01_build_database.py                    Phase 1 - O*NET data prep
│   ├── step_02_build_occupation_forest.py
│   ├── step_03_build_mapping_trees.py
│   ├── step_04_build_global_dicts.py
│   │
│   ├── step_05_decompose_dwa_with_llms.py           Phase 2 - Multi-agent LLM decomposition
│   ├── step_06_cross_synthesize.py
│   ├── step_07_peer_voting.py
│   │
│   ├── step_08_generate_judge_batches.py            Phase 3 - Judge adjudication + golden dataset
│   ├── step_09_merge_judge_answers.py
│   ├── step_10_tally_judge_votes.py
│   ├── step_11_finalize_golden_dataset.py
│   │
│   ├── step_12_flatten_actions.py                   Phase 4 - Feature extraction
│   ├── step_13_extract_features.py
│   │
│   ├── step_14_action_clustering.py                 Phase 5 - Clustering + exposure projection
│   ├── step_15_K7_main_analysis.py
│   ├── step_16_exposure_projection.py
│   │
│   └── robustness/
│       └── K5_raw_appendix_a.py                     K=5 raw cut (Appendix A)
│       (resolution sweep, TOST, and the M4 chimera split are all
│        reproduced by step_16_exposure_projection.py)
│
└── data/
    ├── final/                                       (~10 MB; canonical analytic objects)
    │   ├── final_golden_dataset.csv                 1,961 DWAs × 15,817 micro-actions
    │   ├── actions_with_features.csv                Per-action 6-field LLM feature labels
    │   ├── HDBSCAN_cluster_labels.npy               35 micro-clusters + noise
    │   ├── clustering_summary.json                  Clustering metadata
    │   └── dwa_macro_distribution.csv               Per-DWA macro shares (K=7)
    │
    ├── external/                                    Exposure indicators + auxiliary inputs for step_16
    │   ├── dwa_external_indices.csv                 DWA-level Eloundou / AIOE / Frey-Osborne alignment
    │   ├── actions_intelligence_type_encoderA.csv   BGE intelligence-type labels (15,817 actions)
    │   ├── macro_assignments_K{7,8,10,12,15}.csv    Ward-cut assignments for the resolution sweep
    │   └── macro_era_means_with_fo_original.csv     FO Oxford-Martin-original macro means
    │
    ├── intermediate/                                (~117 MB; full Phase 2-5 chain for replay)
    │   ├── phase2_llm_decomposition/                Output of step_05 (×4 local models)
    │   ├── phase2_llm_synthesis/                    Output of step_06 (×4 local models)
    │   ├── phase2_peer_voting/                      Output of step_07 (×4 voters)
    │   ├── phase3_judge_inputs/                     Batched prompts for the three frontier judges
    │   ├── phase3_judge_answers/                    Raw judge answers (placeholder dirs)
    │   ├── phase3_voting_detail.csv                 Step_09 per-row vote detail
    │   ├── phase3_golden_sequences_auto.csv         Step_10 auto-resolved branch
    │   ├── phase3_conflicts_pending.csv             Step_11 conflict branch (105 cases)
    │   ├── phase4_actions_flat.csv                  Step_12 flattened actions table
    │   ├── phase4_MPNet_embeddings.npy              Step_13 768-d MPNet embeddings
    │   ├── phase5_macro_features_raw.npy            Step_14 / step_15 macro features (raw)
    │   ├── phase5_macro_features_zscored.npy        Step_14 / step_15 macro features (z-scored)
    │   └── phase5_macro_linkage_matrix.npy          Ward linkage matrix (15-d feature vector input)
    │
    └── robustness/                                  (appendix-grade artefacts)
        ├── intelligence_and_era/                    MPNet intelligence-type labels + prototypes
        └── bge_cross_check/                         BGE encoder cross-check + 150-row human audit (Appendices D, E)
```

## Pipeline at a glance

| Phase | Steps | What it does |
|-------|-------|--------------|
| **1. O\*NET data prep** | 01-04 | Load the O\*NET 30.2 release into SQLite, build per-occupation JSON forests, inverted Tasks-to-DWAs mapping trees, and global definition dictionaries. |
| **2. LLM decomposition** | 05-07 | Decompose each DWA into a 5-12 step micro-action sequence using a 4-LLM ensemble (Qwen / Llama / Gemma / Mistral): independent decomposition (step_05), cross-synthesis (step_06), peer voting (step_07). |
| **3. Adjudication + golden dataset** | 08-11 | Route the 502 contested DWAs to three frontier judges (Claude / Gemini / GPT), merge majority votes (step_09), tally + stratified sampling (step_10), produce `final_golden_dataset.csv` (step_11). |
| **4. Feature extraction** | 12-13 | Flatten 1,961 DWAs into 15,817 micro-action rows (step_12), then LLM-extract six structured feature fields per action (step_13). |
| **5. Clustering + exposure projection** | 14-16 | UMAP→HDBSCAN micro-clustering + Ward macro-clustering at K=7 (step_14, step_15), then project the DWA-level GPT-4 task-exposure score onto the typology and run the full statistical battery, including the resolution sweep, TOST, M4 chimera, cross-indicator reproduction, era inversion, and intelligence-type analysis (step_16). |
| **Robustness** | `robustness/` | K=5 raw cut for Appendix A; the resolution sweep, TOST equivalence, and K=12 M4 chimera analyses are reproduced end-to-end inside step_16. |

## What you need to fetch separately

Two dependencies are **not redistributed** here for licensing / size reasons:

1. **O\*NET 30.2 raw database** (~97 MB; 41 tab-delimited `.txt` files)
   - Download release **30.2** from <https://www.onetcenter.org/db_releases.html>
   - Unpack into `data/onet_30_2_raw/`
   - Required by `step_01_build_database.py`

2. **Local LLM weights** (only if you want to re-run the Phase 2 decomposition / synthesis / voting from scratch)
   - Qwen2.5-32B-Instruct • Gemma-2-27b-it • Meta-Llama-3.1-8B-Instruct • Mistral-Nemo-Instruct-2407 (all GGUF Q4\_K\_M)
   - Required by `step_05`, `step_06`, `step_07`
   - You can also point `API_URL` at any OpenAI-compatible backend (vLLM, llama.cpp server, text-generation-webui).

If you only want to reproduce the **paper figures and statistics**, the cached intermediate products under `data/intermediate/` and `data/final/` let you skip Phases 2 and 3 entirely and start from step_12.

## Reproducing the canonical results

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Path A - fastest: use the cached golden dataset (skip Phases 2-3)
python scripts/step_12_flatten_actions.py
python scripts/step_13_extract_features.py
python scripts/step_14_action_clustering.py
python scripts/step_15_K7_main_analysis.py
python scripts/step_16_exposure_projection.py

# Even faster: step_16 alone reproduces every exposure statistic in the paper
# from the shipped data/final/ + data/external/ files (no upstream steps needed).
python scripts/step_16_exposure_projection.py

# Path B - full pipeline (needs O*NET 30.2 + four local LLMs + three frontier model accounts)
python scripts/step_01_build_database.py
python scripts/step_02_build_occupation_forest.py
python scripts/step_03_build_mapping_trees.py
python scripts/step_04_build_global_dicts.py
# For each of qwen / llama / gemma / mistral, edit CURRENT_MODEL_NAME and run:
python scripts/step_05_decompose_dwa_with_llms.py
python scripts/step_06_cross_synthesize.py
python scripts/step_07_peer_voting.py
# Phase 3: judge batches are human-in-the-loop (you paste each batch into the
# corresponding model's web UI and save the answer file).
python scripts/step_08_generate_judge_batches.py
# ... fill in data/intermediate/phase3_judge_answers/{claude,gemini,gpt}/
python scripts/step_09_merge_judge_answers.py
python scripts/step_10_tally_judge_votes.py
python scripts/step_11_finalize_golden_dataset.py
# Then continue with step_12 ... step_16 as in Path A.

# Appendix A robustness cut (after step_15 is done)
python scripts/robustness/K5_raw_appendix_a.py
```

## A note on the exposure indicators

Sections 3.5, 4.3, 4.7, and 5 of the paper project three externally produced
exposure indicators onto the K=7 macro typology: **Eloundou et al.'s GPT-4
task ratings** (the primary measure; tiers mapped to {0, 0.25, 0.5, 0.75, 1}
and averaged to the DWA level via the O\*NET Tasks-to-DWAs crosswalk),
**Felten et al.'s AIOE**, and **Frey-Osborne's computerisation probability**
(parsed directly from the Oxford Martin working-paper appendix). The aligned
DWA-level table is shipped at `data/external/dwa_external_indices.csv`, so
`step_16_exposure_projection.py` runs out of the box. To rebuild the alignment
from the raw third-party sources, fetch: the Eloundou task-rating files from
their public GPTs-are-GPTs repository, the AIOE data appendix from Felten et
al. (2021), and the Frey-Osborne working paper PDF from the Oxford Martin
School. None of the three is redistributed here beyond the derived alignment.

## Environment

- Python 3.11 (tested) or 3.10
- See `requirements.txt`. Major deps: `pandas`, `numpy`, `scipy`, `scikit-learn`, `umap-learn`, `hdbscan`, `sentence-transformers`, `matplotlib`, `seaborn`, `requests`.
- Phase 2 (LLM decomposition) was run on dual NVIDIA RTX 3090 (24 GB ×2). A single 24 GB card works; runtime scales linearly.
- Phase 5 sentence embeddings (`step_13`) use `sentence-transformers/all-mpnet-base-v2`; the BGE cross-check in `data/robustness/bge_cross_check/` uses `BAAI/bge-large-en-v1.5`.

## Citation

```bibtex
@article{gao2026bipolar,
  author = {Gao, Shuyao and Huang, Minghao},
  title  = {Stable Geometry, Reversing Poles: The Bipolar Structure of AI Occupational Substitutability and Its Decade-Scale Inversion},
  year   = {2026}
}
```

## License

MIT - see [LICENSE](LICENSE).

The O\*NET 30.2 raw data is governed by its own license; see <https://www.onetcenter.org/license_onet.html>.

## Contact

- Shuyao Gao: gaoshuyao.mk@gmail.com
- Corresponding author: Minghao Huang, aSSIST University, Seoul
- Issues and questions: please open a [GitHub issue](../../issues).
