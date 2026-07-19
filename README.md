# Occupational Micro-Actions: The Atomic Structure of Work

[![Dataset DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21395793-blue.svg)](https://doi.org/10.5281/zenodo.21395793)
[![arXiv](https://img.shields.io/badge/arXiv-2606.07939-b31b1b.svg)](https://arxiv.org/abs/2606.07939)

Reproducibility package for **"The atomic structure of work: a micro-action instrument reveals two-pole AI occupational exposure and its decade-scale polar inversion"** (Gao & Huang, 2026, arXiv:2606.07939; v1 was titled "Stable Geometry, Reversing Poles"). A Scientific Data descriptor of the dataset is in preparation.

**The dataset is published as [The Micro-Action Dataset on Zenodo](https://doi.org/10.5281/zenodo.21395793)** — 15,817 atomic micro-actions decomposed from 1,961 O\*NET 30.2 Detailed Work Activities (DWAs) across 923 U.S. occupations, with the seven-macro semantic typology, four-way intelligence-type labels, and the DWA-level Occupational Automation Index (OAI).

The pipeline decomposes each DWA into an ordered micro-action sequence via a **consensus multi-agent LLM pipeline** (four open-weight models independently draft, cross-read, and vote; three frontier-class models arbitrate contested cases; 126 low-agreement DWAs are discarded), validated downstream by an independent blind three-annotator study: the four-way intelligence-type construct is highly reproducible across humans (Fleiss κ = 0.90), while the automatic encoder labels match the independent annotators at κ = 0.63 (reliable in aggregate, noisy per action); the earlier in-loop author audit (κ = 0.893) is an optimistic upper bound, not independent validation. Full records in `validation/`. It then clusters the actions in two stages:

1. **HDBSCAN micro-clustering** in a UMAP-reduced sentence-embedding space (35 micro-clusters + 5,637-action "Generic Substrate" noise group),
2. **Hierarchical Ward macro-clustering** of the 35 micro-clusters into a 7-macro typology (M1-M7),

and then projects two methodologically independent exposure indicators — the tech-risk **Occupational Automation Index** (arXiv:2604.04464) and Eloundou et al.'s GPT-4 task ratings — onto this typology to test two questions:

- **(spatial axis)** Is the occupational-substitutability distribution continuous or polarised?
- **(temporal axis)** Is the polar identity of the typology stable across capability eras (Frey-Osborne 2013 vs LLM-era OAI 2026)?

The headline empirical findings are a **bipolar structure** (Cohen's d = 2.41 between M2 "Tool-Mediated Physical" and M7 "Planning & Design") and a **polarity inversion** (macro-level Spearman ρ = -0.750, p = 0.020) over thirteen years.

> **Terminology note**: an *occupational micro-action* is the smallest purposeful step of occupational work — distinct from "micro-action recognition" in video understanding, which denotes subtle involuntary body movements.

## Validation, audits, and dataset-revision scripts (added 2026-07)

| Folder | Content |
|---|---|
| `validation/` | Complete blind three-annotator validation records (bilingual fidelity notes, intelligence-type annotations, sampling key, the 32 intelligence-type prototype sentences, agreement scripts, annotation guide) |
| `audits/` | Substrate filler-integrity audit (filler is depleted in the substrate, OR 0.76) and the same-pipeline comparison against O*NET's 18,796-record task layer (noise 3.3% vs 35.6% — the substrate is a property of step resolution) |
| `omega_provenance/` | The importance-weighted aggregation chain producing the occupation ω table (dataset file 09), verified to reproduce it with zero deviation |
| `v1_1_migration/` | Scripts producing the Zenodo v1.1 dataset revision (field normalisation, `LLM`→`Linguistic` renaming, English status codes, integer cluster IDs) |

Each folder carries its own README.

## Repository layout

```
occupational-micro-actions/
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
│   ├── step_14_action_clustering.py                 Phase 5 - Clustering + OAI projection
│   ├── step_15_K7_main_analysis.py
│   ├── step_16_oai_projection.py
│   │
│   └── robustness/                                  Robustness suite (Sections 4.3, 4.4 + Appendices)
│       ├── resolution_sweep.py                      K=5, 7, 8, 10, 12, 15 sweep
│       ├── tost_equivalence.py                      Two-one-sided equivalence test
│       ├── k12_chimera_split.py                     M4 chimera analysis
│       └── K5_raw_appendix_a.py                     K=5 raw cut (Appendix A)
│
└── data/
    ├── final/                                       (~10 MB; canonical analytic objects)
    │   ├── final_golden_dataset.csv                 1,961 DWAs × 15,817 micro-actions
    │   ├── actions_with_features.csv                Per-action 6-field LLM feature labels
    │   ├── HDBSCAN_cluster_labels.npy               35 micro-clusters + noise
    │   ├── clustering_summary.json                  Clustering metadata
    │   └── dwa_macro_distribution.csv               Per-DWA macro shares (K=7)
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
    └── robustness/                                  (~5 MB; appendix-grade summaries)
        ├── main_outputs/                            Resolution sweep + TOST + K=12 split summaries
        ├── intelligence_and_era/                    Intelligence-type labels + era inversion
        └── bge_cross_check/                         BGE encoder cross-check (Appendices D, E)
```

## Pipeline at a glance

| Phase | Steps | What it does |
|-------|-------|--------------|
| **1. O\*NET data prep** | 01-04 | Load the O\*NET 30.2 release into SQLite, build per-occupation JSON forests, inverted Tasks-to-DWAs mapping trees, and global definition dictionaries (shared with the OAI pipeline, arXiv:2604.04464). |
| **2. LLM decomposition** | 05-07 | Decompose each DWA into a 5-12 step micro-action sequence using a 4-LLM ensemble (Qwen / Llama / Gemma / Mistral): independent decomposition (step_05), cross-synthesis (step_06), peer voting (step_07). |
| **3. Adjudication + golden dataset** | 08-11 | Route the 502 contested DWAs to three frontier judges (Claude / Gemini / GPT), merge majority votes (step_09), tally + stratified sampling (step_10), produce `final_golden_dataset.csv` (step_11). |
| **4. Feature extraction** | 12-13 | Flatten 1,961 DWAs into 15,817 micro-action rows (step_12), then LLM-extract six structured feature fields per action (step_13). |
| **5. Clustering + OAI projection** | 14-16 | UMAP→HDBSCAN micro-clustering + Ward macro-clustering at K=7 (step_14, step_15), then project the DWA-level OAI onto the typology and run the full statistical battery (step_16). |
| **Robustness** | `robustness/` | Resolution sweep K=5..15, TOST equivalence on 15 middle-pair OAIs, the K=12 M4 chimera split, and the K=5 raw cut. |

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
python scripts/step_16_oai_projection.py

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

# Robustness suite (after step_15 / step_16 are done)
python scripts/robustness/resolution_sweep.py
python scripts/robustness/tost_equivalence.py
python scripts/robustness/k12_chimera_split.py
python scripts/robustness/K5_raw_appendix_a.py
```

## A note on the OAI projection

The paper projects the DWA-level Occupational Automation Index (constructed in
arXiv:2604.04464) onto the K=7 macro typology produced here. The OAI table is
included in [The Micro-Action Dataset on Zenodo](https://doi.org/10.5281/zenodo.21395793)
as `06_dwa_automation_index_oai.csv`, and also lives in its own reproducibility
repository (<https://github.com/ShuyaoGao/bounded-risk-oai>); place it under
`data/intermediate/paper1_dwa_oai.csv` before running `step_16_oai_projection.py`.

## Environment

- Python 3.11 (tested) or 3.10
- See `requirements.txt`. Major deps: `pandas`, `numpy`, `scipy`, `scikit-learn`, `umap-learn`, `hdbscan`, `sentence-transformers`, `matplotlib`, `seaborn`, `requests`.
- Phase 2 (LLM decomposition) was run on dual NVIDIA RTX 3090 (24 GB ×2). A single 24 GB card works; runtime scales linearly.
- Phase 5 sentence embeddings (`step_13`) use `sentence-transformers/all-mpnet-base-v2`; the BGE cross-check in `data/robustness/bge_cross_check/` uses `BAAI/bge-large-en-v1.5`.

## Citation

```bibtex
@misc{gao2026atomic,
  author       = {Gao, Shuyao and Huang, Minghao},
  title        = {The Atomic Structure of Work: Micro-Action Decomposition Reveals the Bipolar Geometry of {AI} Occupational Substitutability},
  year         = {2026},
  howpublished = {arXiv preprint arXiv:2606.07939}
}

@dataset{gao2026microactiondataset,
  author    = {Gao, Shuyao},
  title     = {The Micro-Action Dataset: An Atomic Decomposition of O*NET Work Activities},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21395793}
}
```

Companion paper (the DWA-level OAI projected here):

> Gao, S. & Huang, M. (2026). *Bounded by Risk, Not Capability: A Tech-Risk Dual-Factor Decomposition of the Occupational Automation Index.* arXiv preprint arXiv:2604.04464.

## License

MIT - see [LICENSE](LICENSE).

The O\*NET 30.2 raw data is governed by its own license; see <https://www.onetcenter.org/license_onet.html>.

## Contact

- Shuyao Gao: gaoshuyao.mk@gmail.com
- Corresponding author: Minghao Huang, aSSIST University, Seoul
- Issues and questions: please open a [GitHub issue](../../issues).
