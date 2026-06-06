# Intelligence Type × Macro — BGE Re-run

This file replaces the MPNet-based §4.8 of batch 2 with the BGE-labelled version, after the 150-row human audit established that BGE labels match human ground truth at κ=0.893 vs MPNet's κ=0.769. The MPNet numbers are retained in Appendix D as a robustness check (the qualitative conclusions match).

## Overall class distribution (BGE, 15,817 actions)

| Class | n | % |
|---|---|---|
| LLM | 7,880 | 49.8% |
| Multimodal_Perception | 3,241 | 20.5% |
| Embodied | 2,598 | 16.4% |
| Human_Bound | 2,098 | 13.3% |

Median classifier confidence: 0.597; mean: 0.608.

## Cross-tab — intel share within each K=7 macro (%)

| Macro | LLM | MP | Embodied | Human_Bound |
|---|---|---|---|---|
| **M1** | 31% | 17% | 47% | 5% |
| **M2** | 15% | 27% | 56% | 2% |
| **M4** | 51% | 37% | 5% | 7% |
| **M5** | 89% | 7% | 1% | 4% |
| **M6** | 41% | 10% | 6% | 43% |
| **M7** | 95% | 1% | 3% | 1% |
| **Noise** | 52% | 24% | 15% | 9% |

## Cross-tab — intel share with M4 split (K=12 sub-macros)

| Macro | LLM | MP | Embodied | Human_Bound |
|---|---|---|---|---|
| **M2** | 15% | 27% | 56% | 2% |
| **M4-HVAC** | 7% | 83% | 9% | 1% |
| **M4-patrol** | 37% | 30% | 10% | 23% |
| **M1** | 31% | 17% | 47% | 5% |
| **M6** | 41% | 10% | 6% | 43% |
| **Noise** | 52% | 24% | 15% | 9% |
| **M5** | 89% | 7% | 1% | 4% |
| **M4-data** | 87% | 12% | 0% | 1% |
| **M7** | 95% | 1% | 3% | 1% |

## DWA-level: dominant intel type predicts OAI

Each DWA is assigned the modal intelligence type of its actions.

| Class | n_DWAs | mean OAI | median | std |
|---|---|---|---|---|
| **LLM** | 1070 | 0.427 | 0.5 | 0.2828 |
| **Multimodal_Perception** | 331 | 0.1245 | 0.0 | 0.2194 |
| **Embodied** | 322 | 0.0602 | 0.0 | 0.1662 |
| **Human_Bound** | 238 | 0.2193 | 0.0 | 0.2784 |

**Kruskal-Wallis**: H = 527.645, df = 3, p = 4.866e-114

**LLM-class mean OAI** = 0.4270; **other-3 mean** = 0.1266; **ratio = 3.37×**

### Pairwise Mann-Whitney + Bonferroni (6 pairs)

| a | b | p_bonf | sig |
|---|---|---|---|
| LLM | Multimodal_Perception | 1.08e-55 | **✓** |
| LLM | Embodied | 1.76e-77 | **✓** |
| LLM | Human_Bound | 1.20e-20 | **✓** |
| Multimodal_Perception | Embodied | 2.86e-05 | **✓** |
| Multimodal_Perception | Human_Bound | 2.04e-04 | **✓** |
| Embodied | Human_Bound | 7.43e-15 | **✓** |
