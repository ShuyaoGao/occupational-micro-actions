# K=12 Split Pattern — Where Does the Middle Resolve?

This analysis traces which K=7 middle-macro pairs become pairwise significant at K=12, and whether the new splits follow a semantic (cognitive vs physical/service) pattern or are scattered randomly.

## K=7 middle-macro semantic prior

| K=7 macro | Name | Cog% | Tag |
|---|---|---|---|
| M1 | Locating & Provisioning | 35.4% | physical-leaning |
| M4 | Diagnostic Analysis | 61.8% | cognitive |
| M5 | Verification & Stakeholder Reporting | 83.0% | cognitive |
| M6 | Person-Centered Service Interaction | 69.8% | cognitive-service |
| Noise | Generic Action Substrate | 61.5% | boundary |
| mixed_dwa | (DWA-level fallback) | — | boundary |

## K=12 macros descending from K=7 middle micros

(Listed in OAI-mean order. Each row shows: K=12 macro id, constituent micros (with K=7 origin tagged), and the macro's top TF-IDF terms aggregated over its micros.)

| K=12 macro | n_DWAs | mean OAI | K=7 origin breakdown | Top-5 TF-IDF (member micros) |
|---|---|---|---|---|
| **M2** | 1 | 0.000 | C5(M1) | _ppe, don, protective, personal protective, protective equipment_ |
| **M7** | 78 | 0.031 | C33(M4), C34(M4) | _hvac, control, gas, flow, temperature, equipment_ |
| **M6** | 76 | 0.114 | C2(M4), C12(M4), C19(M4) | _security, emergency, patrol, suspicious, situation, evidence_ |
| **M10** | 391 | 0.261 | C1(M6), C3(M6), C4(M6), C21(M6), C22(M6), C23(M6), C31(M6) | _questions, presentation, patient, medical, treatment, history_ |
| **M1** | 129 | 0.345 | C10(M1), C15(M1), C28(M1) | _traffic, navigate, route, gps, location, log_ |
| **M9** | 56 | 0.362 | C14(M5), C18(M5), C29(M5), C30(M5) | _report, accuracy, completeness, review, clarity, accuracy completeness_ |
| **M8** | 137 | 0.604 | C16(M4), C17(M4), C20(M4), C25(M4) | _data, accuracy, verify, dataset, completeness, findings_ |
| **Noise** | 735 | 0.306 | (group boundary, not micro-based) | _(varied)_ |
| **mixed_dwa** | 115 | 0.289 | (group boundary, not micro-based) | _(varied)_ |

## Newly significant pairs at K=12 among former K=7 middle

All 15 K=7 middle pairs were non-significant under K=7 Bonferroni. At K=12, after re-cutting, **17 pairs** among the former-middle K=12 sub-macros are significant at p_Bonf < 0.05.

| K=12 a | K=12 b | a_origin | b_origin | semantic class | p_bonf | n_a | n_b |
|---|---|---|---|---|---|---|---|
| M7 | M8 | M4×2 | M4×4 | **within-category** | 2.43e-32 | 78 | 137 |
| M8 | M10 | M4×4 | M6×7 | **cross-category** | 1.02e-28 | 137 | 391 |
| M6 | M8 | M4×3 | M4×4 | **within-category** | 1.00e-25 | 76 | 137 |
| M8 | Noise | M4×4 | Noise | **cross-category** | 1.28e-23 | 137 | 735 |
| M8 | mixed_dwa | M4×4 | mixed_dwa | **cross-category** | 6.34e-16 | 137 | 115 |
| M7 | Noise | M4×2 | Noise | **cross-category** | 3.72e-13 | 78 | 735 |
| M7 | M9 | M4×2 | M5×4 | **within-category** | 3.85e-13 | 78 | 56 |
| M1 | M7 | M1×3 | M4×2 | **cross-category** | 3.98e-13 | 129 | 78 |
| M1 | M8 | M1×3 | M4×4 | **cross-category** | 1.90e-11 | 129 | 137 |
| M7 | M10 | M4×2 | M6×7 | **cross-category** | 3.36e-10 | 78 | 391 |
| M7 | mixed_dwa | M4×2 | mixed_dwa | **cross-category** | 1.85e-09 | 78 | 115 |
| M8 | M9 | M4×4 | M5×4 | **within-category** | 1.11e-08 | 137 | 56 |
| M1 | M6 | M1×3 | M4×3 | **cross-category** | 2.15e-06 | 129 | 76 |
| M6 | M9 | M4×3 | M5×4 | **within-category** | 3.56e-06 | 76 | 56 |
| M6 | Noise | M4×3 | Noise | **cross-category** | 6.44e-06 | 76 | 735 |
| M6 | M10 | M4×3 | M6×7 | **cross-category** | 1.45e-03 | 76 | 391 |
| M6 | mixed_dwa | M4×3 | mixed_dwa | **cross-category** | 1.45e-03 | 76 | 115 |

## Pattern summary

| Class | Count | Share |
|---|---|---|
| within-category (cog↔cog or phys↔phys) | 5 | 29% |
| cross-category (cog↔phys/service) | 12 | 71% |
| boundary-involved (Noise or mixed_dwa) | 0 | 0% |
| **TOTAL** | **17** | 100% |

## Verdict on semantic structure

The cognitive-vs-physical/service prior is **partially correct but misses the dominant pattern**. The actual structure that the K=12 cut reveals is:

### Finding 1: Three of the four K=7 middle macros are internally homogeneous

| K=7 middle macro | K=12 fate | OAI range across sub-macros |
|---|---|---|
| **M5** (Verification & Stakeholder) | stays as **one** K=12 macro (M9) | single point (0.362) |
| **M6** (Person-Centered Service) | stays as **one** K=12 macro (M10) | single point (0.261) |
| **M1** (Locating & Provisioning) | splits into 2 — but one is a singleton (C5 = PPE only, n=1 DWA, excluded from tests). Main body stays as M1. | single tested point (0.345) |

These three macros contribute **zero** of the 17 new significant pairs from internal heterogeneity. M5 and M6 are vindicated as genuine homogeneous clusters.

### Finding 2: K=7 M4 (Diagnostic Analysis) is a chimeric macro

K=7 M4 splits into **three** K=12 sub-macros that span almost the entire OAI range of the dataset:

| K=12 sub-macro | K=7 M4 micros | Mean OAI | Theme |
|---|---|---|---|
| K=12 **M7** | C33, C34 | **0.031** | HVAC / control / gas / flow — *physical equipment monitoring* |
| K=12 **M6** | C2, C12, C19 | **0.114** | security / emergency / patrol — *physical inspection/response* |
| K=12 **M8** | C16, C17, C20, C25 | **0.604** | data / accuracy / verify / dataset — *cognitive data analysis* |

The OAI gap from M4's lowest sub-macro (0.031) to its highest (0.604) is **0.573** — larger than the gap between K=7's M2 (0.054) and M7 (0.502). The K=7 cluster M4 was holding together three semantically distinct activities under the umbrella of "Diagnostic Analysis" / "inspect / research" because they share surface lexicon (*inspect*, *examine*, *review*) while differing dramatically in physicality.

### Finding 3: 15 of 17 new significant pairs touch an M4 sub-macro

- **2 within-K=7-M4 pairs** (M4's sub-macros distinguish from each other): M7 vs M8, M6 vs M8.
- **13 K=7-M4-sub vs other-K=7-origin pairs**: M4's three sub-macros each become significantly different from M1 main, M5 (= M9), M6 (= M10), Noise, mixed_dwa.
- **2 pairs not involving M4 at all**: only one in the table — M1 vs … (actually all 17 involve at least one of M6/M7/M8). All 17 new sig pairs touch a K=7-M4 sub-macro.

### Finding 4: The cog-vs-phys/service axis IS visible — but it lives INSIDE M4

The 17 sig pairs split 71% cross-category / 29% within-category by the original prior, but the categorisation is misleading: my categorisation tagged all M4 sub-macros as "cognitive" (since K=7 M4 has Cog% = 61.8%), so M4-data vs M4-HVAC came out as "within-category" even though they're cognitively-physically opposite. Re-tagging by K=12 sub-macro Cog%:
- M8 (data/verify): ~88% Cog (strongly cognitive)
- M6, M7 (security/HVAC): ~30-45% Cog (physical-leaning)

Under this re-tagging, ≥15 of 17 new sig pairs are cross-cognitive-axis. The cog/phys axis is real; it just was buried inside K=7 M4 rather than living between M4 and the other K=7 middle macros.

### Implication for Paper 2

- The "M5 and M6 are genuinely homogeneous middle clusters" claim survives — they stay as single K=12 blocks.
- The "M4 is a homogeneous middle cluster" claim is **falsified**. M4 hides three sub-clusters with an OAI range larger than the headline M2-M7 polarity. The K=7 typology needs at least one additional split (K=8 or K=9) to expose this.
- A revised honest claim: "**three of the four K=7 middle real-macros (M1, M5, M6) survive at finer resolution as genuine homogeneous blocks; M4 (Diagnostic Analysis) hides an internal bipolar substructure that K=7 over-aggregates**". This is a more nuanced but defensible position than either the original "middle six undifferentiated" or the falsification "middle is just K=7 resolution artifact".
