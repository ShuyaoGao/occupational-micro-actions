# TOST Equivalence Test — K=7 Middle Six Groups

Locks 'failed to reject H0' (absence of evidence) into 'evidence of equivalence' via two one-sided tests (TOST).

## Equivalence margins (LOCKED before execution)

- **Primary (Cohen's d = 0.2)**: pooled SD across middle six σ_pool = **0.2997**; therefore δ_d02 = 0.2 × σ_pool = **0.0599** on the OAI [0, 1] scale.
- **Robustness (absolute)**: δ_abs = **±0.05** OAI. (Smaller than the smallest Tech-Risk matrix step of 0.2, so |Δ| < 0.05 cannot move a DWA across an OAI level boundary.)

## Sanity check: M2 vs M7 (extreme pair)

✓ **PASSED** — M2 vs M7 is NON-equivalent under both margins, as expected. TOST setup is valid.

| pair | n_a | n_b | mean_a | mean_b | diff | p_TOST_d02 | equiv_d02 | p_TOST_abs | equiv_abs |
|---|---|---|---|---|---|---|---|---|---|
| M2 vs M7 | 170 | 99 | 0.0535 | 0.499 | -0.4455 | 1.000e+00 | False | 1.000e+00 | False |

## Middle six pairs (15 pairs)

- **Equivalent at primary margin (d=0.2)**: **1 / 15**
- **Equivalent at robustness margin (±0.05)**: **0 / 15**

| pair | n_a | n_b | mean_a | mean_b | diff | p_TOST_d02 | equiv_d02 | p_TOST_abs | equiv_abs |
|---|---|---|---|---|---|---|---|---|---|
| M4 vs Noise | 324 | 685 | 0.3228 | 0.3057 | 0.0171 | 1.951e-02 | ✓ | 5.629e-02 | ✗ |
| M1 vs M4 | 132 | 324 | 0.3424 | 0.3228 | 0.0196 | 1.050e-01 | ✗ | 1.721e-01 | ✗ |
| M6 vs mixed_dwa | 396 | 96 | 0.2606 | 0.2823 | -0.0217 | 1.222e-01 | ✗ | 1.941e-01 | ✗ |
| Noise vs mixed_dwa | 685 | 96 | 0.3057 | 0.2823 | 0.0234 | 1.341e-01 | ✗ | 2.098e-01 | ✗ |
| M1 vs Noise | 132 | 685 | 0.3424 | 0.3057 | 0.0367 | 2.101e-01 | ✗ | 3.221e-01 | ✗ |
| M6 vs Noise | 396 | 685 | 0.2606 | 0.3057 | -0.0451 | 2.144e-01 | ✗ | 3.965e-01 | ✗ |
| M1 vs M5 | 132 | 59 | 0.3424 | 0.3678 | -0.0254 | 2.241e-01 | ✗ | 2.942e-01 | ✗ |
| M4 vs mixed_dwa | 324 | 96 | 0.3228 | 0.2823 | 0.0405 | 2.968e-01 | ✗ | 3.972e-01 | ✗ |
| M4 vs M5 | 324 | 59 | 0.3228 | 0.3678 | -0.045 | 3.662e-01 | ✗ | 4.541e-01 | ✗ |
| M1 vs mixed_dwa | 132 | 96 | 0.3424 | 0.2823 | 0.0601 | 5.023e-01 | ✗ | 5.994e-01 | ✗ |
| M5 vs Noise | 59 | 685 | 0.3678 | 0.3057 | 0.0621 | 5.216e-01 | ✗ | 6.169e-01 | ✗ |
| M4 vs M6 | 324 | 396 | 0.3228 | 0.2606 | 0.0622 | 5.414e-01 | ✗ | 7.072e-01 | ✗ |
| M5 vs mixed_dwa | 59 | 96 | 0.3678 | 0.2823 | 0.0855 | 7.044e-01 | ✗ | 7.717e-01 | ✗ |
| M1 vs M6 | 132 | 396 | 0.3424 | 0.2606 | 0.0818 | 7.745e-01 | ✗ | 8.630e-01 | ✗ |
| M5 vs M6 | 59 | 396 | 0.3678 | 0.2606 | 0.1072 | 8.840e-01 | ✗ | 9.258e-01 | ✗ |

## Decision (per locked criteria in PLAN.md)

✗ **Insufficient evidence for equivalence**: only 1/15 pairs reach equivalence at d=0.2. Main claim must be qualified to 'failed to reject H0' rather than 'positive evidence of equivalence'.

At the stricter robustness margin (±0.05 OAI), 0/15 pairs equivalent.