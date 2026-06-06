"""
Direction 5 — TOST equivalence test on K=7 middle six groups.

Computes the equivalence margin BEFORE any TOST is run (locked).
Margin 1 (primary): Cohen's d = 0.2 on pooled SD of middle six OAI.
Margin 2 (robustness): ±0.05 OAI absolute.

Tests all 15 pairs among (M1, M4, M5, M6, Noise, mixed_dwa).
Sanity-check pair: M2 vs M7 (should NOT be equivalent under any margin).
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttost_ind

SEED = 42
np.random.seed(SEED)

def _find_root(p):
    import os as _o
    p = _o.path.abspath(p)
    while p != _o.path.dirname(p):
        if _o.path.isdir(_o.path.join(p, "shared")):
            return p
        p = _o.path.dirname(p)
    raise RuntimeError("repo root (with shared/) not found")
PROJ = _find_root(__file__)

SHARED = os.path.join(PROJ, "shared")
BIPOLAR = os.path.join(PROJ, "paper2_bipolar")
P7   = os.path.join(BIPOLAR, "data", "Part7_outputs", "outputs")
OUT  = os.path.join(BIPOLAR, "hardening", "outputs")
os.makedirs(OUT, exist_ok=True)

MIDDLE = ["M1", "M4", "M5", "M6", "Noise", "mixed_dwa"]
POLES  = ["M2", "M7"]


def pooled_sd(groups_arrays):
    """Pooled within-group SD across multiple groups."""
    total_ss = 0.0
    total_df = 0
    for arr in groups_arrays:
        arr = np.asarray(arr)
        n = len(arr)
        if n < 2:
            continue
        total_ss += (n - 1) * arr.var(ddof=1)
        total_df += (n - 1)
    return float(np.sqrt(total_ss / total_df))


def main():
    print(">>> Loading data ...")
    df = pd.read_csv(os.path.join(P7, "dwa_macro_distribution_with_oai.csv"))
    df = df[["DWA_ID", "analysis_group", "Automation_Index"]].dropna()
    df = df[df["analysis_group"].isin(MIDDLE + POLES)].copy()

    # ----- Compute and LOCK equivalence margins -----
    middle_arrays = [df.loc[df["analysis_group"] == g, "Automation_Index"].values
                     for g in MIDDLE]
    sigma_pool = pooled_sd(middle_arrays)
    delta_d02  = round(0.2 * sigma_pool, 4)     # primary
    delta_abs  = 0.05                            # robustness

    print(f"   Pooled SD (middle six): σ_pool = {sigma_pool:.4f}")
    print(f"   PRIMARY equivalence margin:    δ_d02 = 0.2 × σ_pool = {delta_d02}")
    print(f"   ROBUSTNESS equivalence margin: δ_abs = ±0.05")
    print()

    # ----- Run TOST for each pair under each margin -----
    rows = []
    # 15 middle pairs + 1 sanity-check (M2 vs M7)
    pairs = []
    for i, a in enumerate(MIDDLE):
        for b in MIDDLE[i+1:]:
            pairs.append((a, b, "middle"))
    pairs.append(("M2", "M7", "extreme_sanity"))

    for a, b, kind in pairs:
        xa = df.loc[df["analysis_group"] == a, "Automation_Index"].values
        xb = df.loc[df["analysis_group"] == b, "Automation_Index"].values
        if len(xa) < 2 or len(xb) < 2:
            rows.append({"a": a, "b": b, "kind": kind,
                         "n_a": len(xa), "n_b": len(xb),
                         "mean_a": None, "mean_b": None})
            continue
        mean_a = xa.mean(); mean_b = xb.mean()
        diff   = mean_a - mean_b

        # Primary margin (Cohen's d = 0.2 on σ_pool)
        p_d02, t1_d02, t2_d02 = ttost_ind(xa, xb, low=-delta_d02, upp=+delta_d02,
                                           usevar="pooled")
        equiv_d02 = bool(p_d02 < 0.05)

        # Robustness margin (±0.05 OAI absolute)
        p_abs, t1_abs, t2_abs = ttost_ind(xa, xb, low=-delta_abs, upp=+delta_abs,
                                           usevar="pooled")
        equiv_abs = bool(p_abs < 0.05)

        rows.append({
            "a": a, "b": b, "kind": kind,
            "n_a": int(len(xa)), "n_b": int(len(xb)),
            "mean_a": round(float(mean_a), 4),
            "mean_b": round(float(mean_b), 4),
            "diff":   round(float(diff), 4),
            "p_TOST_d02": float(p_d02),
            "equiv_d02":  equiv_d02,
            "p_TOST_abs": float(p_abs),
            "equiv_abs":  equiv_abs,
        })

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT, "tost_results.csv"), index=False,
                   encoding="utf-8-sig")
    print(results.to_string())
    print()

    # ----- Sanity check verdict -----
    sanity = results[results["kind"] == "extreme_sanity"].iloc[0]
    sanity_ok = (not sanity["equiv_d02"]) and (not sanity["equiv_abs"])
    print(f">>> Sanity check (M2 vs M7 should be NON-equivalent):")
    print(f"     equiv_d02 = {sanity['equiv_d02']}, equiv_abs = {sanity['equiv_abs']}")
    print(f"     Sanity {'PASSED' if sanity_ok else 'FAILED'}")
    print()

    # ----- Middle equivalence count -----
    middle = results[results["kind"] == "middle"]
    n_equiv_d02 = int(middle["equiv_d02"].sum())
    n_equiv_abs = int(middle["equiv_abs"].sum())
    print(f">>> Middle equivalence summary:")
    print(f"     d=0.2 margin: {n_equiv_d02} / 15 pairs equivalent")
    print(f"     ±0.05 margin: {n_equiv_abs} / 15 pairs equivalent")
    print()

    # ----- Markdown report -----
    lines = ["# TOST Equivalence Test — K=7 Middle Six Groups", ""]
    lines.append("Locks 'failed to reject H0' (absence of evidence) into 'evidence of equivalence' "
                 "via two one-sided tests (TOST).")
    lines.append("")

    lines.append("## Equivalence margins (LOCKED before execution)")
    lines.append("")
    lines.append(f"- **Primary (Cohen's d = 0.2)**: pooled SD across middle six "
                 f"σ_pool = **{sigma_pool:.4f}**; therefore δ_d02 = 0.2 × σ_pool = **{delta_d02}** "
                 "on the OAI [0, 1] scale.")
    lines.append(f"- **Robustness (absolute)**: δ_abs = **±0.05** OAI. "
                 "(Smaller than the smallest Tech-Risk matrix step of 0.2, so |Δ| < 0.05 "
                 "cannot move a DWA across an OAI level boundary.)")
    lines.append("")

    lines.append("## Sanity check: M2 vs M7 (extreme pair)")
    lines.append("")
    if sanity_ok:
        lines.append("✓ **PASSED** — M2 vs M7 is NON-equivalent under both margins, "
                     "as expected. TOST setup is valid.")
    else:
        lines.append("✗ **FAILED** — M2 vs M7 came out as equivalent under at least one "
                     "margin. The TOST setup is broken; results below are not trustworthy.")
    lines.append("")
    lines.append(f"| pair | n_a | n_b | mean_a | mean_b | diff | p_TOST_d02 | equiv_d02 | p_TOST_abs | equiv_abs |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|")
    s = sanity
    lines.append(f"| {s['a']} vs {s['b']} | {s['n_a']} | {s['n_b']} | "
                 f"{s['mean_a']} | {s['mean_b']} | {s['diff']} | "
                 f"{s['p_TOST_d02']:.3e} | {s['equiv_d02']} | "
                 f"{s['p_TOST_abs']:.3e} | {s['equiv_abs']} |")
    lines.append("")

    lines.append("## Middle six pairs (15 pairs)")
    lines.append("")
    lines.append(f"- **Equivalent at primary margin (d=0.2)**: **{n_equiv_d02} / 15**")
    lines.append(f"- **Equivalent at robustness margin (±0.05)**: **{n_equiv_abs} / 15**")
    lines.append("")
    lines.append(f"| pair | n_a | n_b | mean_a | mean_b | diff | p_TOST_d02 | equiv_d02 | p_TOST_abs | equiv_abs |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|")
    for _, r in middle.sort_values("p_TOST_d02").iterrows():
        eq_d_str  = "✓" if r["equiv_d02"] else "✗"
        eq_a_str  = "✓" if r["equiv_abs"] else "✗"
        lines.append(f"| {r['a']} vs {r['b']} | {r['n_a']} | {r['n_b']} | "
                     f"{r['mean_a']} | {r['mean_b']} | {r['diff']} | "
                     f"{r['p_TOST_d02']:.3e} | {eq_d_str} | "
                     f"{r['p_TOST_abs']:.3e} | {eq_a_str} |")
    lines.append("")

    lines.append("## Decision (per locked criteria in PLAN.md)")
    lines.append("")
    if n_equiv_d02 >= 10:
        verdict = (f"✓ **Strong support for 'middle band is genuinely undifferentiated, "
                   f"not under-powered'**: {n_equiv_d02}/15 pairs are statistically equivalent "
                   f"at the d=0.2 margin.")
    elif n_equiv_d02 <= 4:
        verdict = (f"✗ **Insufficient evidence for equivalence**: only {n_equiv_d02}/15 pairs "
                   f"reach equivalence at d=0.2. Main claim must be qualified to 'failed to "
                   f"reject H0' rather than 'positive evidence of equivalence'.")
    else:
        verdict = (f"⚠ **Mixed evidence**: {n_equiv_d02}/15 pairs equivalent at d=0.2 "
                   f"(falls between 5 and 9). Honest reporting required.")
    lines.append(verdict)
    lines.append("")
    lines.append(f"At the stricter robustness margin (±0.05 OAI), {n_equiv_abs}/15 pairs equivalent.")

    with open(os.path.join(OUT, "tost_equivalence.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> tost_equivalence.md\n[Done]")


if __name__ == "__main__":
    main()
