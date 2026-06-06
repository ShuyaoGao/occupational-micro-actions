"""
Direction 4 — resolution sweep K = 7, 8, 10, 12, 15.

Reuses 35 micro-clusters and Ward linkage; cuts at multiple K values;
projects OAI onto each; counts how many middle-group pairs remain
non-significant after Bonferroni.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from scipy.cluster.hierarchy import fcluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# K=7 reference for pole-stability check
K7_M2_MICROS = [6, 7, 8, 24, 26]   # Tool-Mediated Physical Execution (low pole)
K7_M7_MICROS = [0, 11, 13, 27, 32] # Planning & Design (high pole)
MIXED_THRESHOLD = 0.40


def assign_dominant_macro(actions, cid_to_macro):
    """For each DWA, compute its action distribution over macro labels and
    return dominant macro (or 'mixed_dwa' if max share < 0.40)."""
    actions = actions.copy()
    def label(cid):
        if cid < 0:
            return "Noise"
        return f"M{cid_to_macro[int(cid)]}"
    actions["macro_label"] = actions["cluster_id"].apply(label)

    rows = []
    for dwa_id, sub in actions.groupby("DWA_ID", sort=False):
        counts = Counter(sub["macro_label"])
        total = len(sub)
        dom = max(counts.keys(), key=lambda k: counts[k])
        dom_share = counts[dom] / total
        analysis_group = "mixed_dwa" if dom_share < MIXED_THRESHOLD else dom
        rows.append({"DWA_ID": dwa_id,
                     "dominant_macro": dom,
                     "dominant_share": dom_share,
                     "analysis_group": analysis_group})
    return pd.DataFrame(rows)


def pairwise_mw_bonf(df, score_col, group_col, present_groups):
    """Returns DataFrame of (a, b, p_raw, p_bonf, sig)."""
    n_pairs = len(present_groups) * (len(present_groups) - 1) // 2
    rows = []
    for i, g1 in enumerate(present_groups):
        for g2 in present_groups[i+1:]:
            a = df.loc[df[group_col] == g1, score_col].dropna().values
            b = df.loc[df[group_col] == g2, score_col].dropna().values
            if len(a) < 2 or len(b) < 2:
                rows.append({"a": g1, "b": g2, "n_a": len(a), "n_b": len(b),
                             "p_raw": None, "p_bonf": None, "sig": False})
                continue
            try:
                U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                p_bonf = min(1.0, p * n_pairs)
                rows.append({"a": g1, "b": g2,
                             "n_a": int(len(a)), "n_b": int(len(b)),
                             "p_raw": float(p), "p_bonf": float(p_bonf),
                             "sig": bool(p_bonf < 0.05)})
            except Exception:
                rows.append({"a": g1, "b": g2, "n_a": len(a), "n_b": len(b),
                             "p_raw": None, "p_bonf": None, "sig": False})
    return pd.DataFrame(rows), n_pairs


def run_one_K(K, Z, cluster_ids, actions, dwa_oai):
    print(f"\n>>> K = {K}")
    # 1. Cut the linkage tree at K
    labels_micro = fcluster(Z, t=K, criterion="maxclust")  # shape (35,)
    cid_to_macro = {int(cid): int(m) for cid, m in zip(cluster_ids, labels_micro)}
    n_macros = len(set(labels_micro))
    print(f"   micro_id → macro_id: {n_macros} macros total")

    # 2. Each action gets a macro label
    # 3. Each DWA → dominant_macro / mixed_dwa
    dwa_dom = assign_dominant_macro(actions, cid_to_macro)

    # Merge with OAI
    merged = dwa_dom.merge(dwa_oai[["DWA_ID", "Automation_Index"]],
                            on="DWA_ID", how="left")
    merged = merged.dropna(subset=["Automation_Index"])

    # 4. Drop singleton macros with 0 DWAs as dominant
    dom_counts = merged["analysis_group"].value_counts()
    present_groups = [g for g, n in dom_counts.items() if n >= 2]
    # Stable sort: M-numeric first by number, then Noise, mixed_dwa
    def sort_key(g):
        if g == "Noise":    return (1, 0)
        if g == "mixed_dwa":return (2, 0)
        try:
            return (0, int(g[1:]))
        except Exception:
            return (3, 0)
    present_groups = sorted(present_groups, key=sort_key)

    # 5. Kruskal-Wallis across all present groups
    samples = [merged.loc[merged["analysis_group"] == g, "Automation_Index"].values
               for g in present_groups]
    H, p_kw = stats.kruskal(*samples)

    # 6. Pairwise MW + Bonferroni
    pw, n_pairs = pairwise_mw_bonf(merged, "Automation_Index",
                                    "analysis_group", present_groups)

    # 7. Identify poles among "real" macros (exclude Noise, mixed_dwa)
    real_macros = [g for g in present_groups if g not in ("Noise", "mixed_dwa")]
    means = {g: merged.loc[merged["analysis_group"] == g, "Automation_Index"].mean()
             for g in real_macros}
    low_pole  = min(means, key=means.get)
    high_pole = max(means, key=means.get)

    # 8. Middle = all groups EXCEPT the two poles
    middle = [g for g in present_groups if g not in (low_pole, high_pole)]
    middle_pairs = pw[(pw["a"].isin(middle)) & (pw["b"].isin(middle))]
    n_middle_pairs = len(middle_pairs)
    n_middle_nonsig = (~middle_pairs["sig"]).sum()
    middle_nonsig_frac = n_middle_nonsig / max(1, n_middle_pairs)

    # 9. Pole stability: which K=7-M2 micros are in the low pole? which K=7-M7 in high?
    low_pole_macro_id  = int(low_pole[1:])
    high_pole_macro_id = int(high_pole[1:])
    low_pole_micros  = [cid for cid in cluster_ids if cid_to_macro[cid] == low_pole_macro_id]
    high_pole_micros = [cid for cid in cluster_ids if cid_to_macro[cid] == high_pole_macro_id]
    m2_in_low  = [c for c in K7_M2_MICROS if c in low_pole_micros]
    m7_in_high = [c for c in K7_M7_MICROS if c in high_pole_micros]
    m2_overlap = len(m2_in_low)  / len(K7_M2_MICROS)
    m7_overlap = len(m7_in_high) / len(K7_M7_MICROS)

    print(f"   present groups: {len(present_groups)}  ({present_groups})")
    print(f"   H = {H:.2f}, p_KW = {p_kw:.2e}")
    print(f"   low pole: {low_pole} (mean OAI {means[low_pole]:.3f})")
    print(f"   high pole: {high_pole} (mean OAI {means[high_pole]:.3f})")
    print(f"   middle pairs: {n_middle_pairs}, non-sig: {n_middle_nonsig} ({middle_nonsig_frac*100:.1f}%)")
    print(f"   M2 overlap with low pole: {len(m2_in_low)}/{len(K7_M2_MICROS)} = {m2_overlap:.0%}")
    print(f"   M7 overlap with high pole: {len(m7_in_high)}/{len(K7_M7_MICROS)} = {m7_overlap:.0%}")

    return {
        "K": K,
        "n_present_groups": len(present_groups),
        "present_groups": present_groups,
        "low_pole": low_pole,
        "high_pole": high_pole,
        "low_pole_mean": float(means[low_pole]),
        "high_pole_mean": float(means[high_pole]),
        "H": float(H),
        "p_kw": float(p_kw),
        "n_total_pairs": len(pw),
        "n_total_sig": int(pw["sig"].sum()),
        "n_middle_pairs": int(n_middle_pairs),
        "n_middle_nonsig": int(n_middle_nonsig),
        "middle_nonsig_frac": float(middle_nonsig_frac),
        "m2_in_low_pole": m2_in_low,
        "m7_in_high_pole": m7_in_high,
        "m2_overlap_frac": float(m2_overlap),
        "m7_overlap_frac": float(m7_overlap),
        "pw_df": pw,
        "merged": merged,
    }


def main():
    print(">>> Loading data ...")
    Z = np.load(os.path.join(P7, "macro_linkage_matrix.npy"))
    cluster_ids = list(range(35))  # 0..34
    actions = pd.read_csv(os.path.join(P7, "actions_with_clusters.csv"))
    dwa_oai = pd.read_csv(os.path.join(P7, "dwa_macro_distribution_with_oai.csv"))

    print(f"   Z shape: {Z.shape}")
    print(f"   actions: {len(actions):,}")
    print(f"   DWAs with OAI: {len(dwa_oai):,}")

    results = []
    for K in [7, 8, 10, 12, 15]:
        r = run_one_K(K, Z, cluster_ids, actions, dwa_oai)
        results.append(r)

        # Save per-K macro assignments
        pd.DataFrame({"micro_id": cluster_ids,
                      "macro_id": [int(m) for m in fcluster(Z, t=K, criterion="maxclust")]
                      }).to_csv(os.path.join(OUT, f"macro_assignments_K{K}.csv"),
                                index=False)

        # Save per-K pairwise full table
        r["pw_df"].to_csv(os.path.join(OUT, f"pairwise_K{K}.csv"),
                          index=False, encoding="utf-8-sig")

    # Build summary DF
    summary = pd.DataFrame([{
        "K": r["K"],
        "n_present_groups": r["n_present_groups"],
        "low_pole": r["low_pole"], "low_pole_mean_OAI": round(r["low_pole_mean"], 3),
        "high_pole": r["high_pole"], "high_pole_mean_OAI": round(r["high_pole_mean"], 3),
        "H": round(r["H"], 2),
        "p_KW": r["p_kw"],
        "n_total_pairs": r["n_total_pairs"],
        "n_total_sig": r["n_total_sig"],
        "n_middle_pairs": r["n_middle_pairs"],
        "n_middle_nonsig": r["n_middle_nonsig"],
        "middle_nonsig_pct": round(r["middle_nonsig_frac"] * 100, 1),
        "m2_overlap_pct": round(r["m2_overlap_frac"] * 100, 0),
        "m7_overlap_pct": round(r["m7_overlap_frac"] * 100, 0),
    } for r in results])
    summary.to_csv(os.path.join(OUT, "resolution_sweep_summary.csv"),
                   index=False, encoding="utf-8-sig")
    print("\n>>> Summary:")
    print(summary.to_string())

    # Plot: x = K, y = middle non-sig %
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    Ks = [r["K"] for r in results]
    ys = [r["middle_nonsig_frac"] * 100 for r in results]
    ax.plot(Ks, ys, marker="o", linewidth=2.4, color="#1f77b4",
            markersize=9, markerfacecolor="white", markeredgewidth=2)
    for k, y in zip(Ks, ys):
        ax.annotate(f"{y:.1f}%", (k, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.axhline(70, color="#888888", linestyle=":", alpha=0.7)
    ax.text(15.2, 71, "70% (claim-support floor)",
            fontsize=9, color="#666666", va="bottom")
    ax.axhline(30, color="#888888", linestyle=":", alpha=0.7)
    ax.text(15.2, 31, "30% (claim-failure ceiling)",
            fontsize=9, color="#666666", va="bottom")
    ax.set_xticks(Ks)
    ax.set_xlabel("K (number of macros from Ward cut)")
    ax.set_ylabel("Middle-group pairs non-significant after Bonferroni (%)")
    ax.set_title("Resolution sweep: does 'middle indistinguishability' survive finer K?")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "resolution_sweep_plot.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> resolution_sweep_plot.png")

    # Markdown report
    lines = ["# Resolution Sweep — K = 7, 8, 10, 12, 15", ""]
    lines.append("Tests whether the K=7 'middle six non-distinguishable' finding "
                 "survives finer Ward cuts, or is an artifact of coarse resolution.")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| K | n_groups | low pole (mean OAI) | high pole (mean OAI) | "
                 "H | p_KW | Middle pairs non-sig (Bonferroni) | M2 micros in low pole | M7 micros in high pole |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| **{r['K']}** | {r['n_present_groups']} | "
            f"{r['low_pole']} ({r['low_pole_mean']:.3f}) | "
            f"{r['high_pole']} ({r['high_pole_mean']:.3f}) | "
            f"{r['H']:.2f} | {r['p_kw']:.2e} | "
            f"**{r['n_middle_nonsig']} / {r['n_middle_pairs']} = {r['middle_nonsig_frac']*100:.1f}%** | "
            f"{len(r['m2_in_low_pole'])} / 5 ({r['m2_overlap_frac']*100:.0f}%) | "
            f"{len(r['m7_in_high_pole'])} / 5 ({r['m7_overlap_frac']*100:.0f}%) |"
        )
    lines.append("")

    # Pole stability section
    lines.append("## Pole stability (which K=7 pole micros land in each K's pole macro)")
    lines.append("")
    lines.append("**K=7 reference**: M2 micros = " + ", ".join(f"C{c}" for c in K7_M2_MICROS) +
                 "; M7 micros = " + ", ".join(f"C{c}" for c in K7_M7_MICROS))
    lines.append("")
    for r in results:
        lines.append(f"- **K={r['K']}**: low pole `{r['low_pole']}` "
                     f"contains M2-micros {sorted(r['m2_in_low_pole'])}; "
                     f"high pole `{r['high_pole']}` contains M7-micros {sorted(r['m7_in_high_pole'])}")
    lines.append("")

    # Decision section
    final = results[-1]
    k12 = next(r for r in results if r["K"] == 12)
    k15 = next(r for r in results if r["K"] == 15)
    lines.append("## Decision (per locked criteria in PLAN.md)")
    lines.append("")
    lines.append(f"- Middle non-sig fraction at K=12: **{k12['middle_nonsig_frac']*100:.1f}%**")
    lines.append(f"- Middle non-sig fraction at K=15: **{k15['middle_nonsig_frac']*100:.1f}%**")
    lines.append("")
    if k12["middle_nonsig_frac"] >= 0.70 and k15["middle_nonsig_frac"] >= 0.70:
        verdict = "✓ **Middle indistinguishability is a true structural feature.** It survives at finer resolution."
    elif k12["middle_nonsig_frac"] <= 0.30 or k15["middle_nonsig_frac"] <= 0.30:
        verdict = "✗ **Middle indistinguishability is largely a K=7 resolution artifact.** Main claim must be qualified."
    else:
        verdict = "⚠ **Gray zone**: middle structure is partly resolution-dependent. Honest reporting required in the paper."
    lines.append(verdict)
    lines.append("")
    lines.append("![Resolution sweep plot](resolution_sweep_plot.png)")
    lines.append("")

    with open(os.path.join(OUT, "resolution_sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> resolution_sweep.md\n[Done]")


if __name__ == "__main__":
    main()
