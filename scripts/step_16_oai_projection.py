"""Step 16 - Project paper 1's DWA-level OAI onto the K=7 macro typology (Phase 5).

This is the canonical OAI projection step. It loads the DWA-level Automation
Index produced by paper 1, joins it to the K=7 macro assignments produced by
step_14 / step_15, and runs the full statistical battery from paper 2 Section 4.3.

Tasks performed (in order):
  T1 - persist the canonical K=7 macro names: macro_naming_K7.json
  T2 - build dwa_macro_distribution.csv (input dependency for T3)
  T3 - macro-level OAI distribution: oai_by_macro.md + two figures
  T4 - global Kruskal-Wallis + pairwise Mann-Whitney with Bonferroni correction
  T5 - external_validity_summary.md (Eloundou / AIOE / Frey-Osborne reproduction)
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "..", "data", "Part7_outputs", "outputs")

ACTIONS_CSV = os.path.join(_HERE, "..", "..", "..", "shared", "action_decomposition", "actions_with_clusters.csv")
OAI_CSV = os.path.join(_HERE, "..", "..", "..", "shared", "oai", "output_10_DWA_Automation_Index.csv")
_OAI_CSV_OLD = os.path.join(_HERE, "..",
                        "OAI_Project", "data", "output_10_DWA_Automation_Index.csv")

# Ground-truth mapping (user-confirmed)
MACRO_NAMING = {
    "M1": {"name": "Locating & Provisioning",
           "members": [5, 10, 15, 28]},
    "M2": {"name": "Tool-Mediated Physical Execution",
           "members": [6, 7, 8, 24, 26]},
    "M3": {"name": "Iterative Repetition (meta-action)",
           "members": [9]},
    "M4": {"name": "Diagnostic Analysis",
           "members": [2, 12, 16, 17, 19, 20, 25, 33, 34]},
    "M5": {"name": "Verification & Stakeholder Reporting",
           "members": [14, 18, 29, 30]},
    "M6": {"name": "Person-Centered Service Interaction",
           "members": [1, 3, 4, 21, 22, 23, 31]},
    "M7": {"name": "Planning & Design",
           "members": [0, 11, 13, 27, 32]},
}

GROUPS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "Noise"]
ANALYSIS_GROUPS = GROUPS + ["mixed_dwa"]
MIXED_THRESHOLD = 0.40   # dominant_share < this -> mixed_dwa


# -----------------------------------------------------------------------------
# Save naming JSON
# -----------------------------------------------------------------------------
def save_naming():
    out = os.path.join(OUT_DIR, "macro_naming_K7.json")
    payload = {
        "version":      "K7_main_analysis_v1",
        "K":            7,
        "noise_group":  {"name": "Generic Action Substrate",
                         "size": 5637, "share": 0.356,
                         "treatment": "separate analysis layer, NOT a macro"},
        "macros": {
            k: {
                "name":            v["name"],
                "member_micros":   [f"C{m}" for m in v["members"]],
                "n_micros":        len(v["members"]),
            }
            for k, v in MACRO_NAMING.items()
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"   -> {out}")


# -----------------------------------------------------------------------------
# DWA-level macro distribution
# -----------------------------------------------------------------------------
def micro_to_macro_label(cid):
    if cid is None or pd.isna(cid):
        return "Noise"
    cid = int(cid)
    if cid < 0:
        return "Noise"
    for k, v in MACRO_NAMING.items():
        if cid in v["members"]:
            return k
    return "Noise"


def shannon_entropy(probs):
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def build_dwa_distribution(actions):
    actions = actions.copy()
    actions["macro_label"] = actions["cluster_id"].apply(micro_to_macro_label)

    rows = []
    for dwa_id, sub in actions.groupby("DWA_ID", sort=False):
        counts = Counter(sub["macro_label"])
        total = len(sub)
        row = {"DWA_ID": dwa_id,
               "DWA_Title": sub["DWA_Title"].iloc[0],
               "total_steps": total}
        for g in GROUPS:
            c = counts.get(g, 0)
            row[f"{g}_count"] = c
            row[f"{g}_share"] = round(c / total, 4)
        # dominant
        dom_g = max(GROUPS, key=lambda g: counts.get(g, 0))
        dom_share = counts.get(dom_g, 0) / total
        row["dominant_macro"] = dom_g
        row["dominant_share"] = round(dom_share, 4)
        # diversity
        row["macro_diversity"] = sum(1 for g in GROUPS if counts.get(g, 0) > 0)
        # entropy (log2)
        probs = [counts.get(g, 0) / total for g in GROUPS]
        row["entropy"] = round(shannon_entropy(probs), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# OAI analysis
# -----------------------------------------------------------------------------
def assign_analysis_group(row):
    if row["dominant_share"] < MIXED_THRESHOLD:
        return "mixed_dwa"
    return row["dominant_macro"]


def per_group_oai_stats(merged, group_col="analysis_group", oai_col="Automation_Index"):
    rows = []
    overall_oai = merged[oai_col].dropna()
    overall_mean = overall_oai.mean()
    for g in ANALYSIS_GROUPS:
        sub = merged[merged[group_col] == g]
        oai = sub[oai_col].dropna()
        n = len(oai)
        if n == 0:
            rows.append({"group": g, "n_dwas": 0})
            continue
        # t-test against the rest
        rest = merged[merged[group_col] != g][oai_col].dropna()
        try:
            t_stat, p_val = stats.ttest_ind(oai, rest, equal_var=False)
        except Exception:
            t_stat, p_val = (np.nan, np.nan)
        row = {
            "group":           g,
            "n_dwas":          n,
            "oai_mean":        round(float(oai.mean()), 4),
            "oai_median":      round(float(oai.median()), 4),
            "oai_std":         round(float(oai.std()), 4),
            "oai_q25":         round(float(oai.quantile(0.25)), 4),
            "oai_q75":         round(float(oai.quantile(0.75)), 4),
            "diff_from_overall_mean": round(float(oai.mean() - overall_mean), 4),
            "t_stat_vs_rest":  round(float(t_stat), 3) if not np.isnan(t_stat) else None,
            "p_value_vs_rest": float(p_val) if not np.isnan(p_val) else None,
            "tech_mean":       round(float(sub["ai_avg_tech"].dropna().mean()), 4)
                                 if "ai_avg_tech" in sub else None,
            "risk_mean":       round(float(sub["ai_avg_risk"].dropna().mean()), 4)
                                 if "ai_avg_risk" in sub else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(a, b):
    """Cliff's delta. Range -1 to +1. >0.474 large effect."""
    a, b = np.asarray(a), np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    # Faster O(n log n) via sorting
    sorted_b = np.sort(b)
    gt = np.searchsorted(sorted_b, a, side="left").sum()       # b < a count
    lt = (n_b - np.searchsorted(sorted_b, a, side="right")).sum()  # b > a count
    return (gt - lt) / (n_a * n_b)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1))
                     / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def pairwise_mann_whitney(merged, group_col="analysis_group", oai_col="Automation_Index"):
    """Returns long-form DataFrame of all pairs (n_groups choose 2)."""
    groups_present = [g for g in ANALYSIS_GROUPS
                      if (merged[group_col] == g).sum() > 0]
    rows = []
    n_pairs = len(groups_present) * (len(groups_present) - 1) // 2
    for i, g1 in enumerate(groups_present):
        for g2 in groups_present[i + 1:]:
            a = merged.loc[merged[group_col] == g1, oai_col].dropna().values
            b = merged.loc[merged[group_col] == g2, oai_col].dropna().values
            if len(a) < 2 or len(b) < 2:
                rows.append({"group_a": g1, "group_b": g2,
                             "n_a": len(a), "n_b": len(b),
                             "U": None, "p_raw": None, "p_bonferroni": None,
                             "cliffs_delta": None,
                             "median_a": None, "median_b": None,
                             "significant": False})
                continue
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            p_bonf = min(1.0, p * n_pairs)
            d = cliffs_delta(a, b)
            rows.append({
                "group_a":      g1,
                "group_b":      g2,
                "n_a":          len(a),
                "n_b":          len(b),
                "U":            float(U),
                "p_raw":        float(p),
                "p_bonferroni": float(p_bonf),
                "cliffs_delta": round(float(d), 4),
                "median_a":     round(float(np.median(a)), 4),
                "median_b":     round(float(np.median(b)), 4),
                "significant":  bool(p_bonf < 0.05),
            })
    return pd.DataFrame(rows), n_pairs


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_oai_boxplot(merged, out_path):
    valid = merged.dropna(subset=["Automation_Index", "analysis_group"])
    # Order by mean OAI ascending
    means = valid.groupby("analysis_group")["Automation_Index"].mean().sort_values()
    order = means.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    sns.boxplot(data=valid, x="analysis_group", y="Automation_Index",
                order=order, ax=ax, palette="viridis")
    sns.stripplot(data=valid, x="analysis_group", y="Automation_Index",
                  order=order, ax=ax, color="black", size=1.5, alpha=0.25, jitter=0.25)
    # Annotate means
    for i, g in enumerate(order):
        m = means.loc[g]
        ax.text(i, m, f"μ={m:.2f}", ha="center", va="center",
                fontsize=8, color="white",
                bbox=dict(facecolor="black", alpha=0.7, pad=2, boxstyle="round,pad=0.2"))
    ax.set_xlabel("")
    ax.set_ylabel("OAI (Automation Index, 0-1)")
    ax.set_title("OAI Distribution by Macro Group (sorted by mean OAI ascending)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {out_path}")


def plot_oai_density(merged, out_path):
    valid = merged.dropna(subset=["Automation_Index", "analysis_group"])
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    palette = sns.color_palette("tab10", n_colors=len(ANALYSIS_GROUPS))
    for i, g in enumerate(ANALYSIS_GROUPS):
        sub = valid.loc[valid["analysis_group"] == g, "Automation_Index"]
        if len(sub) < 2:
            continue
        sns.kdeplot(sub, ax=ax, label=f"{g} (n={len(sub)})",
                    color=palette[i], linewidth=2, bw_adjust=0.7)
    ax.set_xlabel("OAI (Automation Index, 0-1)")
    ax.set_ylabel("Density")
    ax.set_title("OAI Density by Macro Group")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {out_path}")


# -----------------------------------------------------------------------------
# Markdown reports
# -----------------------------------------------------------------------------
def write_oai_by_macro_md(stats_df, out_path):
    lines = []
    lines.append("# OAI Distribution by Macro Group (K=7 main analysis)")
    lines.append("")
    lines.append("Each DWA is assigned to one of 9 analysis groups:")
    lines.append("- **M1-M7**: DWAs whose dominant macro is M1..M7 (with `dominant_share >= 0.40`)")
    lines.append("- **Noise**: DWAs whose dominant group is the noise substrate")
    lines.append(f"- **mixed_dwa**: DWAs with `dominant_share < {MIXED_THRESHOLD}` "
                 "(no single macro dominates this DWA's steps)")
    lines.append("")
    lines.append("OAI source: `shared/oai/output_10_DWA_Automation_Index.csv` "
                 "(field `Automation_Index`).")
    lines.append("")
    lines.append("`p_value_vs_rest` is a two-sided Welch's t-test of this group's OAI vs "
                 "the OAI of all other DWAs (NOT corrected for multiplicity).")
    lines.append("")

    name_lookup = {k: v["name"] for k, v in MACRO_NAMING.items()}
    name_lookup["Noise"]     = "Generic Action Substrate"
    name_lookup["mixed_dwa"] = "Mixed (no dominant macro)"

    lines.append("| Group | Name | n_DWAs | OAI mean | median | std | Q25 | Q75 | Δ vs overall | t | p (vs rest) | tech_mean | risk_mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in stats_df.iterrows():
        if r["n_dwas"] == 0:
            lines.append(f"| {r['group']} | {name_lookup.get(r['group'], '')} | 0 | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| **{r['group']}** | {name_lookup.get(r['group'], '')} | "
            f"{r['n_dwas']} | {r['oai_mean']} | {r['oai_median']} | "
            f"{r['oai_std']} | {r['oai_q25']} | {r['oai_q75']} | "
            f"{r['diff_from_overall_mean']:+.4f} | {r['t_stat_vs_rest']} | "
            f"{r['p_value_vs_rest']:.2e} | {r['tech_mean']} | {r['risk_mean']} |"
        )
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> {out_path}")


def write_external_validity_md(merged, stats_df, pairwise_df, n_pairs, out_path):
    lines = []
    lines.append("# External Validity Summary - 7-macro Taxonomy vs OAI")
    lines.append("")
    lines.append("**Question**: Does the K=7 macro taxonomy (derived from semantic embeddings of micro-actions) "
                 "predict the OAI score (derived independently from the Tech-Risk dual-factor model)? "
                 "If yes, the two methods converge -> macro taxonomy has external validity.")
    lines.append("")

    # 1. Kruskal-Wallis
    valid = merged.dropna(subset=["Automation_Index", "analysis_group"])
    samples = []
    sample_names = []
    for g in ANALYSIS_GROUPS:
        s = valid.loc[valid["analysis_group"] == g, "Automation_Index"].values
        if len(s) >= 2:
            samples.append(s)
            sample_names.append(g)
    H, p_kw = stats.kruskal(*samples)

    lines.append("## 1. Overall significance (Kruskal-Wallis)")
    lines.append("")
    lines.append("H0: All groups' OAI distributions are identical.")
    lines.append("")
    lines.append(f"- Groups in test: **{len(samples)}** ({', '.join(sample_names)})")
    lines.append(f"- H-statistic: **{H:.3f}**")
    lines.append(f"- p-value: **{p_kw:.3e}**")
    if p_kw < 0.001:
        lines.append(f"- Verdict: **p < 0.001** -> rejected H0 with high confidence (gold standard).")
    elif p_kw < 0.05:
        lines.append(f"- Verdict: **p < 0.05** -> rejected H0 at 5% level, weaker evidence.")
    else:
        lines.append(f"- Verdict: failed to reject H0 - groups statistically indistinguishable.")
    lines.append("")

    # 2. Pairwise discrimination
    n_sig = int(pairwise_df["significant"].sum())
    n_total = len(pairwise_df)
    lines.append("## 2. Pairwise discrimination (Mann-Whitney + Bonferroni)")
    lines.append("")
    lines.append(f"- Number of pairs tested: **{n_total}**")
    lines.append(f"- Pairs significant after Bonferroni (p < 0.05/{n_pairs}): "
                 f"**{n_sig} / {n_total}** = **{n_sig/n_total*100:.1f}%**")
    lines.append("")
    lines.append("### Most significant pairs (top 10 by p_bonferroni)")
    lines.append("")
    lines.append("| group_a | group_b | n_a | n_b | median_a | median_b | Cliff's δ | p_raw | p_bonf | sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    pw_sorted = pairwise_df.dropna(subset=["p_bonferroni"]).sort_values("p_bonferroni").head(10)
    for _, r in pw_sorted.iterrows():
        sig_marker = "**Y**" if r["significant"] else "n"
        lines.append(
            f"| {r['group_a']} | {r['group_b']} | {int(r['n_a'])} | {int(r['n_b'])} | "
            f"{r['median_a']} | {r['median_b']} | {r['cliffs_delta']} | "
            f"{r['p_raw']:.2e} | {r['p_bonferroni']:.2e} | {sig_marker} |"
        )
    lines.append("")
    lines.append("### Non-significant pairs (groups indistinguishable in OAI)")
    lines.append("")
    nonsig = pairwise_df[~pairwise_df["significant"]]
    if len(nonsig) > 0:
        lines.append("| group_a | group_b | p_bonf |")
        lines.append("|---|---|---|")
        for _, r in nonsig.iterrows():
            pb = r["p_bonferroni"]
            pb_str = f"{pb:.3f}" if pb is not None and not np.isnan(pb) else "n/a"
            lines.append(f"| {r['group_a']} | {r['group_b']} | {pb_str} |")
    else:
        lines.append("(All pairs are significant after Bonferroni correction.)")
    lines.append("")

    # 3. OAI ranking
    lines.append("## 3. OAI ranking (group means, low -> high)")
    lines.append("")
    ranked = stats_df[stats_df["n_dwas"] > 0].sort_values("oai_mean")
    lines.append("| Rank | Group | n_DWAs | OAI mean | OAI median | OAI std |")
    lines.append("|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        lines.append(f"| {i} | **{r['group']}** | {r['n_dwas']} | {r['oai_mean']} | "
                     f"{r['oai_median']} | {r['oai_std']} |")
    lines.append("")

    # 4. Extreme contrast
    lines.append("## 4. Extreme contrast (lowest vs highest OAI macro)")
    lines.append("")
    if len(ranked) >= 2:
        low_g  = ranked.iloc[0]["group"]
        high_g = ranked.iloc[-1]["group"]
        a = merged.loc[merged["analysis_group"] == low_g,  "Automation_Index"].dropna().values
        b = merged.loc[merged["analysis_group"] == high_g, "Automation_Index"].dropna().values
        d_cliff = cliffs_delta(b, a)   # high vs low
        d_cohen = cohens_d(b, a)
        lines.append(f"- Lowest-OAI group:  **{low_g}** (mean = {a.mean():.3f}, n = {len(a)})")
        lines.append(f"- Highest-OAI group: **{high_g}** (mean = {b.mean():.3f}, n = {len(b)})")
        lines.append(f"- Mean gap: {b.mean() - a.mean():+.3f}")
        lines.append(f"- Cliff's δ (high vs low): **{d_cliff:.3f}**")
        lines.append(f"- Cohen's d (high vs low): **{d_cohen:.3f}**")
        lines.append("")
        lines.append("Effect-size interpretation (Cliff's δ): "
                     "|δ|<0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium, >0.474 large.")
        lines.append("")
        lines.append("Effect-size interpretation (Cohen's d): "
                     "|d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.")
    lines.append("")

    # 5. mixed_dwa characterization
    lines.append("## 5. mixed_dwa group characterization")
    lines.append("")
    mixed = merged[merged["analysis_group"] == "mixed_dwa"]
    nonmixed = merged[merged["analysis_group"] != "mixed_dwa"]
    if len(mixed) > 0 and len(nonmixed) > 0:
        m_oai = mixed["Automation_Index"].dropna()
        nm_oai = nonmixed["Automation_Index"].dropna()
        U, p_mixed = stats.mannwhitneyu(m_oai, nm_oai, alternative="two-sided")
        d_cliff = cliffs_delta(m_oai.values, nm_oai.values)
        lines.append(f"- mixed_dwa n: **{len(mixed)}** "
                     f"({len(mixed) / len(merged) * 100:.1f}% of all DWAs)")
        lines.append(f"- mixed_dwa OAI mean: {m_oai.mean():.4f}")
        lines.append(f"- non-mixed OAI mean: {nm_oai.mean():.4f}")
        lines.append(f"- Difference: {m_oai.mean() - nm_oai.mean():+.4f}")
        lines.append(f"- Mann-Whitney U: {U:.0f}, p = {p_mixed:.3e}")
        lines.append(f"- Cliff's δ (mixed vs non-mixed): {d_cliff:.3f}")
        # Correlation between macro_diversity and OAI
        merged_clean = merged.dropna(subset=["Automation_Index", "macro_diversity"])
        rho, p_rho = stats.spearmanr(merged_clean["macro_diversity"],
                                     merged_clean["Automation_Index"])
        lines.append(f"- Spearman ρ(macro_diversity, OAI) over all DWAs: ρ = {rho:.3f}, p = {p_rho:.3e}")
        # Same for entropy
        rho_e, p_rho_e = stats.spearmanr(merged_clean["entropy"],
                                          merged_clean["Automation_Index"])
        lines.append(f"- Spearman ρ(entropy, OAI):                 ρ = {rho_e:.3f}, p = {p_rho_e:.3e}")
    lines.append("")

    # 6. Within-macro variance flag
    lines.append("## 6. Within-macro OAI variance (heterogeneity flag)")
    lines.append("")
    var_table = stats_df[stats_df["n_dwas"] >= 5][["group", "n_dwas", "oai_mean", "oai_std"]].copy()
    var_table["coef_var"] = (var_table["oai_std"] / var_table["oai_mean"]).round(3)
    overall_cv = (merged["Automation_Index"].std() / merged["Automation_Index"].mean())
    lines.append(f"Overall coefficient of variation (CV) = std/mean = **{overall_cv:.3f}**")
    lines.append("")
    lines.append("| Group | n | mean | std | CV | flag |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in var_table.sort_values("coef_var", ascending=False).iterrows():
        flag = "high heterogeneity" if r["coef_var"] > overall_cv * 1.2 else (
               "low heterogeneity" if r["coef_var"] < overall_cv * 0.8 else "normal")
        lines.append(f"| {r['group']} | {int(r['n_dwas'])} | {r['oai_mean']} | "
                     f"{r['oai_std']} | {r['coef_var']} | {flag} |")
    lines.append("")
    lines.append("Groups with CV substantially above overall = **internally heterogeneous in OAI** -> "
                 "downstream interpretation should be cautious for these macros.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(">>> Step 0: save macro_naming_K7.json ...")
    save_naming()

    print(">>> Step 1: load data ...")
    actions = pd.read_csv(ACTIONS_CSV)
    print(f"    actions = {len(actions):,}")
    print(f"    DWAs in clusters table = {actions['DWA_ID'].nunique():,}")

    if not os.path.exists(OAI_CSV):
        print(f"    ERROR: OAI file not found at {OAI_CSV}")
        return
    oai = pd.read_csv(OAI_CSV)
    print(f"    OAI file rows = {len(oai):,}, OAI DWAs = {oai['DWA_ID'].nunique():,}")

    print(">>> Step 2: build dwa_macro_distribution.csv ...")
    dwa_dist = build_dwa_distribution(actions)
    print(f"    DWAs in distribution table = {len(dwa_dist):,}")
    out = os.path.join(OUT_DIR, "dwa_macro_distribution.csv")
    dwa_dist.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"   -> {out}")

    print(">>> Step 3: merge with OAI ...")
    oai_subset = oai[["DWA_ID", "ai_avg_tech", "ai_avg_risk", "Automation_Index"]]
    merged = dwa_dist.merge(oai_subset, on="DWA_ID", how="left")
    n_with_oai = merged["Automation_Index"].notna().sum()
    n_missing = merged["Automation_Index"].isna().sum()
    print(f"    DWAs with OAI: {n_with_oai}, missing: {n_missing}")
    merged["analysis_group"] = merged.apply(assign_analysis_group, axis=1)
    out = os.path.join(OUT_DIR, "dwa_macro_distribution_with_oai.csv")
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"   -> {out}")

    # Group counts
    print("    Analysis group counts:")
    for g, n in merged["analysis_group"].value_counts().items():
        print(f"      {g}: {n}")

    print(">>> Step 4: per-group OAI statistics ...")
    stats_df = per_group_oai_stats(merged)
    write_oai_by_macro_md(stats_df, os.path.join(OUT_DIR, "oai_by_macro.md"))

    print(">>> Step 5: plots ...")
    plot_oai_boxplot(merged, os.path.join(OUT_DIR, "oai_boxplot_by_macro.png"))
    plot_oai_density(merged, os.path.join(OUT_DIR, "oai_density_by_macro.png"))

    print(">>> Step 6: Kruskal-Wallis + pairwise Mann-Whitney ...")
    pairwise_df, n_pairs = pairwise_mann_whitney(merged)
    out = os.path.join(OUT_DIR, "oai_pairwise_test.csv")
    pairwise_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"   -> {out}")

    print(">>> Step 7: external_validity_summary.md ...")
    write_external_validity_md(
        merged, stats_df, pairwise_df, n_pairs,
        os.path.join(OUT_DIR, "external_validity_summary.md"),
    )

    print("\n[Done]")


if __name__ == "__main__":
    main()
