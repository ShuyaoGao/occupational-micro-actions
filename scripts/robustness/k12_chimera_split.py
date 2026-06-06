"""
Analyze where K=12 cut introduces new pairwise significance among
'former K=7 middle' macros, and whether the splits follow a semantic
(cognitive vs physical/service) pattern.

K=7 middle six = {M1, M4, M5, M6, Noise, mixed_dwa}.
K=7 M1=C{5,10,15,28};  M4=C{2,12,16,17,19,20,25,33,34};
   M5=C{14,18,29,30};   M6=C{1,3,4,21,22,23,31}.

Semantic prior (Paper 2 §4.1 archive):
  M1 (Locating & Provisioning):       35.4% Cog → physical-leaning
  M4 (Diagnostic Analysis):           61.8% Cog → cognitive
  M5 (Verification & Stakeholder):    83.0% Cog → strongly cognitive
  M6 (Person-Centered Service):       69.8% Cog → cognitive-service
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

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
THK  = os.path.join(BIPOLAR, "hardening", "outputs")
OUT  = THK

# K=7 macro membership (from Paper 2 §4.1)
K7_MEMBERS = {
    "M1": [5, 10, 15, 28],            # Locating & Provisioning
    "M2": [6, 7, 8, 24, 26],          # Tool-Mediated Physical Execution
    "M3": [9],                         # Iterative Repetition (meta)
    "M4": [2, 12, 16, 17, 19, 20, 25, 33, 34],   # Diagnostic Analysis
    "M5": [14, 18, 29, 30],            # Verification & Stakeholder Reporting
    "M6": [1, 3, 4, 21, 22, 23, 31],   # Person-Centered Service
    "M7": [0, 11, 13, 27, 32],         # Planning & Design
}
K7_LABELS = {
    "M1": "Locating & Provisioning",
    "M2": "Tool-Mediated Physical Execution",
    "M3": "Iterative Repetition",
    "M4": "Diagnostic Analysis",
    "M5": "Verification & Stakeholder Reporting",
    "M6": "Person-Centered Service Interaction",
    "M7": "Planning & Design",
}
K7_COG_PCT = {"M1": 35.4, "M2": 28.4, "M3": 19.6,
               "M4": 61.8, "M5": 83.0, "M6": 69.8, "M7": 84.2}

# A semantic prior tag per K=7 middle macro
K7_TAG = {
    "M1": "physical-leaning",
    "M4": "cognitive",
    "M5": "cognitive",
    "M6": "cognitive-service",
}

# Build micro → K=7 macro map
MICRO_TO_K7 = {}
for k7, members in K7_MEMBERS.items():
    for cid in members:
        MICRO_TO_K7[cid] = k7


def main():
    print(">>> Loading data ...")
    # K=12 assignments (saved earlier)
    k12 = pd.read_csv(os.path.join(THK, "macro_assignments_K12.csv"))
    micro_to_k12 = dict(zip(k12["micro_id"], k12["macro_id"]))

    # K=12 pairwise from earlier run
    pw12 = pd.read_csv(os.path.join(THK, "pairwise_K12.csv"))

    # Actions + DWA OAI to re-derive K=12 group means & TF-IDF per micro
    actions = pd.read_csv(os.path.join(P7, "actions_with_clusters.csv"))
    dwa_oai = pd.read_csv(os.path.join(P7, "dwa_macro_distribution_with_oai.csv"))
    print(f"   actions: {len(actions):,}, K=12 macros: {len(set(micro_to_k12.values()))}")

    # K=12 macro membership: macro_id (1..12) → list of micros
    k12_members = defaultdict(list)
    for cid, mid in micro_to_k12.items():
        k12_members[int(mid)].append(int(cid))

    # Build per-K=12 macro descriptor: which K=7 parents do its micros come from
    k12_origins = {}
    for mid, micros in k12_members.items():
        parents = [MICRO_TO_K7[c] for c in micros if c in MICRO_TO_K7]
        ct = Counter(parents)
        k12_origins[mid] = ct

    # Per-micro top-5 TF-IDF on the action descriptions (mean across actions in micro)
    print(">>> Computing per-micro TF-IDF top-5 ...")
    micro_top_terms = {}
    all_texts = actions["action_description"].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.85,
                          stop_words="english", lowercase=True,
                          max_features=20000)
    vec.fit(all_texts)
    feats = np.array(vec.get_feature_names_out())
    for cid in range(35):
        sub_idx = actions["cluster_id"] == cid
        if sub_idx.sum() == 0:
            micro_top_terms[cid] = []
            continue
        M = vec.transform(actions.loc[sub_idx, "action_description"].fillna("").astype(str).tolist())
        mean = np.asarray(M.mean(axis=0)).flatten()
        idx = mean.argsort()[::-1][:5]
        micro_top_terms[cid] = [str(feats[i]) for i in idx]

    # Re-derive DWA-level K=12 analysis_group + mean OAI per macro
    def label_action(cid):
        if cid < 0: return "Noise"
        return f"M{int(micro_to_k12[int(cid)])}"
    actions = actions.copy()
    actions["macro_K12"] = actions["cluster_id"].apply(label_action)

    rows = []
    for dwa_id, sub in actions.groupby("DWA_ID", sort=False):
        cnt = Counter(sub["macro_K12"])
        total = len(sub)
        dom = max(cnt.keys(), key=lambda k: cnt[k])
        dom_share = cnt[dom] / total
        ag = "mixed_dwa" if dom_share < 0.40 else dom
        rows.append({"DWA_ID": dwa_id, "analysis_group_K12": ag})
    dwa_k12 = pd.DataFrame(rows)

    merged = dwa_oai[["DWA_ID","Automation_Index"]].merge(dwa_k12, on="DWA_ID")
    merged = merged.dropna(subset=["Automation_Index"])

    # Per K=12 macro mean OAI (and n_DWAs)
    k12_means = merged.groupby("analysis_group_K12")["Automation_Index"].agg(
        ["mean", "median", "count"]).reset_index()

    # Identify "former middle" K=12 macros = those whose micros came from K=7 {M1,M4,M5,M6}
    # Plus Noise and mixed_dwa themselves (which are always "middle"-bucket categories)
    K7_MIDDLE_REAL = {"M1", "M4", "M5", "M6"}
    former_middle_k12 = set()
    for mid, origins in k12_origins.items():
        # Any K=7 middle parent counted?
        if any(k7 in K7_MIDDLE_REAL for k7 in origins):
            former_middle_k12.add(f"M{mid}")
    former_middle_k12.update({"Noise", "mixed_dwa"})

    # Filter K=12 pairwise to "both former middle" pairs
    fm_pw = pw12[(pw12["a"].isin(former_middle_k12)) & (pw12["b"].isin(former_middle_k12))].copy()
    sig_pairs = fm_pw[fm_pw["sig"] == True].copy()
    nonsig_pairs = fm_pw[fm_pw["sig"] == False].copy()

    # Tag each significant pair by what K=7 parents are on each side, and the semantic categories
    def origin_summary(macro_label):
        if macro_label in ("Noise", "mixed_dwa"):
            return macro_label, {macro_label: "boundary"}
        mid = int(macro_label[1:])
        origins = k12_origins[mid]
        parents = list(origins.keys())
        tag_set = set()
        for p in parents:
            if p in K7_TAG:
                tag_set.add(K7_TAG[p])
        return ", ".join(f"{p}×{n}" for p, n in origins.items()), tag_set

    sig_pairs["a_origin"]  = sig_pairs["a"].apply(lambda x: origin_summary(x)[0])
    sig_pairs["b_origin"]  = sig_pairs["b"].apply(lambda x: origin_summary(x)[0])
    sig_pairs["a_tags"]    = sig_pairs["a"].apply(lambda x: origin_summary(x)[1])
    sig_pairs["b_tags"]    = sig_pairs["b"].apply(lambda x: origin_summary(x)[1])

    def cross_class_label(row):
        a_set, b_set = row["a_tags"], row["b_tags"]
        if not a_set or not b_set:
            return "boundary involved"
        if a_set == b_set:
            return "within-category"
        return "cross-category"

    sig_pairs["semantic_class"] = sig_pairs.apply(cross_class_label, axis=1)

    # Count
    n_total_sig = len(sig_pairs)
    n_within = (sig_pairs["semantic_class"] == "within-category").sum()
    n_cross  = (sig_pairs["semantic_class"] == "cross-category").sum()
    n_boundary = (sig_pairs["semantic_class"] == "boundary involved").sum()

    print(f"\n>>> Former-middle K=12 macros: {sorted(former_middle_k12)}")
    print(f">>> Total former-middle pairs at K=12: {len(fm_pw)}")
    print(f">>> Newly significant: {n_total_sig}")
    print(f"     within-category: {n_within}")
    print(f"     cross-category:  {n_cross}")
    print(f"     boundary-involved (Noise/mixed): {n_boundary}")

    # Write report
    lines = ["# K=12 Split Pattern — Where Does the Middle Resolve?", ""]
    lines.append("This analysis traces which K=7 middle-macro pairs become pairwise "
                 "significant at K=12, and whether the new splits follow a semantic "
                 "(cognitive vs physical/service) pattern or are scattered randomly.")
    lines.append("")
    lines.append("## K=7 middle-macro semantic prior")
    lines.append("")
    lines.append("| K=7 macro | Name | Cog% | Tag |")
    lines.append("|---|---|---|---|")
    for k7 in ["M1", "M4", "M5", "M6"]:
        lines.append(f"| {k7} | {K7_LABELS[k7]} | {K7_COG_PCT[k7]}% | {K7_TAG[k7]} |")
    lines.append("| Noise | Generic Action Substrate | 61.5% | boundary |")
    lines.append("| mixed_dwa | (DWA-level fallback) | — | boundary |")
    lines.append("")

    lines.append("## K=12 macros descending from K=7 middle micros")
    lines.append("")
    lines.append("(Listed in OAI-mean order. Each row shows: K=12 macro id, "
                 "constituent micros (with K=7 origin tagged), and the macro's top "
                 "TF-IDF terms aggregated over its micros.)")
    lines.append("")
    lines.append("| K=12 macro | n_DWAs | mean OAI | K=7 origin breakdown | Top-5 TF-IDF (member micros) |")
    lines.append("|---|---|---|---|---|")

    # Order by mean OAI
    fm_macros_sorted = sorted(former_middle_k12 - {"Noise","mixed_dwa"},
                              key=lambda m: k12_means.set_index("analysis_group_K12")
                              ["mean"].get(m, np.nan))
    fm_macros_sorted += ["Noise", "mixed_dwa"]

    for m in fm_macros_sorted:
        kv = k12_means.set_index("analysis_group_K12")
        if m in kv.index:
            mean_oai = kv.loc[m, "mean"]
            n_dwas = int(kv.loc[m, "count"])
        else:
            mean_oai, n_dwas = float("nan"), 0
        if m in ("Noise", "mixed_dwa"):
            origin_str = "(group boundary, not micro-based)"
            top_terms_str = "(varied)"
        else:
            mid = int(m[1:])
            micros = sorted(k12_members[mid])
            origin_str = ", ".join(f"C{c}({MICRO_TO_K7.get(c,'?')})" for c in micros)
            terms = []
            for c in micros:
                terms.extend(micro_top_terms.get(c, []))
            term_freq = Counter(terms).most_common(8)
            top_terms_str = ", ".join(t for t, _ in term_freq[:6])
        lines.append(f"| **{m}** | {n_dwas} | {mean_oai:.3f} | {origin_str} | _{top_terms_str}_ |")
    lines.append("")

    # Significant pairs table
    lines.append("## Newly significant pairs at K=12 among former K=7 middle")
    lines.append("")
    lines.append(f"All 15 K=7 middle pairs were non-significant under K=7 Bonferroni. "
                 f"At K=12, after re-cutting, **{n_total_sig} pairs** among the former-middle "
                 f"K=12 sub-macros are significant at p_Bonf < 0.05.")
    lines.append("")
    lines.append("| K=12 a | K=12 b | a_origin | b_origin | semantic class | p_bonf | n_a | n_b |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in sig_pairs.sort_values("p_bonf").iterrows():
        lines.append(
            f"| {r['a']} | {r['b']} | {r['a_origin']} | {r['b_origin']} | "
            f"**{r['semantic_class']}** | {r['p_bonf']:.2e} | {int(r['n_a'])} | {int(r['n_b'])} |"
        )
    lines.append("")

    # Summary of pattern
    lines.append("## Pattern summary")
    lines.append("")
    lines.append(f"| Class | Count | Share |")
    lines.append("|---|---|---|")
    lines.append(f"| within-category (cog↔cog or phys↔phys) | {n_within} | {n_within/max(1,n_total_sig)*100:.0f}% |")
    lines.append(f"| cross-category (cog↔phys/service) | {n_cross} | {n_cross/max(1,n_total_sig)*100:.0f}% |")
    lines.append(f"| boundary-involved (Noise or mixed_dwa) | {n_boundary} | {n_boundary/max(1,n_total_sig)*100:.0f}% |")
    lines.append(f"| **TOTAL** | **{n_total_sig}** | 100% |")
    lines.append("")

    # Verdict
    lines.append("## Verdict on semantic structure")
    lines.append("")
    if n_total_sig == 0:
        verdict = "No significant new splits — N/A."
    else:
        cross_frac = n_cross / n_total_sig
        within_frac = n_within / n_total_sig
        if cross_frac > 0.55 and within_frac < 0.25:
            verdict = ("**The new significant splits at K=12 are predominantly cross-category** "
                       "(cognitive ↔ physical/service). The K=7 'middle indistinguishable' was "
                       "compressing a real semantic axis that finer resolution exposes.")
        elif within_frac > 0.55:
            verdict = ("**The new splits are mostly within-category** (e.g., one cognitive K=12 "
                       "macro splits from another cognitive K=12 macro). The cognitive/physical "
                       "prior does not capture the structure — the splits are along some other axis.")
        elif n_boundary / n_total_sig > 0.55:
            verdict = ("**Most new splits involve the Noise or mixed_dwa boundary groups**, "
                       "not the K=7 'real macro' members. The K=7 real-macro middle is more "
                       "homogeneous than the resolution sweep suggested; the additional "
                       "discrimination comes from Noise/mixed_dwa being heterogeneous.")
        else:
            verdict = ("**Mixed pattern**: splits include both cross-category and within-category "
                       "and boundary cases. No single semantic axis dominates the new splits.")
    lines.append(verdict)
    lines.append("")

    # Persist non-sig pairs for the record
    nonsig_pairs.to_csv(os.path.join(OUT, "k12_former_middle_nonsig_pairs.csv"),
                        index=False, encoding="utf-8-sig")
    sig_pairs.drop(columns=["a_tags","b_tags"], errors="ignore").to_csv(
        os.path.join(OUT, "k12_former_middle_sig_pairs.csv"),
        index=False, encoding="utf-8-sig")

    with open(os.path.join(OUT, "k12_split_pattern.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n   -> k12_split_pattern.md")
    print(f"   -> k12_former_middle_sig_pairs.csv  ({n_total_sig} rows)")
    print(f"   -> k12_former_middle_nonsig_pairs.csv  ({len(nonsig_pairs)} rows)")


if __name__ == "__main__":
    main()
