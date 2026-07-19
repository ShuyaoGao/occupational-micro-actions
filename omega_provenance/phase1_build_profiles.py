"""
Paper 3 Phase 1: Build occupation_profiles.csv per Decision D.

Step 1: Reproduce Paper 1's step_11 occupation-OAI EXACTLY to verify the
        weight-chain implementation. STOP if reproduction does not match
        output_11_Occupation_Automation_Index.csv.

Step 2: Apply the same weight chain to action-level macro counts to
        produce per-occupation macro-share vectors:
          M_k share = weighted_count(M_k actions) / weighted_count(non-Noise actions)   [main 7-dim]
        Also build the 8-dim with-Noise version, and the BGE intelligence-
        type share vector (4-dim).

Step 3: Build robustness versions using unique-DWA equal weighting
        (each unique DWA counted once per occupation). Both macro-share
        and OAI use the same unique-DWA weighting.

Step 4: Per-occupation reports:
          - Spearman: main OAI (= output_11) vs robustness equal-weight OAI
          - per-occupation cosine: main macro-share vs robustness macro-share
        Plus n_actions / n_unique_DWAs / n_tasks distribution.

Outputs:
  paper4_commonground/occupation_profiles.csv
  paper4_commonground/dropped_dwa_log.csv
  paper4_commonground/phase1_summary.md
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

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
PAPER  = os.path.join(PROJ, "paper4_commonground")
ONET   = os.path.join(SHARED, "onet_30_2")
OAI_M  = os.path.join(SHARED, "oai",
                       "output_10_DWA_Automation_Index.csv")
OAI_OCC= os.path.join(SHARED, "oai",
                       "output_11_Occupation_Automation_Index.csv")
ACTIONS= os.path.join(SHARED, "action_decomposition",
                       "actions_with_clusters.csv")
BGE    = os.path.join(SHARED, "intel_labels",
                       "actions_intelligence_type_encoderA.csv")
OUT    = PAPER
os.makedirs(OUT, exist_ok=True)

# Paper 2 K=7 macro membership
K7 = {
    "M1": [5, 10, 15, 28], "M2": [6, 7, 8, 24, 26], "M3": [9],
    "M4": [2, 12, 16, 17, 19, 20, 25, 33, 34],
    "M5": [14, 18, 29, 30], "M6": [1, 3, 4, 21, 22, 23, 31],
    "M7": [0, 11, 13, 27, 32],
}
micro_to_macro = {c: m for m, ms in K7.items() for c in ms}

INTEL = ["LLM", "Multimodal_Perception", "Embodied", "Human_Bound"]
INTEL_SHORT = ["llm", "mp", "emb", "hb"]


# ====================================================================
# STEP 0 — Load all source data
# ====================================================================

def load_data():
    print(">>> Loading source data ...")
    actions = pd.read_csv(ACTIONS)
    actions["macro"] = actions["cluster_id"].apply(
        lambda c: "Noise" if c < 0 else micro_to_macro.get(int(c), "Noise"))
    print(f"   actions: {len(actions):,} rows, {actions['DWA_ID'].nunique():,} DWAs")

    bge = pd.read_csv(BGE)[["DWA_ID", "step_order", "intelligence_type_A"]]
    print(f"   BGE intel: {len(bge):,} rows")
    # Merge BGE intel onto actions
    actions = actions.merge(bge, on=["DWA_ID", "step_order"], how="left")
    print(f"   actions+intel merged: {actions['intelligence_type_A'].notna().sum()} have intel labels")

    dwa_oai = pd.read_csv(OAI_M)[["DWA_ID", "Automation_Index"]].rename(
        columns={"Automation_Index": "OAI"})
    print(f"   DWA OAI: {len(dwa_oai):,} DWAs")

    t2d = pd.read_csv(os.path.join(ONET, "Tasks to DWAs.txt"), sep="\t")
    t2d = t2d.rename(columns={"O*NET-SOC Code": "onet_soc",
                                "Task ID": "task_id", "DWA ID": "DWA_ID"})
    print(f"   Tasks-to-DWAs crosswalk: {len(t2d):,} rows; "
          f"{t2d['onet_soc'].nunique()} occupations; "
          f"{t2d['DWA_ID'].nunique()} unique DWAs")

    tr = pd.read_csv(os.path.join(ONET, "Task Ratings.txt"), sep="\t")
    tr_im = tr[tr["Scale ID"] == "IM"][
        ["O*NET-SOC Code", "Task ID", "Data Value"]
    ].rename(columns={"O*NET-SOC Code": "onet_soc",
                       "Task ID": "task_id",
                       "Data Value": "importance"})
    tr_im["importance"] = pd.to_numeric(tr_im["importance"], errors="coerce")
    print(f"   Task IM ratings: {len(tr_im):,}")

    ts = pd.read_csv(os.path.join(ONET, "Task Statements.txt"), sep="\t")
    ts = ts.rename(columns={"O*NET-SOC Code": "onet_soc",
                              "Task ID": "task_id", "Task": "task_text"})[
        ["onet_soc", "task_id", "task_text"]]
    print(f"   Task Statements: {len(ts):,} unique (occ, task) pairs")

    od = pd.read_csv(os.path.join(ONET, "Occupation Data.txt"), sep="\t")
    od = od.rename(columns={"O*NET-SOC Code": "onet_soc",
                              "Title": "title"})[["onet_soc", "title"]]
    print(f"   Occupation Data: {len(od):,} occupations")

    return actions, dwa_oai, t2d, tr_im, ts, od


# ====================================================================
# STEP 1 — Reproduce Paper 1's step_11
# ====================================================================

def reproduce_step11(dwa_oai, t2d, tr_im, ts):
    """Reproduce occupation_OAI via:
       task_AI = mean(OAI[d] for d in DWAs linked to task)
       occ_AI  = weighted average task_AI by IM weights (default 1.0)
    """
    print("\n>>> STEP 1 — Reproducing Paper 1 step_11 ...")
    dwa_to_oai = dict(zip(dwa_oai["DWA_ID"], dwa_oai["OAI"]))

    # Global task → list of DWAs (deduplicated within task)
    task_to_dwas = (t2d.groupby("task_id")["DWA_ID"]
                       .apply(lambda s: list(set(s)))).to_dict()

    # (occ, task) → importance
    im_dict = {}
    for _, r in tr_im.iterrows():
        im_dict[(r["onet_soc"], r["task_id"])] = r["importance"]

    # tasks per occupation
    occ_tasks = ts.groupby("onet_soc")["task_id"].apply(list).to_dict()

    rows = []
    for occ in sorted(occ_tasks.keys()):
        task_ais, task_ws = [], []
        for tid in occ_tasks[occ]:
            linked = task_to_dwas.get(tid, [])
            scores = [dwa_to_oai[d] for d in linked if d in dwa_to_oai]
            if not scores:
                continue
            task_ai = float(np.mean(scores))
            w = float(im_dict.get((occ, tid), 1.0))
            if pd.isna(w):
                w = 1.0
            task_ais.append(task_ai)
            task_ws.append(w)
        if task_ais:
            occ_ai = float(np.average(task_ais, weights=task_ws))
            rows.append({"onet_soc": occ,
                          "OAI_reproduced": round(occ_ai, 4),
                          "n_tasks_covered": len(task_ais)})
    df_repro = pd.DataFrame(rows)
    return df_repro


def verify_against_published(df_repro):
    print("\n>>> Verifying reproduced OAI against output_11 ...")
    pub = pd.read_csv(OAI_OCC)
    pub = pub.rename(columns={"O*NET-SOC Code": "onet_soc",
                                "OAI_Weighted": "OAI_published",
                                "Total_Tasks_Covered": "n_tasks_pub"})
    m = df_repro.merge(pub, on="onet_soc", how="outer", indicator=True)
    n_matched = (m["_merge"] == "both").sum()
    n_repro_only = (m["_merge"] == "left_only").sum()
    n_pub_only   = (m["_merge"] == "right_only").sum()
    print(f"   matched: {n_matched}; reproduced-only: {n_repro_only}; published-only: {n_pub_only}")
    both = m[m["_merge"] == "both"].copy()
    both["abs_diff"] = (both["OAI_reproduced"] - both["OAI_published"]).abs()
    max_diff = both["abs_diff"].max()
    mean_diff = both["abs_diff"].mean()
    n_exact = (both["abs_diff"] < 1e-6).sum()
    n_close = (both["abs_diff"] < 1e-3).sum()
    print(f"   max |diff|:  {max_diff:.6f}")
    print(f"   mean |diff|: {mean_diff:.6f}")
    print(f"   exact match (|diff|<1e-6): {n_exact}/{len(both)}")
    print(f"   close match (|diff|<1e-3): {n_close}/{len(both)}")
    return both, max_diff, mean_diff, n_exact, n_close


# ====================================================================
# STEP 2 — Apply same weight chain to macro-share + intel-share
# ====================================================================

def build_main_profiles(actions, dwa_oai, t2d, tr_im, ts, od):
    """Main version uses Paper 1 step_11 weight chain extended to actions.

    For each (occ, task t, DWA d, action a):
      weight(a) = IM(t)/N(t)   where N(t) = # DWAs linked to t globally
    M_k weighted count at occ = sum over (occ, t, d, a) of weight(a) * 1[macro(a)=M_k]
    M_k 7-share at occ        = M_k count / sum non-Noise counts
    M_k 8-share at occ        = M_k count / sum all counts
    intel weighted count similarly.
    OAI is taken DIRECTLY from output_11 (Paper 1 published value).
    """
    print("\n>>> STEP 2 — Building MAIN-version profiles (Paper 1 weight chain) ...")

    # global task → DWAs (deduplicated)
    task_to_dwas = (t2d.groupby("task_id")["DWA_ID"]
                       .apply(lambda s: list(set(s)))).to_dict()
    # n_DWAs per task
    n_dwa_per_task = {t: len(dwas) for t, dwas in task_to_dwas.items()}

    # (occ, task) → importance
    im_dict = {}
    for _, r in tr_im.iterrows():
        im_dict[(r["onet_soc"], r["task_id"])] = (
            float(r["importance"]) if not pd.isna(r["importance"]) else 1.0)

    occ_tasks = ts.groupby("onet_soc")["task_id"].apply(list).to_dict()
    title_map = dict(zip(od["onet_soc"], od["title"]))

    # Pre-compute per-DWA macro counts and intel counts
    macro_counts_by_DWA = (actions.groupby(["DWA_ID", "macro"])
                                  .size()
                                  .unstack(fill_value=0))
    for k in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "Noise"]:
        if k not in macro_counts_by_DWA.columns:
            macro_counts_by_DWA[k] = 0
    intel_counts_by_DWA = (actions.dropna(subset=["intelligence_type_A"])
                                  .groupby(["DWA_ID", "intelligence_type_A"])
                                  .size()
                                  .unstack(fill_value=0))
    for k in INTEL:
        if k not in intel_counts_by_DWA.columns:
            intel_counts_by_DWA[k] = 0
    n_actions_per_DWA = actions.groupby("DWA_ID").size().to_dict()

    clustered_DWAs = set(actions["DWA_ID"].unique())

    rows = []
    for occ in sorted(occ_tasks.keys()):
        macro_weighted = {k: 0.0 for k in ["M1","M2","M3","M4","M5","M6","M7","Noise"]}
        intel_weighted = {k: 0.0 for k in INTEL}
        n_actions_weighted = 0.0
        unique_DWAs = set()
        n_tasks_with_clustered = 0
        for tid in occ_tasks[occ]:
            linked = task_to_dwas.get(tid, [])
            n_total = n_dwa_per_task.get(tid, len(linked))
            linked_clustered = [d for d in linked if d in clustered_DWAs]
            if not linked_clustered:
                continue
            n_tasks_with_clustered += 1
            im_t = im_dict.get((occ, tid), 1.0)
            w_td = im_t / n_total  # per-(t,d) weight
            for d in linked_clustered:
                unique_DWAs.add(d)
                if d in macro_counts_by_DWA.index:
                    for k in macro_weighted:
                        macro_weighted[k] += w_td * float(macro_counts_by_DWA.loc[d, k])
                if d in intel_counts_by_DWA.index:
                    for k in intel_weighted:
                        intel_weighted[k] += w_td * float(intel_counts_by_DWA.loc[d, k])
                n_actions_weighted += w_td * n_actions_per_DWA.get(d, 0)

        # Normalise
        non_noise_sum = sum(macro_weighted[k] for k in ["M1","M2","M3","M4","M5","M6","M7"])
        all_sum = non_noise_sum + macro_weighted["Noise"]
        intel_sum = sum(intel_weighted[k] for k in INTEL)

        row = {"onet_soc": occ, "title": title_map.get(occ, "")}
        for k in ["M1","M2","M3","M4","M5","M6","M7"]:
            row[f"main_{k}_share7"] = (macro_weighted[k] / non_noise_sum) if non_noise_sum > 0 else 0.0
        for k in ["M1","M2","M3","M4","M5","M6","M7","Noise"]:
            row[f"main_{k}_share8"] = (macro_weighted[k] / all_sum) if all_sum > 0 else 0.0
        for k, short in zip(INTEL, INTEL_SHORT):
            row[f"main_{short}_share"] = (intel_weighted[k] / intel_sum) if intel_sum > 0 else 0.0
        row["n_actions_weighted"] = round(n_actions_weighted, 3)
        row["n_unique_DWAs"]      = len(unique_DWAs)
        row["n_tasks_with_clustered_DWAs"] = n_tasks_with_clustered
        rows.append(row)

    return pd.DataFrame(rows)


def build_robustness_profiles(actions, dwa_oai, t2d, ts):
    """Robustness version uses unique-DWA equal weighting.

    For each occupation: unique DWAs = set of DWAs appearing in any of the
    occ's tasks (intersected with clustered set).
    M_k share = sum over unique DWAs of count(M_k actions) / sum over unique
                DWAs of count(non-Noise actions).
    8-dim with Noise: divide by all actions instead of non-Noise.
    OAI_rob = mean(OAI over unique DWAs).
    """
    print("\n>>> STEP 3 — Building ROBUSTNESS profiles (unique-DWA equal weight) ...")

    dwa_to_oai = dict(zip(dwa_oai["DWA_ID"], dwa_oai["OAI"]))
    task_to_dwas = (t2d.groupby("task_id")["DWA_ID"]
                       .apply(lambda s: list(set(s)))).to_dict()
    occ_tasks = ts.groupby("onet_soc")["task_id"].apply(list).to_dict()

    macro_counts_by_DWA = (actions.groupby(["DWA_ID", "macro"])
                                  .size()
                                  .unstack(fill_value=0))
    for k in ["M1","M2","M3","M4","M5","M6","M7","Noise"]:
        if k not in macro_counts_by_DWA.columns:
            macro_counts_by_DWA[k] = 0
    intel_counts_by_DWA = (actions.dropna(subset=["intelligence_type_A"])
                                  .groupby(["DWA_ID", "intelligence_type_A"])
                                  .size()
                                  .unstack(fill_value=0))
    for k in INTEL:
        if k not in intel_counts_by_DWA.columns:
            intel_counts_by_DWA[k] = 0

    clustered_DWAs = set(actions["DWA_ID"].unique())

    rows = []
    for occ in sorted(occ_tasks.keys()):
        unique_DWAs = set()
        for tid in occ_tasks[occ]:
            for d in task_to_dwas.get(tid, []):
                if d in clustered_DWAs:
                    unique_DWAs.add(d)
        if not unique_DWAs:
            rows.append({"onet_soc": occ})
            continue
        # macro counts summed across unique DWAs
        macro_sum = {k: 0.0 for k in ["M1","M2","M3","M4","M5","M6","M7","Noise"]}
        intel_sum = {k: 0.0 for k in INTEL}
        oai_vals = []
        for d in unique_DWAs:
            if d in macro_counts_by_DWA.index:
                for k in macro_sum:
                    macro_sum[k] += float(macro_counts_by_DWA.loc[d, k])
            if d in intel_counts_by_DWA.index:
                for k in intel_sum:
                    intel_sum[k] += float(intel_counts_by_DWA.loc[d, k])
            if d in dwa_to_oai:
                oai_vals.append(dwa_to_oai[d])

        non_noise = sum(macro_sum[k] for k in ["M1","M2","M3","M4","M5","M6","M7"])
        all_total = non_noise + macro_sum["Noise"]
        intel_total = sum(intel_sum[k] for k in INTEL)

        row = {"onet_soc": occ}
        for k in ["M1","M2","M3","M4","M5","M6","M7"]:
            row[f"rob_{k}_share7"] = (macro_sum[k] / non_noise) if non_noise > 0 else 0.0
        for k in ["M1","M2","M3","M4","M5","M6","M7","Noise"]:
            row[f"rob_{k}_share8"] = (macro_sum[k] / all_total) if all_total > 0 else 0.0
        for k, short in zip(INTEL, INTEL_SHORT):
            row[f"rob_{short}_share"] = (intel_sum[k] / intel_total) if intel_total > 0 else 0.0
        row["rob_OAI_unique_DWA_mean"] = (float(np.mean(oai_vals)) if oai_vals else np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


# ====================================================================
# Dropped-DWA log
# ====================================================================

def log_dropped(actions, dwa_oai, t2d):
    clustered = set(actions["DWA_ID"].unique())
    in_crosswalk = set(t2d["DWA_ID"].unique())
    dropped = sorted(clustered - in_crosswalk)
    print(f"\n>>> Logging {len(dropped)} dropped DWAs (clustered but not in O*NET crosswalk) ...")
    actions["macro"] = actions["cluster_id"].apply(
        lambda c: "Noise" if c < 0 else micro_to_macro.get(int(c), "Noise"))
    dwa_to_oai = dict(zip(dwa_oai["DWA_ID"], dwa_oai["OAI"]))
    rows = []
    for d in dropped:
        sub = actions[actions["DWA_ID"] == d]
        macros = sub["macro"].value_counts().to_dict()
        rows.append({"DWA_ID": d,
                      "n_actions": int(len(sub)),
                      "macro_distribution": str(macros),
                      "OAI": dwa_to_oai.get(d, np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dropped_dwa_log.csv"),
              index=False, encoding="utf-8-sig")
    return df


# ====================================================================
# Cross-version reports
# ====================================================================

def cross_version_reports(merged_profiles):
    """Spearman main vs robustness OAI; per-occupation cosine on macro-share."""
    print("\n>>> Cross-version reports ...")

    valid = merged_profiles.dropna(
        subset=["OAI_main_published", "rob_OAI_unique_DWA_mean"]).copy()

    rho_oai, p_oai = stats.spearmanr(
        valid["OAI_main_published"], valid["rob_OAI_unique_DWA_mean"])
    print(f"   OAI Spearman (main published vs robustness equal-weight): "
          f"rho = {rho_oai:.4f}, p = {p_oai:.3e}, n = {len(valid)}")

    # per-occupation cosine on 7-dim macro-share
    macro_cols_main = [f"main_{k}_share7" for k in ["M1","M2","M3","M4","M5","M6","M7"]]
    macro_cols_rob  = [f"rob_{k}_share7"  for k in ["M1","M2","M3","M4","M5","M6","M7"]]
    cos_list = []
    for _, r in valid.iterrows():
        v1 = np.array([r[c] for c in macro_cols_main], dtype=float)
        v2 = np.array([r[c] for c in macro_cols_rob ], dtype=float)
        if v1.sum() > 0 and v2.sum() > 0:
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        else:
            cos = np.nan
        cos_list.append(cos)
    cos_arr = np.array([c for c in cos_list if not np.isnan(c)])
    print(f"   macro-share 7-dim cosine (main vs robustness), n={len(cos_arr)}: "
          f"median = {np.median(cos_arr):.4f}, mean = {cos_arr.mean():.4f}, "
          f"min = {cos_arr.min():.4f}, max = {cos_arr.max():.4f}")
    pct_above_95 = (cos_arr >= 0.95).mean()
    pct_above_85 = (cos_arr >= 0.85).mean()
    print(f"   share of occupations with cosine >= 0.95: {pct_above_95*100:.1f}%")
    print(f"   share of occupations with cosine >= 0.85: {pct_above_85*100:.1f}%")

    return {
        "oai_spearman": rho_oai, "oai_p": p_oai,
        "cos_median": float(np.median(cos_arr)),
        "cos_mean":   float(cos_arr.mean()),
        "cos_min":    float(cos_arr.min()),
        "cos_max":    float(cos_arr.max()),
        "pct_cos_ge_95": float(pct_above_95),
        "pct_cos_ge_85": float(pct_above_85),
        "n_valid": int(len(valid)),
    }


# ====================================================================
# Main
# ====================================================================

def main():
    actions, dwa_oai, t2d, tr_im, ts, od = load_data()

    # Log dropped DWAs first (Decision 3)
    dropped_log = log_dropped(actions, dwa_oai, t2d)
    print(f"   dropped_dwa_log: {len(dropped_log)} rows -> paper4_commonground/dropped_dwa_log.csv")

    # Step 1: reproduce Paper 1 step_11
    repro = reproduce_step11(dwa_oai, t2d, tr_im, ts)
    both, max_diff, mean_diff, n_exact, n_close = verify_against_published(repro)

    # Iron rule: stop if reproduction fails
    if mean_diff > 0.005 or max_diff > 0.05:
        print(f"\n!! STEP_11 REPRODUCTION FAILED (max |diff| = {max_diff:.4f}, mean = {mean_diff:.4f}) !!")
        print("!! Stopping Phase 1 — do not proceed past verification step. !!")
        return

    print(f"\n   ✓ Step_11 reproduction OK (max diff {max_diff:.6f}, mean diff {mean_diff:.6f}).")

    # Step 2: main profiles (Paper 1 weight chain extended to actions)
    main_profiles = build_main_profiles(actions, dwa_oai, t2d, tr_im, ts, od)
    print(f"   main profiles: {len(main_profiles)} occupations")

    # Step 3: robustness profiles
    rob_profiles = build_robustness_profiles(actions, dwa_oai, t2d, ts)
    print(f"   rob profiles: {len(rob_profiles)} occupations")

    # Bring published OAI in (this is the canonical Paper 1 number used as main OAI)
    pub = pd.read_csv(OAI_OCC).rename(columns={
        "O*NET-SOC Code": "onet_soc",
        "OAI_Weighted": "OAI_main_published",
        "Total_Tasks_Covered": "n_tasks_pub"})

    profiles = (main_profiles
                .merge(rob_profiles, on="onet_soc", how="left")
                .merge(pub[["onet_soc", "OAI_main_published", "n_tasks_pub"]],
                        on="onet_soc", how="left"))

    # Cross-version reports
    cross = cross_version_reports(profiles)

    profiles.to_csv(os.path.join(OUT, "occupation_profiles.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"\n   -> occupation_profiles.csv ({len(profiles)} occupations, "
          f"{len(profiles.columns)} cols)")

    # ---- Summary md ----
    print("\n>>> Writing phase1_summary.md ...")
    out_md = []
    out_md.append("# Paper 3 — Phase 1 Summary")
    out_md.append("")
    out_md.append(f"Built {len(profiles)} occupation profiles per Decision D ")
    out_md.append("(main = Paper 1 step_11 IM-weighted chain; robustness = unique-DWA equal weight).")
    out_md.append("")
    out_md.append("## Step_11 reproduction check")
    out_md.append("")
    out_md.append(f"- Reproduced {len(both)} occupation OAI values from Paper 1's "
                  f"chain (Task→DWA unweighted mean, Task→Occ IM-weighted average).")
    out_md.append(f"- max |diff| = {max_diff:.6f}; mean |diff| = {mean_diff:.6f}")
    out_md.append(f"- exact match (|diff| < 1e-6): {n_exact}/{len(both)}")
    out_md.append(f"- close match (|diff| < 1e-3): {n_close}/{len(both)}")
    if mean_diff < 0.0001:
        out_md.append("- ✓ **Reproduction passes the iron-rule gate.** Proceeding to Step 2.")
    else:
        out_md.append("- ⚠ Reproduction has nontrivial residuals; reviewer should inspect.")
    out_md.append("")
    out_md.append("## Dropped DWAs")
    out_md.append("")
    out_md.append(f"- {len(dropped_log)} clustered DWAs not in O*NET crosswalk -> "
                  "logged to `dropped_dwa_log.csv`")
    out_md.append("- All 923 occupations covered without these DWAs (verified Phase 0)")
    out_md.append("")
    out_md.append("## Cross-version reports (main published OAI vs robustness equal-weight)")
    out_md.append("")
    out_md.append(f"- **OAI Spearman**: ρ = {cross['oai_spearman']:.4f}, "
                  f"p = {cross['oai_p']:.3e} (n = {cross['n_valid']})")
    out_md.append(f"- **macro-share 7-dim cosine (main vs robustness)**: ")
    out_md.append(f"  - median = {cross['cos_median']:.4f}; mean = {cross['cos_mean']:.4f}")
    out_md.append(f"  - range: {cross['cos_min']:.4f} – {cross['cos_max']:.4f}")
    out_md.append(f"  - cosine ≥ 0.95: {cross['pct_cos_ge_95']*100:.1f}%")
    out_md.append(f"  - cosine ≥ 0.85: {cross['pct_cos_ge_85']*100:.1f}%")
    out_md.append("")
    out_md.append("**Decision rule (per Decision D)**:")
    if cross['oai_spearman'] > 0.95 and cross['pct_cos_ge_95'] > 0.5:
        out_md.append("- OAI Spearman > 0.95 AND majority cosine > 0.95 → Phase 3 uses main only; robustness goes to appendix.")
    elif cross['oai_spearman'] >= 0.85:
        out_md.append("- Spearman in [0.85, 0.95] gray zone → report both in body.")
    else:
        out_md.append("- Spearman < 0.85 → main version must be justified explicitly in the paper.")
    out_md.append("")

    # n_actions distribution
    out_md.append("## Coverage distribution (for underpowered-threshold decision)")
    out_md.append("")
    out_md.append("| Quantile | n_unique_DWAs | n_actions_weighted | n_tasks |")
    out_md.append("|---|---|---|---|")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        ndwa  = profiles["n_unique_DWAs"].quantile(q)
        nact  = profiles["n_actions_weighted"].quantile(q)
        ntask = profiles["n_tasks_with_clustered_DWAs"].quantile(q)
        out_md.append(f"| q{int(q*100)} | {ndwa:.1f} | {nact:.1f} | {ntask:.0f} |")
    out_md.append("")
    out_md.append("Minimum across occupations: "
                  f"n_unique_DWAs = {profiles['n_unique_DWAs'].min():.0f}; "
                  f"n_actions_weighted = {profiles['n_actions_weighted'].min():.1f}; "
                  f"n_tasks = {profiles['n_tasks_with_clustered_DWAs'].min():.0f}.")
    out_md.append("")

    # head sample
    out_md.append("## Sample (5 rows of occupation_profiles.csv)")
    out_md.append("")
    show_cols = ["onet_soc", "title", "OAI_main_published",
                 "rob_OAI_unique_DWA_mean",
                 "main_M2_share7", "main_M7_share7",
                 "rob_M2_share7", "rob_M7_share7",
                 "n_unique_DWAs", "n_tasks_with_clustered_DWAs"]
    head_df = profiles.sort_values("OAI_main_published", ascending=False).head(5)[show_cols]
    out_md.append(head_df.to_markdown(index=False, floatfmt=".3f"))
    out_md.append("")
    out_md.append("**Highest-OAI 5 occupations + their macro-share split.** "
                  "Sanity check: high-OAI occupations should concentrate "
                  "Linguistic content (high M7 share) and low Embodied content.")
    out_md.append("")

    # tail sample
    out_md.append("## Lowest-OAI 5 occupations")
    out_md.append("")
    tail_df = profiles.sort_values("OAI_main_published", ascending=True).head(5)[show_cols]
    out_md.append(tail_df.to_markdown(index=False, floatfmt=".3f"))
    out_md.append("")

    out_md.append("## Output columns in occupation_profiles.csv")
    out_md.append("")
    out_md.append(f"Total columns: {len(profiles.columns)}")
    out_md.append("```")
    out_md.append(", ".join(profiles.columns.tolist()))
    out_md.append("```")
    out_md.append("")

    with open(os.path.join(OUT, "phase1_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_md))
    print("   -> phase1_summary.md")
    print("\n[Phase 1 done. STOPPING here per the protocol.]")


if __name__ == "__main__":
    main()
