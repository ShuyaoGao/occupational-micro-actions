# -*- coding: utf-8 -*-
"""Step 16 - Exposure projection and the full statistical battery.

Projects the DWA-level GPT-4 task-exposure score (Eloundou et al. 2023/2024,
packaged as data/external/dwa_external_indices.csv) onto the K=7 macro
typology and reproduces every exposure statistic in the paper:

  * group stats + Kruskal-Wallis + 28 pairwise Mann-Whitney (Bonferroni)
  * M7-vs-M2 extreme-pair effect sizes (Cliff's delta, Cohen's d)
  * TOST equivalence on the 15 middle pairs
  * M2 dip test + per-micro means; Noise-vs-population KS
  * M4 chimera (K=12 sub-macros)
  * resolution sweep K = 7/8/10/12/15
  * AIOE + Frey-Osborne cross-indicator reproduction, era gradient
  * 9-macro era inversion (FO Oxford-Martin original vs Eloundou/AIOE)
  * intelligence-type (BGE) exposure prediction

Inputs (all shipped in this repository):
  data/final/actions_with_clusters.csv
  data/external/dwa_external_indices.csv          (Eloundou_task / AIOE / FO)
  data/external/macro_assignments_K{7,8,10,12,15}.csv
  data/external/actions_intelligence_type_encoderA.csv
  data/external/macro_era_means_with_fo_original.csv

Outputs -> outputs/exposure_projection/
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import diptest
from statsmodels.stats.weightstats import ttost_ind

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs", "exposure_projection")
os.makedirs(OUT, exist_ok=True)

ELX = "Eloundou_task"
K7_MEMBERS = {
    "M1": [5, 10, 15, 28], "M2": [6, 7, 8, 24, 26], "M3": [9],
    "M4": [2, 12, 16, 17, 19, 20, 25, 33, 34],
    "M5": [14, 18, 29, 30], "M6": [1, 3, 4, 21, 22, 23, 31],
    "M7": [0, 11, 13, 27, 32],
}
MICRO_TO_K7 = {c: k for k, v in K7_MEMBERS.items() for c in v}
ANALYSIS_GROUPS = ["M1", "M2", "M4", "M5", "M6", "M7", "Noise", "mixed_dwa"]
MIDDLE6 = ["M1", "M4", "M5", "M6", "Noise", "mixed_dwa"]
K12_SUBMACRO = {6: "M4-patrol", 7: "M4-HVAC", 8: "M4-data"}


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sb = np.sort(b)
    gt = np.searchsorted(sb, a, side="left").sum()
    lt = (len(b) - np.searchsorted(sb, a, side="right")).sum()
    return (gt - lt) / (len(a) * len(b))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def dominant_assignment(actions, micro_to_macro, threshold=0.40, tiebreak="fixed_order"):
    lab = actions["cluster_id"].apply(
        lambda c: "Noise" if int(c) < 0 else micro_to_macro.get(int(c), "Noise"))
    tmp = pd.DataFrame({"DWA_ID": actions["DWA_ID"], "macro": lab})
    all_groups = sorted({g for g in micro_to_macro.values()},
                        key=lambda g: int(g[1:])) + ["Noise"]
    rows = []
    for dwa_id, sub in tmp.groupby("DWA_ID", sort=False):
        counts = Counter(sub["macro"])
        if tiebreak == "fixed_order":
            dom = max(all_groups, key=lambda g: counts.get(g, 0))
        else:
            dom = max(counts.keys(), key=lambda k: counts[k])
        share = counts.get(dom, 0) / len(sub)
        rows.append({"DWA_ID": dwa_id,
                     "group": "mixed_dwa" if share < threshold else dom})
    return pd.DataFrame(rows)


def pairwise_mw(df, groups, col):
    present = [g for g in groups if (df["group"] == g).sum() >= 2]
    n_pairs = len(present) * (len(present) - 1) // 2
    rows = []
    for i, g1 in enumerate(present):
        for g2 in present[i + 1:]:
            a = df.loc[df["group"] == g1, col].dropna().values
            b = df.loc[df["group"] == g2, col].dropna().values
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"a": g1, "b": g2, "n_a": len(a), "n_b": len(b),
                         "U": float(U), "p_raw": float(p),
                         "p_bonf": float(min(1.0, p * n_pairs)),
                         "cliffs_delta": float(cliffs_delta(a, b)),
                         "sig": bool(min(1.0, p * n_pairs) < 0.05)})
    return pd.DataFrame(rows), n_pairs


print(">>> Loading ...")
ext = pd.read_csv(os.path.join(DATA, "external", "dwa_external_indices.csv"))
actions = pd.read_csv(os.path.join(DATA, "final", "actions_with_clusters.csv"))
intel = pd.read_csv(os.path.join(DATA, "external", "actions_intelligence_type_encoderA.csv"))

dwa7 = dominant_assignment(actions, MICRO_TO_K7)
df = ext.merge(dwa7, on="DWA_ID", how="left")
assert (df["group"] != df["analysis_group"]).sum() == 0
dfe = df.dropna(subset=[ELX]).copy()
report = {}

# ---- group stats / KW / pairwise / extremes -------------------------------
gs = dfe.groupby("group")[ELX].agg(["count", "mean", "median", "std"]).round(4)
gs.to_csv(os.path.join(OUT, "group_stats.csv"))
samples = [dfe.loc[dfe["group"] == g, ELX].values for g in ANALYSIS_GROUPS]
H, p = stats.kruskal(*samples)
pw, n_pairs = pairwise_mw(dfe, ANALYSIS_GROUPS, ELX)
pw.to_csv(os.path.join(OUT, "pairwise_mannwhitney.csv"), index=False)
m2 = dfe.loc[dfe["group"] == "M2", ELX].values
m7 = dfe.loc[dfe["group"] == "M7", ELX].values
U27, p27 = stats.mannwhitneyu(m7, m2, alternative="two-sided")
mid = pw[pw["a"].isin(MIDDLE6) & pw["b"].isin(MIDDLE6)]
report["kw"] = {"H": round(float(H), 2), "p": float(p)}
report["extreme"] = {"U": float(U27), "p_raw": float(p27),
                     "p_bonf": float(min(1, p27 * n_pairs)),
                     "cliffs_delta": round(float(cliffs_delta(m7, m2)), 3),
                     "cohens_d": round(float(cohens_d(m7, m2)), 2)}
report["pairwise"] = {"n_sig": int(pw["sig"].sum()),
                      "middle_nonsig": int((~mid["sig"]).sum())}

# ---- TOST ------------------------------------------------------------------
mid_arrays = [dfe.loc[dfe["group"] == g, ELX].values for g in MIDDLE6]
sigma_pool = float(np.sqrt(sum((len(x) - 1) * np.var(x, ddof=1) for x in mid_arrays)
                           / sum(len(x) - 1 for x in mid_arrays)))
d02 = round(0.2 * sigma_pool, 4)
tost_rows = []
pairs = [(a, b) for i, a in enumerate(MIDDLE6) for b in MIDDLE6[i + 1:]] + [("M2", "M7")]
for a, b in pairs:
    xa = dfe.loc[dfe["group"] == a, ELX].values
    xb = dfe.loc[dfe["group"] == b, ELX].values
    p_d, _, _ = ttost_ind(xa, xb, low=-d02, upp=d02, usevar="pooled")
    p_a, _, _ = ttost_ind(xa, xb, low=-0.05, upp=0.05, usevar="pooled")
    tost_rows.append({"a": a, "b": b, "p_TOST_d02": float(p_d), "p_TOST_abs": float(p_a)})
pd.DataFrame(tost_rows).to_csv(os.path.join(OUT, "tost.csv"), index=False)
report["tost"] = {"sigma_pool": round(sigma_pool, 4), "delta_d02": d02,
                  "n_equiv_d02": int(sum(r["p_TOST_d02"] < 0.05 for r in tost_rows[:-1])),
                  "n_equiv_abs": int(sum(r["p_TOST_abs"] < 0.05 for r in tost_rows[:-1]))}

# ---- M2 heterogeneity / dip / Noise KS --------------------------------------
D2, pdip = diptest.diptest(m2)
m2_dwas = dfe.loc[dfe["group"] == "M2", "DWA_ID"]
am2 = actions[actions["DWA_ID"].isin(m2_dwas) & actions["cluster_id"].isin(K7_MEMBERS["M2"])]
dom_micro = am2.groupby("DWA_ID")["cluster_id"].agg(lambda s: s.value_counts().idxmax())
m2tab = dfe[dfe["group"] == "M2"].merge(dom_micro.rename("micro").reset_index(), on="DWA_ID")
m2tab.groupby("micro")[ELX].agg(["count", "mean"]).round(4).to_csv(
    os.path.join(OUT, "m2_micro_means.csv"))
Dn, pks = stats.ks_2samp(dfe.loc[dfe["group"] == "Noise", ELX].values, dfe[ELX].values)
report["m2_dip"] = {"D": round(float(D2), 4), "p": round(float(pdip), 3)}
report["noise_ks"] = {"D": round(float(Dn), 4), "p": round(float(pks), 3)}

# ---- M4 chimera --------------------------------------------------------------
k12 = pd.read_csv(os.path.join(DATA, "external", "macro_assignments_K12.csv"))
micro_to_k12 = {int(r.micro_id): f"M{int(r.macro_id)}" for r in k12.itertuples()}
dwa12 = dominant_assignment(actions, micro_to_k12, tiebreak="first_occurrence")
df12 = ext.merge(dwa12.rename(columns={"group": "g12"}), on="DWA_ID", how="left").dropna(subset=[ELX])
chim = {}
for mid_id, name in {f"M{k}": v for k, v in K12_SUBMACRO.items()}.items():
    x = df12.loc[df12["g12"] == mid_id, ELX]
    chim[name] = {"n": int(len(x)), "mean": round(float(x.mean()), 3)}
report["m4_chimera"] = chim

# ---- resolution sweep ----------------------------------------------------------
sweep = []
for K in [7, 8, 10, 12, 15]:
    kf = pd.read_csv(os.path.join(DATA, "external", f"macro_assignments_K{K}.csv"))
    m2m = {int(r.micro_id): f"M{int(r.macro_id)}" for r in kf.itertuples()}
    dk = dominant_assignment(actions, m2m, tiebreak="first_occurrence")
    dke = ext.merge(dk, on="DWA_ID", how="left").dropna(subset=[ELX])
    present = [g for g in dke["group"].unique() if (dke["group"] == g).sum() >= 2]
    real = [g for g in present if g not in ("Noise", "mixed_dwa")]
    means = {g: dke.loc[dke["group"] == g, ELX].mean() for g in real}
    lo, hi = min(means, key=means.get), max(means, key=means.get)
    pwk, _ = pairwise_mw(dke, present, ELX)
    middle = [g for g in present if g not in (lo, hi)]
    mpk = pwk[pwk["a"].isin(middle) & pwk["b"].isin(middle)]
    smp = [dke.loc[dke["group"] == g, ELX].values for g in present]
    Hk, _ = stats.kruskal(*smp)
    sweep.append({"K": K, "low": lo, "high": hi,
                  "low_mean": round(float(means[lo]), 3),
                  "high_mean": round(float(means[hi]), 3),
                  "KW_H": round(float(Hk), 1),
                  "middle_nonsig": f"{int((~mpk['sig']).sum())}/{len(mpk)}"})
pd.DataFrame(sweep).to_csv(os.path.join(OUT, "resolution_sweep.csv"), index=False)

# ---- cross indicators + era gradient -------------------------------------------
for col in ["AIOE", "FreyOsborne"]:
    d = df.dropna(subset=[col])
    smp = [d.loc[d["group"] == g, col].values for g in ANALYSIS_GROUPS]
    Hx, px = stats.kruskal(*smp)
    pwx, _ = pairwise_mw(d, ANALYSIS_GROUPS, col)
    midx = pwx[pwx["a"].isin(MIDDLE6) & pwx["b"].isin(MIDDLE6)]
    report[col] = {"KW_H": round(float(Hx), 1), "n_sig": int(pwx["sig"].sum()),
                   "middle_nonsig": int((~midx["sig"]).sum())}

# ---- era inversion ------------------------------------------------------------
def macro_exp(cid):
    cid = int(cid)
    if cid < 0:
        return "Noise"
    k7 = MICRO_TO_K7.get(cid)
    if k7 == "M4":
        return K12_SUBMACRO.get(int(micro_to_k12[cid][1:]), "M4-other")
    return k7

acts = actions.copy()
acts["mx"] = acts["cluster_id"].apply(macro_exp)
dx = ext.merge(acts.groupby("DWA_ID")["mx"].agg(lambda s: s.value_counts().idxmax()
                                                ).rename("mx").reset_index(),
               on="DWA_ID", how="left")
era = dx.groupby("mx").agg(n=("DWA_ID", "size"),
                           mean_El=(ELX, "mean"), mean_AIOE=("AIOE", "mean"),
                           mean_FO=("FreyOsborne", "mean")).round(3).reset_index()
fo_orig = pd.read_csv(os.path.join(DATA, "external", "macro_era_means_with_fo_original.csv"))
era = era.merge(fo_orig[["macro", "mean_FO_orig"]], left_on="mx", right_on="macro", how="inner")
era.to_csv(os.path.join(OUT, "era_inversion_means.csv"), index=False)
rho, p9 = stats.spearmanr(era["mean_FO_orig"], era["mean_El"])
rho_a, p9a = stats.spearmanr(era["mean_FO_orig"], era["mean_AIOE"])
dfo = df.dropna(subset=[ELX, "FreyOsborne"])
rho_dwa, p_dwa = stats.spearmanr(dfo[ELX], dfo["FreyOsborne"])
report["era_inversion"] = {
    "spearman_FO_Eloundou_9macro": {"rho": round(float(rho), 3), "p": round(float(p9), 4)},
    "spearman_FO_AIOE_9macro": {"rho": round(float(rho_a), 3), "p": round(float(p9a), 4)},
    "dwa_FO_Eloundou": {"rho": round(float(rho_dwa), 3), "p": float(p_dwa), "n": int(len(dfo))},
}

# ---- intelligence types ----------------------------------------------------------
dom_t = intel.groupby("DWA_ID")["intelligence_type_A"].agg(
    lambda s: s.value_counts().idxmax()).rename("t").reset_index()
di = dfe.merge(dom_t, on="DWA_ID", how="left")
types = sorted(di["t"].dropna().unique())
itab = di.groupby("t")[ELX].agg(["count", "mean", "median"]).round(4)
itab.to_csv(os.path.join(OUT, "intel_exposure.csv"))
Hi, pi = stats.kruskal(*[di.loc[di["t"] == t, ELX].values for t in types])
lng = di.loc[di["t"] == "LLM", ELX]
rest = di.loc[di["t"] != "LLM", ELX]
report["intel"] = {"KW_H": round(float(Hi), 1), "p": float(pi),
                   "ratio_lng_vs_rest": round(float(lng.mean() / rest.mean()), 2)}

with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print("\n[Done] ->", OUT)
