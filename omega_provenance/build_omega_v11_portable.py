# -*- coding: utf-8 -*-
"""Portable, v1.1-compatible reproduction of 09_occupation_omega_table.csv.

Runs against (a) the released v1.1 dataset files and (b) the public O*NET 30.2
text distribution. Reproduces the released table to machine precision.

Formula (see the Data Descriptor, Methods "Occupation-level composition"):
  For occupation j, each of its task statements t carries an O*NET importance
  rating IM(t, j) (Scale ID "IM"; **links without an IM rating default to 1.0**
  - 1,066 of 23,850 crosswalk links); each (task, DWA) link receives weight
  IM(t, j)/N(t), where N(t) is the number of distinct DWAs linked to t; every
  micro-action of a linked, decomposed DWA inherits that weight.
    omega_k(j)        = weighted count of type-k actions / weighted count of all actions
    substrate_share(j) = weighted count of substrate (cluster_id == -1) actions
                         / weighted count of all actions
  (omega uses the four-way intelligence-type partition and sums to 1;
   substrate_share overlaps it.)

Inputs (edit the two path constants below):
  ONET/  : "Task Statements.txt", "Tasks to DWAs.txt", "Task Ratings.txt"
  DATA/  : "08_micro_actions_intelligence_types_BGE.csv" (v1.1 vocabulary),
           "09_occupation_omega_table.csv" (for the verification diff)
"""
import pandas as pd

ONET = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar/decisive_tests/onet_30_2"
DATA = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar/zenodo_v1.1_staging"
REF  = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar/dataset_zenodo_v1"  # v1.1-unchanged file 09

TYPES = ["Linguistic", "Multimodal_Perception", "Embodied", "Human_Bound"]
OUT   = {"Linguistic": "omega_Linguistic", "Multimodal_Perception": "omega_MultimodalPerception",
         "Embodied": "omega_Embodied", "Human_Bound": "omega_HumanBound"}

# --- released per-action labels (v1.1 vocabulary) ---
f08 = pd.read_csv(f"{DATA}/08_micro_actions_intelligence_types_BGE.csv", encoding="utf-8-sig")
g = f08.groupby("DWA_ID")
cnt   = {t: g["intelligence_type_A"].apply(lambda s, t=t: int((s == t).sum())) for t in TYPES}
n_sub = g["cluster_id"].apply(lambda s: int((s == -1).sum()))
n_act = g.size()
decomposed = set(n_act.index)

# --- O*NET 30.2 public files ---
ts  = pd.read_csv(f"{ONET}/Task Statements.txt", sep="\t", dtype=str)
t2d = pd.read_csv(f"{ONET}/Tasks to DWAs.txt", sep="\t", dtype=str)
tr  = pd.read_csv(f"{ONET}/Task Ratings.txt", sep="\t", dtype=str)
for df in (ts, t2d, tr):
    df.columns = [c.strip() for c in df.columns]

im = tr[tr["Scale ID"] == "IM"]
im_dict = {(r["O*NET-SOC Code"], r["Task ID"]): float(r["Data Value"]) for _, r in im.iterrows()}

task_to_dwas = t2d.groupby("Task ID")["DWA ID"].apply(lambda s: sorted(set(s))).to_dict()
occ_tasks    = ts.groupby("O*NET-SOC Code")["Task ID"].apply(list).to_dict()

rows = []
for occ in sorted(occ_tasks):
    w_type = {t: 0.0 for t in TYPES}; w_sub = 0.0; w_all = 0.0
    uniq = set()
    for tid in occ_tasks[occ]:
        linked = task_to_dwas.get(tid, [])
        n_total = len(linked)
        lc = [d for d in linked if d in decomposed]
        if not lc:
            continue
        w_td = im_dict.get((occ, tid), 1.0) / n_total   # IM default 1.0 when unrated
        for d in lc:
            uniq.add(d)
            for t in TYPES:
                w_type[t] += w_td * cnt[t][d]
            w_sub += w_td * n_sub[d]
            w_all += w_td * n_act[d]
    if not uniq:
        continue
    row = {"onet_soc": occ, "n_unique_DWAs": len(uniq),
           "substrate_share": w_sub / w_all}
    for t in TYPES:
        row[OUT[t]] = w_type[t] / w_all
    rows.append(row)

rep = pd.DataFrame(rows)
ref = pd.read_csv(f"{REF}/09_occupation_omega_table.csv", encoding="utf-8-sig")
m = rep.merge(ref, on="onet_soc", suffixes=("_rep", ""))
cols = list(OUT.values()) + ["substrate_share"]
maxdiff = max((m[c + "_rep"] - m[c]).abs().max() for c in cols)
print(f"occupations reproduced: {len(m)} / {len(ref)}")
print(f"max |diff| over the five share columns: {maxdiff:.2e}")
print(f"n_unique_DWAs exact: {(m.n_unique_DWAs_rep == m.n_unique_DWAs).all()}")
assert len(m) == len(ref) and maxdiff < 1e-9
print("ZERO-DEVIATION REPRODUCTION CONFIRMED (IM default 1.0 for unrated links)")
