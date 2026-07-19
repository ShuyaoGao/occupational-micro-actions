# -*- coding: utf-8 -*-
"""v1.1 batch, part 3: file 01 Status English normalisation, file 07 integer
cluster IDs, sample-B header hygiene + a proper sampling key.

- 01: Status (Chinese local-vote tags) -> English codes; original kept in Status_zh.
- 07: micro_cluster_id 'C0'..'C34' -> integer 0..34 (alignment with files 04/08
      verified by count reconciliation, cf. referee check).
- sampleB_intelligence_type_annotations.csv: column 'label (L / MP / E / HB)' -> 'label'.
- sampleB_sampling_key.csv: item -> action_description -> (DWA_ID, step_order)
      + encoder label/confidence + macro class. Multi-match texts flagged; the
      stratification variables (macro class, encoder confidence) are reconstructed
      from the dataset, since the original sampling script's bin boundaries were
      not preserved.
"""
import pandas as pd

V = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar"
STATUS_EN = {
    "4票最高": "local_vote_4_0",
    "3票最高": "local_vote_3_1",
    "2:2 平局或严重分歧": "local_vote_2_2_tied",
    "1:1:1:1 彻底混乱": "local_vote_split_1_1_1_1",
}

# ---- file 01 ----
f01 = pd.read_csv(f"{V}/dataset_zenodo_v1/01_dwa_decomposition_sequences.csv",
                  encoding="utf-8-sig")
f01["Status_zh"] = f01["Status"]
f01["Status"] = f01["Status"].map(STATUS_EN)
assert f01.Status.notna().all(), f01[f01.Status.isna()].Status_zh.unique()
f01.to_csv(f"{V}/zenodo_v1.1_staging/01_dwa_decomposition_sequences.csv",
           index=False, encoding="utf-8-sig")
print("01: Status ->", f01.Status.value_counts().to_dict())

# ---- file 07 ----
f07 = pd.read_csv(f"{V}/dataset_zenodo_v1/07_micro_to_macro_mapping.csv",
                  encoding="utf-8-sig")
f07["micro_cluster_id"] = f07["micro_cluster_id"].str.lstrip("C").astype(int)
f07 = f07.sort_values("micro_cluster_id")
f07.to_csv(f"{V}/zenodo_v1.1_staging/07_micro_to_macro_mapping.csv",
           index=False, encoding="utf-8-sig")
print(f"07: integer IDs {f07.micro_cluster_id.min()}..{f07.micro_cluster_id.max()}, {len(f07)} rows")

# ---- sample B header + sampling key ----
b = pd.read_csv(f"{V}/人工验证/sampleB_intelligence_type_annotations.csv",
                encoding="utf-8-sig")
b = b.rename(columns={"label (L / MP / E / HB)": "label"})
b.to_csv(f"{V}/zenodo_v1.1_staging/sampleB_intelligence_type_annotations.csv",
         index=False, encoding="utf-8-sig")

key = pd.read_csv(f"{V}/人工验证/sampleB_KEY_model_labels.csv", encoding="utf-8-sig")
items = b[b.annotator == 1][["item", "action_description"]]
assert items.action_description.is_unique, "duplicate texts within the 150-sample"
key = items.merge(key, on="action_description", how="left")
assert key.intelligence_type_A.notna().all()

d08 = pd.read_csv(f"{V}/dataset_zenodo_v1/08_micro_actions_intelligence_types_BGE.csv",
                  encoding="utf-8-sig")
f07i = f07.set_index("micro_cluster_id")
rows = []
for _, r in key.iterrows():
    m = d08[d08.action_description == r.action_description]
    if len(m) == 1:
        mm = m.iloc[0]
        dwa, step, amb = mm.DWA_ID, int(mm.step_order), 0
        cid = int(mm.cluster_id)
    else:  # ambiguous text: same sentence occurs in several DWAs
        dwa, step, amb = ";".join(m.DWA_ID), ";".join(str(int(x)) for x in m.step_order), 1
        cid = int(m.iloc[0].cluster_id) if m.cluster_id.nunique() == 1 else -999
    macro = ("substrate" if cid == -1 else
             ("ambiguous" if cid == -999 else f07i.loc[cid, "macro_id"]))
    rows.append(dict(item=int(r["item"]), action_description=r.action_description,
                     DWA_ID=dwa, step_order=step, ambiguous_match=amb,
                     encoder_label=r.intelligence_type_A.replace("LLM", "Linguistic"),
                     encoder_confidence=r.confidence_A, macro_class=macro))
kdf = pd.DataFrame(rows)
kdf.to_csv(f"{V}/zenodo_v1.1_staging/sampleB_sampling_key.csv",
           index=False, encoding="utf-8-sig")
print(f"sampling key: {len(kdf)} rows, ambiguous matches: {kdf.ambiguous_match.sum()}")
print("macro_class distribution:", kdf.macro_class.value_counts().to_dict())
