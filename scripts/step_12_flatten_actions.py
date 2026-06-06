"""Step 12 - Flatten the golden DWA dataset into per-action rows (Phase 4).

Expands Final_Golden_Dataset.csv (1,961 DWAs, each carrying a JSON
sequence) into a flat micro-action table where each row is a single
micro-action carrying its parent DWA_ID and its structured fields.

Input:  data/final/final_golden_dataset.csv
Output: data/intermediate/phase4_actions_flat.csv (~15,817 rows)
"""

import pandas as pd
import json
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
PROJ      = os.path.dirname(_HERE)
INPUT_CSV = os.path.join(PROJ, "data", "final", "final_golden_dataset.csv")
OUT_CSV   = os.path.join(PROJ, "data", "intermediate", "phase4_actions_flat.csv")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

df = pd.read_csv(INPUT_CSV)

rows = []
for _, dwa_row in df.iterrows():
    try:
        steps = json.loads(dwa_row["Final_Sequence_JSON"])
    except Exception:
        continue
    for s in steps:
        rows.append({
            "DWA_ID":              dwa_row["DWA_ID"],
            "DWA_Title":           dwa_row["DWA_Title"],
            "Resolution_Source":   dwa_row["Resolution_Source"],
            "step_order":          s.get("step_order"),
            "action_description":  s.get("action_description", ""),
            "mapped_stage":        s.get("mapped_stage", ""),
            "cognitive_or_physical": s.get("cognitive_or_physical", ""),
            "key_challenge":       s.get("key_challenge", ""),
        })

flat = pd.DataFrame(rows)
flat.insert(0, "action_id", range(1, len(flat) + 1))   # global numbering 1..N
flat.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"Flattened {len(df)} DWAs -> {len(flat)} micro-actions")
print(f"Output: {OUT_CSV}")
print(f"\nColumns: {list(flat.columns)}")
print(f"\nFirst 3 rows:")
print(flat.head(3).to_string())
