# -*- coding: utf-8 -*-
"""v1.1: LLM -> Linguistic renaming for file 08 (BGE encoder of record)."""
import pandas as pd

V = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar"
d = pd.read_csv(f"{V}/dataset_zenodo_v1/08_micro_actions_intelligence_types_BGE.csv",
                encoding="utf-8-sig")
d["intelligence_type_A"] = d["intelligence_type_A"].replace({"LLM": "Linguistic"})
d = d.rename(columns={"conf_A_LLM": "conf_A_Linguistic"})
assert "LLM" not in set(d.intelligence_type_A)
d.to_csv(f"{V}/zenodo_v1.1_staging/08_micro_actions_intelligence_types_BGE.csv",
         index=False, encoding="utf-8-sig")
print("file 08 staged:", d.intelligence_type_A.value_counts().to_dict())
