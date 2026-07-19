# -*- coding: utf-8 -*-
"""v1.1 normalisation of the two structured per-action fields.

cognitive_or_physical: 39 raw LLM-emitted variants -> {Cognitive, Physical, Mixed, Other}
  Rule (deterministic, case-insensitive):
    mentions both "cognitive" and "physical", or equals "both"        -> Mixed
    mentions "cognitive" only                                          -> Cognitive
    mentions "physical" only                                           -> Physical
    otherwise (Sensory, Passive, System Automated; 6 rows)             -> Other
mapped_stage: 5 canonical stages; 10 rows carry compound values -> primary stage =
  first stage listed; raw preserved.
Raw values are preserved in *_raw columns. File 01's Final_Sequence_JSON keeps the
original raw strings as the provenance layer; normalisation applies to the flat
analysis files (02, 04).
Outputs staged copies: zenodo_v1.1_staging/02_micro_actions_flat.csv and
04_micro_actions_intelligence_types.csv (the latter also applies the v1.1
label renaming LLM -> Linguistic incl. confidence columns).
"""
import pandas as pd, re

V = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar"
CANON = ["Intent_Communication", "Navigation_Addressing", "Perception_Diagnosis",
         "Manipulation_Execution", "Feedback_Verification"]

def norm_cp(v: str) -> str:
    s = str(v).lower()
    has_c, has_p = "cognitive" in s, "physical" in s
    if (has_c and has_p) or s.strip() == "both":
        return "Mixed"
    if has_c:
        return "Cognitive"
    if has_p:
        return "Physical"
    return "Other"

def norm_stage(v: str) -> str:
    s = str(v)
    if s in CANON:
        return s
    for tok in re.split(r"[,&]", s):          # first listed stage is primary
        tok = tok.strip()
        if tok in CANON:
            return tok
    return s  # unreachable for current data; kept for safety

def process(fname, rename_labels=False):
    d = pd.read_csv(f"{V}/dataset_zenodo_v1/{fname}", encoding="utf-8-sig")
    d["cognitive_or_physical_raw"] = d["cognitive_or_physical"]
    d["cognitive_or_physical"] = d["cognitive_or_physical"].map(norm_cp)
    if "mapped_stage" in d.columns:
        d["mapped_stage_raw"] = d["mapped_stage"]
        d["mapped_stage"] = d["mapped_stage"].map(norm_stage)
    if rename_labels:
        d["intelligence_type"] = d["intelligence_type"].replace({"LLM": "Linguistic"})
        d = d.rename(columns={"conf_LLM": "conf_Linguistic"})
    d.to_csv(f"{V}/zenodo_v1.1_staging/{fname}", index=False, encoding="utf-8-sig")
    print(f"{fname}: cognitive_or_physical -> {d.cognitive_or_physical.value_counts().to_dict()}")
    if "mapped_stage" in d.columns:
        n_compound = (d.mapped_stage_raw != d.mapped_stage).sum()
        assert set(d.mapped_stage) <= set(CANON), set(d.mapped_stage) - set(CANON)
        print(f"  mapped_stage: 5 canonical confirmed; {n_compound} compound rows normalised")

process("02_micro_actions_flat.csv")
process("04_micro_actions_intelligence_types.csv", rename_labels=True)
