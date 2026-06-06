"""Step 11 - Finalize the golden dataset (Phase 3).

Merges every resolved DWA into a single canonical golden dataset.

Inputs:
  data/intermediate/phase3_golden_sequences_auto.csv      # 1,564 auto-resolved (4-LLM majority)
  data/intermediate/Expert_Validation_Pool_Filled.csv     # 502 cases (397 frontier-2/3 + 105 conflicts)
  data/intermediate/phase3_voting_detail.csv              # per-judge vote detail

Outputs:
  data/final/final_golden_dataset.csv                     # ~1,961 resolved DWAs
  data/intermediate/phase3_conflicts_pending.csv          # 105 cases needing human arbitration
"""

import pandas as pd
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)
INTER = os.path.join(PROJ, "data", "intermediate")

AUTO_CSV     = os.path.join(INTER, "phase3_golden_sequences_auto.csv")
FILLED_CSV   = os.path.join(INTER, "Expert_Validation_Pool_Filled.csv")
GOLDEN_OUT   = os.path.join(PROJ, "data", "final", "final_golden_dataset.csv")
CONFLICT_OUT = os.path.join(INTER, "phase3_conflicts_pending.csv")


def main():
    auto   = pd.read_csv(AUTO_CSV)
    filled = pd.read_csv(FILLED_CSV)

    # 1) Auto-resolved branch: take Final_Sequence_JSON as-is
    auto_records = auto[["DWA_ID", "DWA_Title", "Final_Sequence_JSON"]].copy()
    auto_records["Resolution_Source"] = "auto_4local_majority"
    auto_records["Status"] = auto["Status"]

    # 2) Frontier-consensus branch: map A/B/C/D to the corresponding Cand_* column
    consensus = filled[filled["Expert_Choice (A/B/C/D)"].isin(["A", "B", "C", "D"])].copy()
    cand_map = {"A": "Cand_A", "B": "Cand_B", "C": "Cand_C", "D": "Cand_D"}
    consensus["Final_Sequence_JSON"] = consensus.apply(
        lambda r: r[cand_map[r["Expert_Choice (A/B/C/D)"]]], axis=1
    )
    consensus_records = consensus[["DWA_ID", "DWA_Title", "Final_Sequence_JSON"]].copy()
    consensus_records["Resolution_Source"] = "consensus_3frontier_2of3"
    consensus_records["Status"] = consensus["Status"]

    # 3) Concatenate into golden dataset
    golden = pd.concat([auto_records, consensus_records], ignore_index=True)
    golden = golden[["DWA_ID", "DWA_Title", "Resolution_Source", "Status", "Final_Sequence_JSON"]]
    golden.to_csv(GOLDEN_OUT, index=False, encoding="utf-8-sig")

    # 4) Conflict branch: cases with 1:1:1 disagreement; retain 4 candidates + 3 judge votes
    conflicts = filled[filled["Expert_Choice (A/B/C/D)"] == "CONFLICT"].copy()
    voting = pd.read_csv(os.path.join(INTER, "phase3_voting_detail.csv"))
    conflicts = conflicts.merge(
        voting[["DWA_ID", "claude_vote", "gemini_vote", "gpt_vote"]],
        on="DWA_ID", how="left"
    )
    conflicts.to_csv(CONFLICT_OUT, index=False, encoding="utf-8-sig")

    # Summary report
    print("=== Final Golden Dataset ===")
    print(f"  Auto-resolved (4 local models majority):  {len(auto_records)}")
    print(f"  Consensus (3 frontier models 2/3+):        {len(consensus_records)}")
    print(f"  Total in golden dataset:                   {len(golden)}")
    print(f"  Pending human review (1:1:1 conflict):     {len(conflicts)}")
    print(f"  Grand total covered:                       {len(golden) + len(conflicts)}")
    print()
    print(f"  -> {GOLDEN_OUT}")
    print(f"  -> {CONFLICT_OUT}")


if __name__ == "__main__":
    main()
