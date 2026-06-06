"""Step 10 - Tally the 4-LLM peer votes and route to Auto-Resolved / Expert pool (Phase 3).

For each DWA, counts the four LLM votes (qwen / llama / gemma / mistral) cast in
step_07 and decides:
  - 4:0  --> auto-resolved (5%% sampled to expert pool as QC)
  - 3:1  --> auto-resolved (15%% sampled as QC)
  - 2:2  --> sent to expert pool (50/50 tie)
  - 1:1:1:1 --> sent to expert pool (no consensus)

Inputs:
  data/intermediate/phase2_peer_voting/Vote_{model}.csv (4 files)
  data/intermediate/phase2_llm_synthesis/Synthesized_{model}.csv (4 files)
Outputs:
  data/intermediate/phase3_golden_sequences_auto.csv
  data/intermediate/Expert_Validation_Pool.csv
"""
import pandas as pd
from collections import Counter
import random
import os

_HERE   = os.path.dirname(os.path.abspath(__file__))
PROJ    = os.path.dirname(_HERE)
_P3_OUT = os.path.join(PROJ, "data", "intermediate", "phase2_llm_synthesis")
_P4_OUT = os.path.join(PROJ, "data", "intermediate", "phase2_peer_voting")
_OUT    = os.path.join(PROJ, "data", "intermediate")

def tally_votes():
    print(">>> Tallying 4-LLM peer votes (stratified routing) ...")
    votes_df = [pd.read_csv(os.path.join(_P4_OUT, f"Vote_{m}.csv")).set_index("DWA_ID")
                for m in ["qwen", "llama", "gemma", "mistral"]]
    synth_dfs = {opt: pd.read_csv(os.path.join(_P3_OUT, f"Synthesized_{m}.csv")).set_index("DWA_ID")
                 for opt, m in zip(["A", "B", "C", "D"], ["qwen", "llama", "gemma", "mistral"])}

    common_ids = list(set.intersection(*(set(df.index) for df in votes_df)))
    
    auto_resolved = []
    expert_pool = []
    
    for dwa_id in common_ids:
        title = synth_dfs["A"].loc[dwa_id, "DWA_Title"]
        votes = [df.loc[dwa_id, "Voted_Option"] for df in votes_df]
        counts = Counter([v for v in votes if v in ["A", "B", "C", "D"]]).most_common()
        if not counts: continue
        
        top_opt, top_count = counts[0]
        needs_expert = False
        status = f"top={top_count} votes"

        if top_count == 4:
            needs_expert = random.random() < 0.05  # QC sample: 5% of 4:0 cases
        elif top_count == 3:
            needs_expert = random.random() < 0.15  # QC sample: 15% of 3:1 cases
        elif top_count == 2:
            status = "2:2 tie"
            needs_expert = True
        else:
            status = "1:1:1:1 no consensus"
            needs_expert = True
            
        row_base = {"DWA_ID": dwa_id, "DWA_Title": title, "Votes_Array": str(votes), "Status": status}
        
        if needs_expert:
            expert_pool.append({**row_base, "Expert_Choice (A/B/C/D)": "",
                                "Cand_A": synth_dfs["A"].loc[dwa_id, "Synthesized_Sequence_JSON"],
                                "Cand_B": synth_dfs["B"].loc[dwa_id, "Synthesized_Sequence_JSON"],
                                "Cand_C": synth_dfs["C"].loc[dwa_id, "Synthesized_Sequence_JSON"],
                                "Cand_D": synth_dfs["D"].loc[dwa_id, "Synthesized_Sequence_JSON"]})
        else:
            auto_resolved.append({**row_base, "Final_Sequence_JSON": synth_dfs[top_opt].loc[dwa_id, "Synthesized_Sequence_JSON"]})

    pd.DataFrame(auto_resolved).to_csv(os.path.join(_OUT, "phase3_golden_sequences_auto.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(expert_pool).to_csv(os.path.join(_OUT, "Expert_Validation_Pool.csv"), index=False, encoding="utf-8-sig")
    print(f"Done. auto-resolved: {len(auto_resolved)}; sent to expert pool: {len(expert_pool)}.")
    print("Next: open Expert_Validation_Pool.csv, fill column 'Expert_Choice' with A/B/C/D,")
    print("      then merge with the auto-resolved table to obtain the final golden dataset (step_11).")

if __name__ == "__main__": tally_votes()