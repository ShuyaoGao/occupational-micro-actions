"""Step 09 - Merge three frontier-judge answers via majority vote (Phase 3).

Voting rule:
  3:0 or 2:1 -> adopt the majority choice into Expert_Choice
  1:1:1      -> flag as CONFLICT, send to human review

Usage:
  1. Drop each model's answer file into the matching directory:
       data/intermediate/phase3_judge_answers/claude/batch_01_answers.txt
       data/intermediate/phase3_judge_answers/gemini/batch_01_answers.txt
       data/intermediate/phase3_judge_answers/gpt/batch_01_answers.txt
  2. Run this script.
  3. Outputs:
       data/intermediate/Expert_Validation_Pool_Filled.csv   (main result)
       data/intermediate/phase3_voting_detail.csv            (per-row 3-model vote detail)
"""

import pandas as pd
import re
import os
from collections import Counter

_HERE       = os.path.dirname(os.path.abspath(__file__))
PROJ        = os.path.dirname(_HERE)
INTER       = os.path.join(PROJ, "data", "intermediate")

ANSWERS_DIR = os.path.join(INTER, "phase3_judge_answers")
META_CSV    = os.path.join(INTER, "phase3_judge_inputs", "metadata.csv")
POOL_CSV    = os.path.join(INTER, "Expert_Validation_Pool.csv")
OUTPUT_CSV  = os.path.join(INTER, "Expert_Validation_Pool_Filled.csv")
DETAIL_CSV  = os.path.join(INTER, "phase3_voting_detail.csv")

MODELS = ["claude", "gemini", "gpt"]


def load_answers(model):
    """Read all answer files for one model; return {item_num: choice}."""
    model_dir = os.path.join(ANSWERS_DIR, model)
    choices = {}
    if not os.path.exists(model_dir):
        return choices
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(model_dir, fname), encoding="utf-8") as f:
            content = f.read()
        # Tolerant pattern; matches all of:
        #   N: X | reason
        #   N: X
        #   "N": "X | reason"      <- Gemini sometimes emits JSON-style output
        pattern = r'["\']?(\d+)["\']?\s*[:.]\s*["\']?\s*([ABCDabcd])\b'
        for num_str, letter in re.findall(pattern, content):
            choices[int(num_str)] = letter.upper()
    return choices


def main():
    if not os.path.exists(META_CSV):
        print("ERROR: metadata.csv not found. Run generate_judge_batches.py first.")
        return

    meta = pd.read_csv(META_CSV).set_index("item_num")
    pool = pd.read_csv(POOL_CSV).set_index("DWA_ID")
    # Column is initially all-empty; pandas infers float64 and refuses string assignment.
    # Force object dtype so we can write letters into it.
    pool["Expert_Choice (A/B/C/D)"] = pool["Expert_Choice (A/B/C/D)"].astype("object")

    # Load each model's answers
    all_choices = {m: load_answers(m) for m in MODELS}
    for m, d in all_choices.items():
        print(f"  {m}: {len(d)} answers loaded")

    # Majority vote across the three judges
    detail_rows = []
    filled = conflict = missing = 0

    for item_num in meta.index:
        dwa_id = meta.loc[item_num, "DWA_ID"]
        votes  = {m: all_choices[m].get(item_num) for m in MODELS}
        valid  = [v for v in votes.values() if v is not None]

        if not valid:
            missing += 1
            continue

        count = Counter(valid)
        top_choice, top_count = count.most_common(1)[0]

        if top_count >= 2:          # 2:1 or 3:0
            final = top_choice
            status = f"{top_count}/3 consensus"
            filled += 1
        else:                       # 1:1:1 split
            final = "CONFLICT"
            status = "1:1:1 conflict -- needs human review"
            conflict += 1

        if dwa_id in pool.index:
            pool.loc[dwa_id, "Expert_Choice (A/B/C/D)"] = final

        detail_rows.append({
            "item_num":   item_num,
            "DWA_ID":     dwa_id,
            "claude_vote": votes.get("claude"),
            "gemini_vote": votes.get("gemini"),
            "gpt_vote":    votes.get("gpt"),
            "final_choice": final,
            "status":       status,
        })

    pool.reset_index().to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")

    print(f"\n--- Voting Results ---")
    print(f"  Consensus (2/3 or 3/3): {filled}")
    print(f"  Conflict  (1:1:1):      {conflict}  <- needs human review")
    print(f"  Not yet answered:       {missing}")
    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"  Detail: {DETAIL_CSV}")

    if conflict > 0:
        print(f"\n  Tip: filter DETAIL_CSV for status='1:1:1 conflict' to find items needing review.")


if __name__ == "__main__":
    main()
