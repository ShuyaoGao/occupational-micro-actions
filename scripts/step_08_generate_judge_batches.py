"""Step 08 - Generate judge batches for three frontier models (Phase 3).

Builds self-contained batch files so each of three frontier models
(Claude, Gemini, GPT) can adjudicate every entry of the 502-row Expert
Validation Pool independently. merge_judge_answers (step_09) later takes
the majority vote (2/3 agreement adopted; 1/1/1 flagged for human review).

Batch sizing (calibrated to each model's web-UI paste ceiling):
  claude/  batch_01..  -> 8 items/batch     -> Claude Opus 4.7
  gemini/  batch_01..  -> 8 items/batch     -> Gemini 3 Pro
  gpt/     batch_01..  -> 40 items/batch    -> GPT (thinking mode)

Adjust per-model batch_size below.

Input:  data/intermediate/Expert_Validation_Pool.csv
Output: data/intermediate/phase3_judge_inputs/{claude,gemini,gpt}/batch_*.txt
"""

import pandas as pd
import json
import os
import math

_HERE      = os.path.dirname(os.path.abspath(__file__))
PROJ       = os.path.dirname(_HERE)
INPUT_CSV  = os.path.join(PROJ, "data", "intermediate", "Expert_Validation_Pool.csv")
OUTPUT_DIR = os.path.join(PROJ, "data", "intermediate", "phase3_judge_inputs")

MODEL_CONFIGS = {
    "claude": {"batch_size":  8, "label": "Claude Opus 4.7"},
    "gemini": {"batch_size":  8, "label": "Gemini 3 Pro"},
    "gpt":    {"batch_size": 40, "label": "GPT (thinking mode)"},
}

STAGE_ABBREV = {
    "Intent_Communication":   "IC",
    "Navigation_Addressing":  "NA",
    "Perception_Diagnosis":   "PD",
    "Manipulation_Execution": "ME",
    "Feedback_Verification":  "FV",
}

def abbrev_stage(s):
    for full, short in STAGE_ABBREV.items():
        if full.lower() in s.lower():
            return short
    return s[:2].upper()

def parse_steps_compact(json_str):
    try:
        steps = json.loads(json_str)
        return "\n".join(
            f"  {s.get('step_order','?')}. [{abbrev_stage(s.get('mapped_stage','?'))}] {s.get('action_description','').strip()}"
            for s in steps
        )
    except Exception:
        return "  [parse error]"

def compute_ranges(total, batch_size):
    n = math.ceil(total / batch_size)
    base, rem = divmod(total, n)
    ranges, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges

def make_header(label, batch_num, n_batches, g_first, g_last):
    return (
        f"BATCH {batch_num}/{n_batches} | Model: {label} | Tasks {g_first}-{g_last}\n"
        "================================================================\n"
        "EXPERT REVIEW -- Micro-Action Sequence Evaluation\n"
        "For each task, read ALL four options, then choose the BEST (A/B/C/D).\n\n"
        "Criteria:\n"
        "  1. Completeness  -- no critical steps missing\n"
        "  2. Realism       -- every step is concretely executable\n"
        "  3. Stage accuracy -- stage label correct for each step:\n"
        "     IC=Intent_Communication  NA=Navigation_Addressing\n"
        "     PD=Perception_Diagnosis  ME=Manipulation_Execution  FV=Feedback_Verification\n"
        "  4. Conciseness   -- no redundant or duplicate steps\n"
        "  5. Scope         -- starts and ends at exactly the right point\n\n"
        "OUTPUT: fill the ANSWER TEMPLATE at the bottom.\n"
        "Format: N: X | one-sentence reason (max 12 words)\n"
        "IMPORTANT: Each task has a different best answer. Do NOT pick the same\n"
        "letter for every task without reading.\n"
        "================================================================\n\n"
    )

def main():
    df    = pd.read_csv(INPUT_CSV)
    total = len(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write metadata mapping global item number -> DWA_ID
    meta = [{"item_num": i + 1, "DWA_ID": row["DWA_ID"]}
            for i, (_, row) in enumerate(df.iterrows())]
    pd.DataFrame(meta).to_csv(
        os.path.join(OUTPUT_DIR, "metadata.csv"),
        index=False, encoding="utf-8-sig")

    print(f"[INFO] total={total} items, all models evaluate all items")
    print(f"{'Model':<10} {'Batch size':<12} {'Batches'}")
    print("-" * 35)

    for key, cfg in MODEL_CONFIGS.items():
        label      = cfg["label"]
        batch_size = cfg["batch_size"]
        ranges     = compute_ranges(total, batch_size)
        n_batches  = len(ranges)

        model_dir = os.path.join(OUTPUT_DIR, key)
        os.makedirs(model_dir, exist_ok=True)

        # Companion directory where the judge's answer files will land
        os.makedirs(os.path.join(PROJ, "data", "intermediate", "phase3_judge_answers", key),
                    exist_ok=True)

        print(f"{key:<10} {batch_size:<12} {n_batches}")

        for b, (loc_s, loc_e) in enumerate(ranges):
            batch_df  = df.iloc[loc_s:loc_e]
            batch_num = b + 1
            g_first   = loc_s + 1   # 1-indexed first task
            g_last    = loc_e       # 1-indexed last task (inclusive)

            fname = os.path.join(model_dir, f"batch_{batch_num:02d}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(make_header(label, batch_num, n_batches, g_first, g_last))

                for local_i, (_, row) in enumerate(batch_df.iterrows()):
                    gnum = loc_s + local_i + 1
                    f.write(f"--- Task {gnum}: {row['DWA_Title']}\n")
                    for opt, col in [("A","Cand_A"),("B","Cand_B"),
                                     ("C","Cand_C"),("D","Cand_D")]:
                        f.write(f"[{opt}]\n")
                        f.write(parse_steps_compact(str(row[col])))
                        f.write("\n")
                    f.write("\n")

                # Append the answer template at the end of every batch file
                f.write("================================================================\n")
                f.write("ANSWER TEMPLATE\n\n")
                for local_i in range(loc_e - loc_s):
                    f.write(f"{loc_s + local_i + 1}: _ | \n")
                f.write("\n")

    print("\nDone.")
    print("  judge_inputs/claude/  ->  paste to Claude Opus 4.7")
    print("  judge_inputs/gemini/  ->  paste to Gemini 3 Pro")
    print("  judge_inputs/gpt/     ->  paste to GPT thinking")
    print("  Save answers to judge_answers/<model>/batch_XX_answers.txt")
    print("  Then run merge_judge_answers.py for majority voting")

if __name__ == "__main__":
    main()
