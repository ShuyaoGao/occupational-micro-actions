"""Step 07 - Peer voting across the four synthesized sequences (Phase 2).

For each DWA, asks one LLM (Qwen / Llama / Gemma / Mistral) to vote which
of the four step_06 synthesized sequences (Option A=qwen, B=llama,
C=gemma, D=mistral) is best. Run this script four times, once per voter
model. The four resulting vote CSVs feed step_10 (tally + routing).

Resume-friendly: the script detects partial output and only fills in
missing rows, preserving the master order from Option A.

Input:  data/intermediate/phase2_llm_synthesis/Synthesized_{qwen,llama,gemma,mistral}.csv
Output: data/intermediate/phase2_peer_voting/Vote_{model}.csv
"""
import json, requests, os, time, csv, re
import pandas as pd

API_URL = "http://127.0.0.1:5000/v1/chat/completions"
# Switch this between qwen / llama / gemma / mistral when running each voter.
CURRENT_MODEL_NAME = "gemma"

_HERE   = os.path.dirname(os.path.abspath(__file__))
PROJ    = os.path.dirname(_HERE)
_P3_OUT = os.path.join(PROJ, "data", "intermediate", "phase2_llm_synthesis")
OUTPUT_CSV = os.path.join(PROJ, "data", "intermediate", "phase2_peer_voting",
                          f"Vote_{CURRENT_MODEL_NAME}.csv")

SYNTH_FILES = {
    "A": os.path.join(_P3_OUT, "Synthesized_qwen.csv"),
    "B": os.path.join(_P3_OUT, "Synthesized_llama.csv"),
    "C": os.path.join(_P3_OUT, "Synthesized_gemma.csv"),
    "D": os.path.join(_P3_OUT, "Synthesized_mistral.csv"),
}

# Strict no-yapping voting prompt: just an A/B/C/D choice in JSON.
VOTING_PROMPT = """
You are an impartial Peer-Review Judge.
Evaluate which of the 4 provided micro-action sequences (Option A, B, C, or D) is the most logically sound, physically realistic, and perfectly bounded.

CRITICAL INSTRUCTION:
ABSOLUTELY NO YAPPING! DO NOT explain your reasoning. DO NOT output markdown.
Output STRICTLY a JSON object with a single key "best_option".
Example: {"best_option": "C"}
"""


def extract_vote(text):
    """Robust vote extractor: handles JSON, malformed JSON, and bare 'Option X' answers."""
    text = text.strip()

    # 1. Strict JSON match
    match_strict = re.search(r'\{\s*"best_option"\s*:\s*"([ABCDabcd])"\s*\}', text)
    if match_strict:
        return match_strict.group(1).upper()

    match_json = re.search(r'\{[\s\S]*\}', text)
    if match_json:
        try:
            vote = json.loads(match_json.group(0)).get("best_option", "").strip().upper()
            if vote in ["A", "B", "C", "D"]:
                return vote
        except Exception:
            pass

    # 2. Fallback regex: catches malformed JSON containing "best_option": "X"
    match_regex = re.search(r'\"best_option\"\s*:\s*\"([ABCD])\"', text, re.IGNORECASE)
    if match_regex:
        return match_regex.group(1).upper()

    # 3. Last resort: a bare "Option X" mention
    match_char = re.search(r'(?:Option|best_option)["\s:=]+([A-D])\b', text, re.IGNORECASE)
    if match_char:
        return match_char.group(1).upper()

    return None


def run_voting_ordered():
    print(f">>> Peer voting | judge model: {CURRENT_MODEL_NAME.upper()} ...\n")

    # Iterate in the master order defined by Option A (qwen), so all four vote files align.
    try:
        df_master = pd.read_csv(SYNTH_FILES["A"])
        master_order_ids = df_master['DWA_ID'].tolist()
        dfs = {opt: pd.read_csv(path).set_index("DWA_ID")
               for opt, path in SYNTH_FILES.items() if os.path.exists(path)}
    except Exception as e:
        print(f"  ! Failed to read synthesis CSVs (need all four). Error: {e}")
        return

    # Resume support: pick up where the prior run stopped.
    processed_ids = set()
    if os.path.isfile(OUTPUT_CSV):
        try:
            processed_ids = set(pd.read_csv(OUTPUT_CSV)['DWA_ID'].astype(str))
            print(f"Resume: {len(processed_ids)} votes already cast; scanning for gaps ...")
        except Exception:
            pass
    else:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['DWA_ID', 'Voted_Option'])

    missing_ids = [did for did in master_order_ids if str(did) not in processed_ids]

    if not missing_ids:
        print("All votes complete for this model.")
    else:
        print(f"Missing {len(missing_ids)} votes; resuming ...\n")
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)

            for idx, dwa_id in enumerate(missing_ids):
                title = df_master.loc[df_master['DWA_ID'] == dwa_id, 'DWA_Title'].values[0]
                print(f"[{idx+1}/{len(missing_ids)}] Voting on: {title}")

                combined_text = f"Task: {title}\n\n"
                for opt in ["A", "B", "C", "D"]:
                    if dwa_id in dfs[opt].index:
                        try:
                            combined_text += (f"--- Option {opt} ---\n"
                                              f"{dfs[opt].loc[dwa_id, 'Synthesized_Sequence_JSON']}\n\n")
                        except Exception:
                            pass

                # Lock down output: zero temperature + tight token budget.
                payload = {
                    "messages": [{"role": "system", "content": VOTING_PROMPT},
                                 {"role": "user",   "content": combined_text}],
                    "temperature": 0.0,
                    "max_tokens": 100,
                    "max_new_tokens": 100
                }

                # Up to 3 retries on transient API failures.
                success = False
                for attempt in range(3):
                    try:
                        resp = requests.post(API_URL, json=payload, timeout=45)
                        if resp.status_code != 200:
                            print(f"  └─ API status {resp.status_code}")
                            continue

                        raw_content = resp.json()['choices'][0]['message']['content']
                        vote = extract_vote(raw_content)

                        if vote in ["A", "B", "C", "D"]:
                            writer.writerow([dwa_id, vote])
                            csvfile.flush()
                            print(f"  └─ vote: {vote}")
                            success = True
                            break
                        else:
                            print(f"  └─ attempt {attempt+1}/3 unparseable: {raw_content.strip()}")
                    except requests.exceptions.Timeout:
                        print(f"  └─ timeout, retry {attempt+1}/3 ...")
                    except Exception as e:
                        print(f"  └─ API error: {e}")
                    time.sleep(0.5)

                if not success:
                    print(f"  └─ all retries failed; recording as empty.")

    # Final pass: enforce strict master ordering on the output CSV.
    print("\n>>> Re-sorting output to match the master DWA order ...")
    try:
        df_final = pd.read_csv(OUTPUT_CSV)
        df_final = df_final.drop_duplicates(subset=['DWA_ID'], keep='last')

        existing_ids = set(df_final['DWA_ID'].astype(str))
        valid_master_ids = [did for did in master_order_ids if str(did) in existing_ids]

        df_final['DWA_ID'] = pd.Categorical(df_final['DWA_ID'],
                                            categories=valid_master_ids, ordered=True)
        df_final = df_final.sort_values('DWA_ID').reset_index(drop=True)
        df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"Done. Output aligned to master order; total votes: {len(df_final)}.")
    except Exception as e:
        print(f"  ! Sort step failed (data is fine, ordering only): {e}")


if __name__ == "__main__":
    run_voting_ordered()
