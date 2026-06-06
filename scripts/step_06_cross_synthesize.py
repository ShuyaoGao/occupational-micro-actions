"""Step 06 - Cross-synthesis across the four LLMs (Phase 2).

For each DWA, takes the four parallel step_05 decompositions and asks one
LLM to synthesize them into a single canonical micro-action sequence.
Run this script four times (qwen / llama / gemma / mistral) to get four
synthesized versions; step_07 then peer-votes across these four.

Input:  data/intermediate/phase2_llm_decomposition/DWA_Results_{qwen,llama,gemma,mistral}.csv
Output: data/intermediate/phase2_llm_synthesis/Synthesized_{model}.csv
"""
import json, requests, os, time, csv, re
import pandas as pd

API_URL = "http://127.0.0.1:5000/v1/chat/completions"
# Switch model name when running this script against a different local backend.
CURRENT_MODEL_NAME = "mistral"  # one of: qwen / llama / gemma / mistral

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)
_P2_OUT = os.path.join(PROJ, "data", "intermediate", "phase2_llm_decomposition")

INITIAL_FILES = {
    "qwen":    os.path.join(_P2_OUT, "DWA_Results_qwen.csv"),
    "llama":   os.path.join(_P2_OUT, "DWA_Results_llama.csv"),
    "gemma":   os.path.join(_P2_OUT, "DWA_Results_gemma.csv"),
    "mistral": os.path.join(_P2_OUT, "DWA_Results_mistral.csv"),
}
OUTPUT_CSV = os.path.join(PROJ, "data", "intermediate", "phase2_llm_synthesis",
                          f"Synthesized_{CURRENT_MODEL_NAME}.csv")

SYNTHESIS_PROMPT = """
You are a Lead AI & Ergonomics Expert.
I will provide you with a specific work task and 4 different preliminary micro-action sequences drafted by other AI assistants.

Your Goal:
Critically evaluate these 4 drafts. Absorb their best insights regarding logical flow, chronological order, and the boundary between physical and cognitive actions. 
Synthesize them into ONE PERFECT, standardized action sequence.

Rules:
1. Remove redundant steps. Aim for 5 to 12 core steps.
2. Every step MUST use one of the 5 standard stages: Intent_Communication, Navigation_Addressing, Perception_Diagnosis, Manipulation_Execution, Feedback_Verification.
3. Output STRICTLY as a JSON array of objects. NO markdown outside the array.

Output Format:
[
  {
    "step_order": 1,
    "action_description": "...",
    "mapped_stage": "...",
    "cognitive_or_physical": "...",
    "key_challenge": "..."
  }
]
"""

def extract_json_array(text):
    match = re.search(r'\[[\s\S]*\]', text.strip())
    return json.loads(match.group(0)) if match else None

def run_synthesis():
    print(f">>> Cross-synthesis | active model: {CURRENT_MODEL_NAME.upper()} ...\n")
    dfs = {name: pd.read_csv(path).set_index("DWA_ID") for name, path in INITIAL_FILES.items() if os.path.exists(path)}
    common_ids = list(set.intersection(*(set(df.index) for df in dfs.values())))
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists: writer.writerow(['DWA_ID', 'DWA_Title', 'Synthesized_Sequence_JSON'])
        processed_ids = set(pd.read_csv(OUTPUT_CSV)['DWA_ID']) if file_exists else set()

        for idx, dwa_id in enumerate(common_ids):
            if dwa_id in processed_ids: continue
            title = dfs["qwen"].loc[dwa_id, "DWA_Title"]
            print(f"[{idx+1}/{len(common_ids)}] Synthesizing: {title}")

            # Concatenate only the action arrays to keep prompt length manageable
            combined_text = f"Task: {title}\n\n"
            for name, df in dfs.items():
                try:
                    seq = json.loads(df.loc[dwa_id, 'action_sequence_json'])
                    clean_seq = [{"step": s.get("step_order"), "action": s.get("action_description"), "stage": s.get("mapped_stage")} for s in seq]
                    combined_text += f"--- Draft {name.upper()} ---\n{json.dumps(clean_seq, ensure_ascii=False)}\n\n"
                except: pass

            payload = {"messages": [{"role": "system", "content": SYNTHESIS_PROMPT}, {"role": "user", "content": combined_text}], "temperature": 0.2, "max_tokens": 1500}
            try:
                resp = requests.post(API_URL, json=payload, timeout=90)
                synth_json = extract_json_array(resp.json()['choices'][0]['message']['content'])
                if synth_json:
                    writer.writerow([dwa_id, title, json.dumps(synth_json, ensure_ascii=False)])
                    csvfile.flush()
                    print("  └─ Synthesized sequence written")
            except Exception as e: print(f"  └─ Error: {e}")
            time.sleep(0.5)

if __name__ == "__main__": run_synthesis()