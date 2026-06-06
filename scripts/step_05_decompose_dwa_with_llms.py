"""Step 05 - Multi-agent LLM decomposition (Phase 2).

For each O*NET DWA, queries one locally-deployed open-weight LLM to
produce a 5-12 step "micro-action sequence". Run this script four
times by switching CURRENT_MODEL_NAME across qwen / llama / gemma /
mistral; the four output CSVs feed step_06 (cross-synthesis).

Input:  data/output_04_global_dicts/All_Global_Dictionaries.json
Output: data/intermediate/phase2_llm_decomposition/DWA_Results_{model}.csv
"""
import json
import requests
import time
import os
import csv
import re

# --- Path configuration ---
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)

CURRENT_MODEL_NAME = "qwen"  # one of: "qwen" / "llama" / "gemma" / "mistral"

INPUT_JSON_PATH = os.path.join(PROJ, "data", "output_04_global_dicts",
                               "All_Global_Dictionaries.json")
OUTPUT_CSV_PATH = os.path.join(PROJ, "data", "intermediate",
                               "phase2_llm_decomposition",
                               f"DWA_Results_{CURRENT_MODEL_NAME}.csv")

# Local OpenAI-compatible API endpoint (text-generation-webui, vLLM, llama.cpp server).
API_URL = "http://127.0.0.1:5000/v1/chat/completions"

# ==========================================
# [The Epic System Prompt in English]
# ==========================================
SYSTEM_PROMPT = """
# Role
You are a world-class expert at the intersection of labor economics and robotics. Your core capability is taking any macro-level Detailed Work Activity (DWA) description and decomposing it into an extremely micro, strictly closed-loop "Micro-Action Sequence" grounded in a specific, realistic physical or digital scenario.

# Objective
Receive a task description. First, invent a highly specific, concrete real-world scenario where this task is performed. Then, explain your reasoning. Finally, decompose this process into a chronological sequence of actions, outputting the result STRICTLY in JSON format.

# Core Rules for Decomposition (CRITICAL!)
1. **Scenario-Driven:** Do not decompose in the abstract. Base every micro-action heavily on the specific `assumed_scenario` you define. 
2. **Task Boundary (Stop at delivery!):** The sequence MUST reflect a SINGLE execution cycle or workday. Stop the sequence once the immediate task is successfully delivered. DO NOT include multi-year lifecycle management, long-term maintenance, or end-of-life disposal.
3. **Step Limit:** Keep the sequence concise but granular. Aim for 5 to 12 core steps. Do not exceed 15 steps unless absolutely necessary.
4. **Micro-level Granularity:** Use physical/cognitive primitives, such as "move across obstacles," "apply specific torque," "read the third paragraph," or "listen to the client's verbal description."

# The 5 Standard Stage Mappings
* **Intent_Communication:** Understanding requirements, answering phones, confirming boundaries.
* **Navigation_Addressing:** Carrying a payload to a physical coordinate, or locating a specific file in a database.
* **Perception_Diagnosis:** Observing a leak, checking logs, abstracting logic from sparse clues.
* **Manipulation_Execution:** Hand-eye coordinated assembly, applying torque, writing core scripts.
* **Feedback_Verification:** Testing for leaks, running code, delivering the final result to the client.

# Output Format (Strict JSON)
Your output MUST be a valid JSON object. Do not include any Markdown formatting. Use the exact structure below:

{
  "task_name": "The original input task name",
  "assumed_scenario": "Describe a very specific, concrete real-world scenario (e.g., 'A technician is designing an HVAC environmental control system for a 3-story commercial office building in a humid climate').",
  "expert_reasoning": "Explain your logic: Why did you bound the task this way? What are the core cognitive/physical friction points in this specific scenario?",
  "success_criteria": "What constitutes a true, real-world closed-loop success for this specific execution?",
  "total_steps": [Integer],
  "action_sequence": [
    {
      "step_order": 1,
      "action_description": "Detailed description of the micro-action tightly linked to the assumed scenario.",
      "mapped_stage": "Intent_Communication", 
      "cognitive_or_physical": "Cognitive", 
      "key_challenge": "Describe the hardest part of this specific action for AI/Robotics."
    }
  ]
}

# Input Task
Please decompose the following task: {{Insert O*NET DWA here}}
"""

def extract_json(text):
    """Strip Markdown / chain-of-thought wrappers and return parsed JSON, or None."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to bracket-matching for ```json ... ``` style wrappers
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print(f"      [parse error] braces matched but JSON malformed: {e}")
            return None
    return None


def decompose_dwa(dwa_title):
    """Send one DWA to the local model and return the parsed decomposition."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please decompose the following task: [{dwa_title}]"}
        ],
        "temperature": 0.3,   # mild temperature: richer detail, stable format
        "max_tokens": 2500    # micro-action sequences can be long; raise ceiling to avoid truncation
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        return extract_json(content)
    except Exception as e:
        print(f"  └─ API request error: {e}")
        return None


def run_production_decomposition():
    print(">>> Starting DWA micro-action decomposition (Sequence Engine V2) ...\n")

    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ! Unable to read input file: {e}")
        return

    dwa_list = data.get('DWA_Reference', [])
    if not dwa_list:
        print("  ! 'DWA_Reference' not found in input.")
        return

    total_dwas = len(dwa_list)
    print(f"Loaded {total_dwas} DWAs to decompose.\n")

    processed_ids = set()
    file_exists = os.path.isfile(OUTPUT_CSV_PATH)
    if file_exists:
        with open(OUTPUT_CSV_PATH, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_ids.add(row['DWA_ID'])
        print(f"Resume: skipping {len(processed_ids)} already-decomposed DWAs.\n")

    with open(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['DWA_ID', 'DWA_Title', 'assumed_scenario', 'expert_reasoning',
                      'success_criteria', 'total_steps', 'action_sequence_json']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0
        for item in dwa_list:
            dwa_id = item.get('DWA ID')
            dwa_title = item.get('DWA Title')

            if dwa_id in processed_ids:
                continue

            count += 1
            print(f"[{len(processed_ids) + count}/{total_dwas}] Decomposing: {dwa_title}")

            start_time = time.time()
            result_json = decompose_dwa(dwa_title)
            elapsed = time.time() - start_time

            if result_json and isinstance(result_json, dict) and "action_sequence" in result_json:
                steps = result_json.get('total_steps', len(result_json.get('action_sequence', [])))
                print(f"  └─ decomposed into {steps} micro-steps ({elapsed:.2f}s)")
                row_data = {
                    'DWA_ID': dwa_id,
                    'DWA_Title': dwa_title,
                    'assumed_scenario': result_json.get('assumed_scenario', ''),
                    'expert_reasoning': result_json.get('expert_reasoning', ''),
                    'success_criteria': result_json.get('success_criteria', ''),
                    'total_steps': steps,
                    'action_sequence_json': json.dumps(result_json.get('action_sequence', []),
                                                       ensure_ascii=False)
                }
                writer.writerow(row_data)
                csvfile.flush()
            else:
                print("  └─ decomposition failed or JSON incomplete; skipping.")

            # Small breathing room for the GPU
            time.sleep(1)

    print(f"\nDone. Decomposed {count} DWAs this run; results written to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_production_decomposition()