"""Step 13 - Extract six structured feature fields per micro-action (Phase 4).

For every micro-action in actions_flat.csv, queries an LLM to label:
  input_modality, output_modality, environment_structure,
  liability_risk, safety_risk, compliance_burden

LLM endpoint:
  Defaults to the local OpenAI-compatible API at http://127.0.0.1:5000.
  Edit API_URL to point at any OpenAI-compatible backend (vLLM,
  text-generation-webui, llama.cpp server, ...).

Usage:
  1. Run step_12_flatten_actions.py first to produce actions_flat.csv.
  2. Run this script; it iterates row-by-row and extracts features.
  3. Resume-friendly: each completed row is flushed to actions_with_features.csv,
     so re-running picks up where the previous run left off.
"""

import pandas as pd
import json
import os
import re
import time
import requests

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(_HERE, "..", "..", "data", "Part6_outputs", "outputs", "actions_flat.csv")
OUTPUT_CSV = os.path.join(_HERE, "..", "..", "data", "Part6_outputs", "outputs", "actions_with_features.csv")

API_URL  = "http://127.0.0.1:5000/v1/chat/completions"
TIMEOUT  = 45
RETRIES  = 3

FEATURE_FIELDS = [
    "input_modality",
    "output_modality",
    "environment_structure",
    "liability_risk",
    "safety_risk",
    "compliance_burden",
]
CONFIDENCE_FIELDS = [f"{f}_conf" for f in FEATURE_FIELDS]

VALID = {
    "input_modality":        {"text", "vision", "audio", "haptic", "multimodal", "proprioceptive", "data_stream"},
    "output_modality":       {"text", "decision", "physical_action", "signal", "none"},
    "environment_structure": {"structured", "semi_structured", "unstructured"},
    "liability_risk":        {"high", "medium", "low"},
    "safety_risk":           {"high", "medium", "low"},
    "compliance_burden":     {"high", "medium", "low"},
}

SYSTEM_PROMPT = """You are a labor analyst classifying occupational micro-actions on six dimensions.

Return STRICT JSON with exactly these keys (no markdown, no commentary):
{
  "input_modality":        "text | vision | audio | haptic | multimodal | proprioceptive | data_stream",
  "output_modality":       "text | decision | physical_action | signal | none",
  "environment_structure": "structured | semi_structured | unstructured",
  "liability_risk":        "high | medium | low",
  "safety_risk":           "high | medium | low",
  "compliance_burden":     "high | medium | low",
  "confidence":            { "<each field>": 0.0-1.0 }
}

Definitions:
- input_modality: dominant sensory input.
  text=reading documents/messages; vision=visual inspection or scene parsing;
  audio=hearing speech/sound; haptic=touch-based feedback;
  proprioceptive=body position sense; data_stream=structured digital telemetry/dashboards;
  multimodal=clearly two or more.
- output_modality: dominant output action.
  text=writing/typing; decision=internal cognitive judgment with no overt output;
  physical_action=manipulating an object/body; signal=transmitting machine-readable signal;
  none=pure perception with no output.
- environment_structure:
  structured=lab/office/control room; semi_structured=factory/clinic/warehouse/store;
  unstructured=outdoor/home/disaster zone/road/field.
- liability_risk (legal exposure if step is wrong): high/medium/low
- safety_risk (physical harm to person if step is wrong): high/medium/low
- compliance_burden (regulatory or certification requirements): high/medium/low

Confidence: 0.9 = highly certain; 0.7 = reasonable; 0.5 = ambiguous.
"""

def build_user_prompt(row):
    return (
        f"Action description: {row['action_description']}\n"
        f"Key challenge: {row.get('key_challenge', '') or '(none)'}\n"
        f"Existing labels (for context): stage={row['mapped_stage']}, type={row['cognitive_or_physical']}\n"
        f"Task: classify on the 6 dimensions, return JSON only."
    )

def call_llm(user_msg):
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "max_new_tokens": 400,
    }
    for attempt in range(RETRIES):
        try:
            resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        time.sleep(0.5)
    return None

def parse_response(raw):
    """Extract JSON from raw model output, validate fields."""
    if not raw:
        return None
    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for f in FEATURE_FIELDS:
        v = str(obj.get(f, "")).strip().lower()
        if v in VALID[f]:
            out[f] = v
        else:
            return None
    confs = obj.get("confidence", {}) or {}
    for f in FEATURE_FIELDS:
        try:
            out[f"{f}_conf"] = float(confs.get(f, 0.5))
        except Exception:
            out[f"{f}_conf"] = 0.5
    return out

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")

    # Prepare output columns
    for c in FEATURE_FIELDS + CONFIDENCE_FIELDS:
        if c not in df.columns:
            df[c] = pd.NA

    # Resume support: load existing partial output
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        # Merge any pre-filled rows back into df
        for c in FEATURE_FIELDS + CONFIDENCE_FIELDS:
            if c in existing.columns:
                df[c] = existing[c].combine_first(df[c]) if c in df.columns else existing[c]
        print(f"Resume: {df[FEATURE_FIELDS[0]].notna().sum()} rows already labeled.")

    # Skip rows that are already fully labeled
    todo = df[df[FEATURE_FIELDS[0]].isna()].index.tolist()
    print(f"To label: {len(todo)} rows")

    for i, idx in enumerate(todo, 1):
        row = df.loc[idx]
        user_msg = build_user_prompt(row)
        raw = call_llm(user_msg)
        result = parse_response(raw)
        if result:
            for k, v in result.items():
                df.at[idx, k] = v
        # Flush every 25 rows
        if i % 25 == 0:
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"  [{i}/{len(todo)}] checkpoint saved")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone. Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
