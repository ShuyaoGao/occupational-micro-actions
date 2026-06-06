"""Step 04 — Extract global definition dictionaries from O*NET.

Any table that is neither occupation-keyed (no `O*NET-SOC Code` column)
nor a junction table (no `_to_` in its name) is treated as a reference
dictionary (e.g. DWA_Reference, Task_Categories). All such tables are
exported into one consolidated JSON.

Input:  data/output_01_onet_30_2.db
Output: data/output_04_global_dicts/All_Global_Dictionaries.json
"""
import sqlite3
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)
DB_PATH    = os.path.join(PROJ, "data", "output_01_onet_30_2.db")
OUTPUT_DIR = os.path.join(PROJ, "data", "output_04_global_dicts")


def build_dictionaries():
    print(">>> Step 04: extracting global definition dictionaries ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row["name"] for row in cursor.fetchall()]

    global_dict = {}
    for table in all_tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [col["name"] for col in cursor.fetchall()]
        # A "reference dictionary" has neither the SOC key nor the _to_ pattern
        if "O*NET-SOC Code" not in cols and "_to_" not in table:
            cursor.execute(f"SELECT * FROM {table}")
            global_dict[table] = [dict(row) for row in cursor.fetchall()]
            print(f"  + {table}")

    file_path = os.path.join(OUTPUT_DIR, "All_Global_Dictionaries.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(global_dict, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(global_dict)} dictionary tables merged into {file_path}")
    conn.close()


if __name__ == "__main__":
    build_dictionaries()
