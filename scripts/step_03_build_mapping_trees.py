"""Step 03 — Build inverted mapping trees between O*NET entities.

For every junction table whose name contains '_to_' (e.g. Tasks_to_DWAs,
Abilities_to_Work_Activities), this script builds an inverted JSON of
the form  parent_id -> [child_id, ...].  These inverted indices are the
backbone for the downstream Importance-Weighted aggregation chain.

Input:  data/output_01_onet_30_2.db
Output: data/output_03_mapping_trees/{TableName}_Inverted.json
"""
import sqlite3
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)
DB_PATH    = os.path.join(PROJ, "data", "output_01_onet_30_2.db")
OUTPUT_DIR = os.path.join(PROJ, "data", "output_03_mapping_trees")


def build_mappings():
    print(">>> Step 03: building inverted mapping trees ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    mapping_tables = [
        row["name"] for row in cursor.fetchall() if "_to_" in row["name"]
    ]
    print(f"Found {len(mapping_tables)} junction tables; processing ...\n")

    for table in mapping_tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col["name"] for col in cursor.fetchall()]

        try:
            # Special case: Tasks_to_DWAs has 4 columns; we want Task ID <-> DWA ID
            if table == "Tasks_to_DWAs":
                child_key, parent_key = "Task ID", "DWA ID"
            elif len(columns) >= 3:
                child_key, parent_key = columns[0], columns[2]
            else:
                child_key, parent_key = columns[0], columns[1]

            print(f"Processing [{table}] (parent='{parent_key}', child='{child_key}') ...")

            cursor.execute(f"SELECT * FROM {table}")
            inverted = defaultdict(list)
            for row in cursor.fetchall():
                parent = row[parent_key]
                child = row[child_key]
                if parent and child:
                    inverted[parent].append(child)

            file_path = os.path.join(OUTPUT_DIR, f"{table}_Inverted.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(inverted, f, indent=2, ensure_ascii=False)

            print(f"  -> wrote {len(inverted)} parent nodes\n")
        except Exception as e:
            print(f"  ! failed on {table}: {e}\n")

    print("Done. Inverted mapping trees written to", OUTPUT_DIR)
    conn.close()


if __name__ == "__main__":
    build_mappings()
