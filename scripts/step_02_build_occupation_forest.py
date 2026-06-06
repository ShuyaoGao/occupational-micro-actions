"""Step 02 — Build a per-SOC "occupation forest" (one JSON per occupation).

For every O*NET-SOC code in `Occupation_Data`, this script joins every
attribute table that has a `O*NET-SOC Code` column and writes one JSON file
under data/output_02_occupation_forest/.

Input:  data/output_01_onet_30_2.db
Output: data/output_02_occupation_forest/Occupation_{soc}.json  (~1,000 files)
"""
import sqlite3
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)
DB_PATH    = os.path.join(PROJ, "data", "output_01_onet_30_2.db")
OUTPUT_DIR = os.path.join(PROJ, "data", "output_02_occupation_forest")


def build_forest():
    print(">>> Step 02: building per-SOC occupation forest ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Collect all tables that carry the O*NET-SOC Code key (i.e. occupation-level tables)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row["name"] for row in cursor.fetchall()]
    core_tables = []
    for table in all_tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [col["name"] for col in cursor.fetchall()]
        if "O*NET-SOC Code" in cols and table != "Occupation_Data":
            core_tables.append(table)

    # Iterate over every occupation and assemble its JSON tree
    cursor.execute("SELECT * FROM Occupation_Data")
    occupations = [dict(row) for row in cursor.fetchall()]
    print(f"Found {len(core_tables)} attribute tables; "
          f"generating {len(occupations)} occupation trees.")

    for index, occ in enumerate(occupations, start=1):
        soc_code = occ["O*NET-SOC Code"]
        tree = {"Occupation_Info": occ}
        for table in core_tables:
            cursor.execute(
                f'SELECT * FROM {table} WHERE "O*NET-SOC Code" = ?',
                (soc_code,),
            )
            data = [dict(row) for row in cursor.fetchall()]
            if data:
                tree[table] = data

        file_path = os.path.join(
            OUTPUT_DIR, f"Occupation_{soc_code.replace('.', '_')}.json"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

        if index % 200 == 0:
            print(f"  progress: {index} / {len(occupations)}")

    print("Done. Occupation forest written to", OUTPUT_DIR)
    conn.close()


if __name__ == "__main__":
    build_forest()
