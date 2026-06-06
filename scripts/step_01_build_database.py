"""Step 01 — Build the O*NET 30.2 relational SQLite database.

Reads every .txt file from the O*NET 30.2 raw download and loads it as a
table in a single SQLite database, preserving original column types as
strings (to avoid SOC codes such as "11-1011.00" being parsed as floats).

Input:  data/onet_30_2_raw/*.txt          (41 tab-delimited files)
Output: data/output_01_onet_30_2.db       (single SQLite database)
"""
import os
import pandas as pd
import sqlite3

# --- Path configuration (anchored to this file, no cwd assumption) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(_HERE)                                # repo root
ONET_DATA_DIR = os.path.join(PROJ, "data", "onet_30_2_raw")    # see README for download
DB_PATH = os.path.join(PROJ, "data", "output_01_onet_30_2.db")


def build_relational_db():
    print("Building O*NET relational database ...")
    conn = sqlite3.connect(DB_PATH)

    file_list = [f for f in os.listdir(ONET_DATA_DIR) if f.endswith(".txt")]
    total_files = len(file_list)

    for index, filename in enumerate(file_list, start=1):
        file_path = os.path.join(ONET_DATA_DIR, filename)
        # SQL-safe table name: strip .txt and replace spaces/hyphens/commas
        table_name = (
            filename.replace(".txt", "")
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace(",", "")
        )
        print(f"[{index}/{total_files}] Loading table: {table_name} ...")
        try:
            # O*NET files are tab-separated; load everything as string first.
            df = pd.read_csv(file_path, sep="\t", dtype=str)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        except Exception as e:
            print(f"  ! Error reading/writing {filename}: {e}")

    conn.commit()
    conn.close()
    print("-" * 40)
    print(f"Done. Relational database written to: {DB_PATH}")


if __name__ == "__main__":
    build_relational_db()
