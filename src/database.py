from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "careerpulse.db"

def init_db():
    conn = sqlite3.connect(DB)
    df = pd.read_csv(ROOT / "data" / "clean_jobs.csv")
    df.to_sql("jobs", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Database created: {DB}")

if __name__ == "__main__":
    init_db()
