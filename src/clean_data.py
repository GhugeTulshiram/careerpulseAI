from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "job_data.csv"
OUTPUT = DATA / "clean_jobs.csv"

def clean():
    df = pd.read_csv(INPUT)
    df = df.drop_duplicates()
    required = ["job_title", "skills", "description"]
    df = df.dropna(subset=required).copy()

    for col in ["job_title", "skills", "description", "company", "location", "industry"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")
    df = df.dropna(subset=["salary_min", "salary_max"])
    df = df[(df["salary_min"] >= 0) & (df["salary_max"] >= df["salary_min"])]
    df["salary_avg"] = (df["salary_min"] + df["salary_max"]) / 2
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} cleaned jobs to {OUTPUT}")
    return df

if __name__ == "__main__":
    clean()
