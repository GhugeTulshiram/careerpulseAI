import re
from collections import Counter
from pathlib import Path
import pandas as pd

SKILLS = [
    "python","java","c++","sql","excel","power bi","tableau",
    "machine learning","deep learning","tensorflow","pytorch",
    "pandas","numpy","aws","azure","docker","kubernetes","git",
    "nlp","statistics","scikit-learn","transformers","linux",
    "networking","dsa","javascript","react","node.js","etl",
    "airflow","spark","siem","a/b testing","ci/cd"
]

def extract_skills(text):
    text = str(text).lower()
    found = []
    for skill in SKILLS:
        if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", text):
            found.append(skill)
    return found

def skill_counts(df):
    counter = Counter()
    for text in df["description"].fillna(""):
        counter.update(extract_skills(text))
    return pd.DataFrame(counter.most_common(), columns=["skill", "demand"])

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "clean_jobs.csv")
    print(skill_counts(df).head(20))
