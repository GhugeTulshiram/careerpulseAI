from pathlib import Path
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "clean_jobs.csv")

fig = px.bar(
    df["location"].value_counts().head(10).reset_index(),
    x="location", y="count",
    title="Top Job Locations"
)
fig.write_html(OUT / "top_locations.html")

fig2 = px.box(
    df, x="industry", y="salary_avg",
    title="Salary Distribution by Industry"
)
fig2.write_html(OUT / "salary_by_industry.html")

print("EDA reports saved in reports/")
