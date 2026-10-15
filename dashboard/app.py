from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nlp_skills import skill_counts, extract_skills
from career_engine import CAREERS, skill_gap, recommend_careers

st.set_page_config(page_title="CareerPulse AI", page_icon="🚀", layout="wide")

@st.cache_data
def load_data():
    path = ROOT / "data" / "clean_jobs.csv"
    return pd.read_csv(path)

df = load_data()

st.title("🚀 CareerPulse AI")
st.caption("AI-Powered Job Market, Skill Intelligence & Career Analytics Platform")

# Sidebar
st.sidebar.header("Filters")
locations = ["All"] + sorted(df["location"].dropna().unique().tolist())
industries = ["All"] + sorted(df["industry"].dropna().unique().tolist())

location = st.sidebar.selectbox("Location", locations)
industry = st.sidebar.selectbox("Industry", industries)

filtered = df.copy()
if location != "All":
    filtered = filtered[filtered["location"] == location]
if industry != "All":
    filtered = filtered[filtered["industry"] == industry]

tabs = st.tabs([
    "🏠 Overview", "📊 Job Market", "🧠 Skill Intelligence",
    "💰 Salary Analytics", "🎯 Skill Gap", "🚀 Career Recommendation",
    "🔎 Job Search"
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Jobs", len(filtered))
    c2.metric("Companies", filtered["company"].nunique())
    c3.metric("Locations", filtered["location"].nunique())
    c4.metric("Industries", filtered["industry"].nunique())

    st.subheader("Job Distribution")
    counts = filtered["industry"].value_counts().reset_index()
    counts.columns = ["industry", "jobs"]
    st.plotly_chart(px.pie(counts, names="industry", values="jobs", hole=.45),
                    use_container_width=True)

with tabs[1]:
    st.subheader("Jobs by Location")
    counts = filtered["location"].value_counts().reset_index()
    counts.columns = ["location", "jobs"]
    st.plotly_chart(px.bar(counts, x="location", y="jobs"),
                    use_container_width=True)

    st.subheader("Popular Job Roles")
    roles = filtered["job_title"].value_counts().reset_index()
    roles.columns = ["job_title", "jobs"]
    st.plotly_chart(px.bar(roles.head(15), x="jobs", y="job_title", orientation="h"),
                    use_container_width=True)

with tabs[2]:
    st.subheader("🔥 Most Demanded Skills")
    counts = skill_counts(filtered).head(20)
    st.plotly_chart(px.bar(counts, x="demand", y="skill", orientation="h"),
                    use_container_width=True)

    text = st.text_area("Paste a job description to extract skills:")
    if text:
        found = extract_skills(text)
        st.success("Detected skills: " + (", ".join(found) if found else "None"))

with tabs[3]:
    st.subheader("Salary Analytics")
    st.plotly_chart(
        px.box(filtered, x="industry", y="salary_avg",
               title="Average Salary Distribution"),
        use_container_width=True
    )
    st.plotly_chart(
        px.scatter(filtered, x="experience", y="salary_avg",
                   size="salary_avg", hover_name="job_title",
                   title="Experience vs Salary"),
        use_container_width=True
    )

with tabs[4]:
    st.subheader("🎯 Personal Skill-Gap Analyzer")
    all_skills = sorted(set(s for values in CAREERS.values() for s in values))
    user_skills = st.multiselect("Select your current skills", all_skills)
    target = st.selectbox("Target Career", list(CAREERS.keys()))

    matched, missing, score = skill_gap(user_skills, CAREERS[target])

    st.metric("Skill Match", f"{score}%")
    c1, c2 = st.columns(2)
    with c1:
        st.write("### ✅ Matched")
        st.write(matched if matched else "No matched skills yet")
    with c2:
        st.write("### ⚠️ Skill Gap")
        st.write(missing if missing else "No gap for this career")

with tabs[5]:
    st.subheader("🚀 Career Recommendation Engine")
    all_skills = sorted(set(s for values in CAREERS.values() for s in values))
    user_skills = st.multiselect("Choose your skills", all_skills, key="recommend")
    results = recommend_careers(user_skills)
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("🔎 Search Jobs")
    query = st.text_input("Search by job title, skill, company or description")
    result = filtered.copy()
    if query:
        mask = (
            result["job_title"].str.contains(query, case=False, na=False) |
            result["skills"].str.contains(query, case=False, na=False) |
            result["company"].str.contains(query, case=False, na=False) |
            result["description"].str.contains(query, case=False, na=False)
        )
        result = result[mask]
    st.write(f"{len(result)} jobs found")
    st.dataframe(
        result[["job_title","company","location","experience","skills","salary_min","salary_max","industry"]],
        use_container_width=True, hide_index=True
    )

st.divider()
st.caption("CareerPulse AI | Academic prototype. Salary figures are dataset-based estimates, not employment guarantees.")
