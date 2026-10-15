# CareerPulse AI
## AI-Powered Job Market, Skill Intelligence & Career Analytics Platform

CareerPulse AI is a Data Science project that analyzes job-market data, extracts skills from job descriptions, predicts salary ranges, calculates student skill gaps, and recommends career roles.

### Main Features
1. Job market analytics
2. NLP-based skill extraction
3. Skill demand analysis
4. Salary prediction
5. Job category classification
6. Personal skill-gap analyzer
7. Career recommendation
8. Interactive Streamlit dashboard
9. SQLite database
10. Model training and saving

### Project Structure

CareerPulse_AI/
- data/job_data.csv              Sample job dataset
- models/                        Trained ML models
- src/clean_data.py              Data cleaning
- src/eda.py                     EDA charts
- src/database.py                SQLite database
- src/nlp_skills.py              Skill extraction
- src/train_models.py            ML training
- src/career_engine.py           Skill gap + career recommendation
- dashboard/app.py               Streamlit dashboard
- requirements.txt
- run_project.py                 One-command setup/training

### Installation

Open terminal in the project folder:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
python run_project.py
streamlit run dashboard/app.py
```

The dashboard will open in the browser.

### Notes
The included dataset is a small demonstration dataset so the project runs immediately. For a final academic project, replace it with a larger public/licensed job dataset and document its source and date.
