CAREERS = {
    "Data Analyst": ["python","sql","excel","power bi","statistics"],
    "Data Scientist": ["python","sql","statistics","machine learning","pandas","numpy"],
    "ML Engineer": ["python","machine learning","pytorch","tensorflow","docker","aws"],
    "Cloud Engineer": ["linux","aws","docker","kubernetes","networking"],
    "Software Engineer": ["java","python","sql","git","dsa"],
    "Data Engineer": ["python","sql","etl","airflow","spark","aws"],
    "Cybersecurity Analyst": ["linux","python","networking","siem","sql"],
    "NLP Engineer": ["python","nlp","transformers","pytorch","sql"]
}

def normalize(skills):
    return {str(s).strip().lower() for s in skills if str(s).strip()}

def skill_gap(user_skills, required_skills):
    user = normalize(user_skills)
    required = normalize(required_skills)
    matched = sorted(user & required)
    missing = sorted(required - user)
    score = round((len(matched) / len(required)) * 100, 2) if required else 0
    return matched, missing, score

def recommend_careers(user_skills):
    user = normalize(user_skills)
    results = []
    for career, required in CAREERS.items():
        req = set(required)
        matched = user & req
        score = round(len(matched) / len(req) * 100, 2)
        results.append({
            "career": career,
            "match": score,
            "matched": ", ".join(sorted(matched)),
            "missing": ", ".join(sorted(req - user))
        })
    return sorted(results, key=lambda x: x["match"], reverse=True)
