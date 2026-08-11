"""
Job requirement definitions used by the Hiring Assistant.
Each job posting is represented as a structured dict so the scoring
and feedback modules can reason about required vs preferred skills,
experience, and education instead of just raw text.
"""

JOB_POSTINGS = {
    "data_scientist": {
        "title": "Data Scientist",
        "description": (
            "We are looking for a Data Scientist to build predictive models, "
            "run statistical analysis, and communicate insights to stakeholders. "
            "The ideal candidate is comfortable with the full ML lifecycle from "
            "data cleaning to deployment."
        ),
        "required_skills": [
            "python", "machine learning", "pandas", "numpy", "sql",
            "scikit-learn", "statistics"
        ],
        "preferred_skills": [
            "deep learning", "tensorflow", "pytorch", "nlp", "aws",
            "docker", "tableau", "power bi"
        ],
        "min_experience_years": 1,
        "education": ["b.e.", "b.tech", "m.tech", "msc", "bsc", "bachelor", "master"],
    },
    "ml_engineer": {
        "title": "Machine Learning Engineer",
        "description": (
            "We need an ML Engineer to design, train, and deploy production "
            "machine learning and deep learning systems, and own the pipeline "
            "from experimentation to serving."
        ),
        "required_skills": [
            "python", "machine learning", "deep learning", "pytorch",
            "tensorflow", "git", "sql"
        ],
        "preferred_skills": [
            "docker", "kubernetes", "aws", "mlops", "nlp", "computer vision",
            "flask", "fastapi"
        ],
        "min_experience_years": 2,
        "education": ["b.e.", "b.tech", "m.tech", "msc", "bachelor", "master"],
    },
    "software_engineer": {
        "title": "Software Engineer",
        "description": (
            "We are hiring a Software Engineer to design and build scalable "
            "backend services and APIs, write clean maintainable code, and "
            "collaborate across teams."
        ),
        "required_skills": [
            "python", "java", "sql", "git", "data structures", "algorithms",
            "rest api"
        ],
        "preferred_skills": [
            "docker", "aws", "microservices", "react", "javascript", "system design"
        ],
        "min_experience_years": 0,
        "education": ["b.e.", "b.tech", "bca", "bachelor"],
    },
}


def get_job(job_key: str) -> dict:
    if job_key not in JOB_POSTINGS:
        raise KeyError(f"Unknown job posting: {job_key}")
    return JOB_POSTINGS[job_key]


def list_jobs():
    return {k: v["title"] for k, v in JOB_POSTINGS.items()}
