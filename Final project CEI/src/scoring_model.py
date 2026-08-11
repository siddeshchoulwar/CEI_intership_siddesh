"""
Scoring Models
--------------
Two models score how well a resume fits a job:

1. A classical ML model (Logistic Regression) trained on engineered
   features: skill overlap ratio, experience gap, education match,
   and TF-IDF cosine similarity between resume and JD text.
2. A small feed-forward neural network (MLPClassifier, i.e. a compact
   deep learning model) trained on the same feature space, so we can
   compare "ML vs DL" behaviour as required by the assignment brief.

Both are trained on a synthetically generated but realistic dataset
(no external data needed), then saved to /models so the app doesn't
retrain on every request.
"""
import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

from job_requirements import JOB_POSTINGS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def build_features(resume: dict, job: dict, tfidf: TfidfVectorizer) -> np.ndarray:
    """
    Turn (parsed resume, job posting) into a fixed-length numeric
    feature vector:
      [skill_match_ratio, preferred_match_ratio, experience_ratio,
       education_match, tfidf_cosine_sim]
    """
    resume_skills = set(resume.get("skills", []))
    required = set(job["required_skills"])
    preferred = set(job["preferred_skills"])

    skill_match_ratio = len(resume_skills & required) / max(len(required), 1)
    preferred_match_ratio = len(resume_skills & preferred) / max(len(preferred), 1)

    exp_years = resume.get("experience_years", 0)
    min_exp = max(job["min_experience_years"], 0.5)
    experience_ratio = min(exp_years / min_exp, 2.0) / 2.0  # cap & normalize 0-1

    edu = set(resume.get("education", []))
    education_match = 1.0 if edu & set(job["education"]) else 0.0

    resume_text = resume.get("raw_text", "") or " ".join(resume_skills)
    job_text = job["description"] + " " + " ".join(required | preferred)
    try:
        vecs = tfidf.transform([resume_text, job_text])
        sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
    except Exception:
        sim = 0.0

    return np.array([
        skill_match_ratio, preferred_match_ratio,
        experience_ratio, education_match, sim,
    ])


# ---------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------
def _generate_synthetic_dataset(n_per_job=300, seed=42):
    """
    Generates synthetic candidate feature rows + fit labels for each
    job posting by randomly sampling skill subsets, experience, and
    education, then labeling via a rule-based ground truth. This
    gives the models something non-trivial to learn without needing
    a real labeled hiring dataset.
    """
    rng = random.Random(seed)
    tfidf_corpus = []
    for job in JOB_POSTINGS.values():
        tfidf_corpus.append(job["description"] + " " + " ".join(
            job["required_skills"] + job["preferred_skills"]))

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf.fit(tfidf_corpus + [
        "generic resume text with some skills and experience mentioned"
    ])

    X, y = [], []
    for job in JOB_POSTINGS.values():
        required = job["required_skills"]
        preferred = job["preferred_skills"]
        for _ in range(n_per_job):
            n_req = rng.randint(0, len(required))
            n_pref = rng.randint(0, len(preferred))
            has_skills = rng.sample(required, n_req) + rng.sample(preferred, n_pref)
            exp_years = round(rng.uniform(0, 6), 1)
            has_edu = rng.random() > 0.15  # most candidates meet edu bar

            resume = {
                "raw_text": job["description"] + " " + " ".join(has_skills)
                if rng.random() > 0.3 else " ".join(has_skills),
                "skills": has_skills,
                "experience_years": exp_years,
                "education": job["education"][:1] if has_edu else [],
            }
            feats = build_features(resume, job, tfidf)

            # Rule-based ground-truth label (0=weak,1=moderate,2=strong)
            skill_ratio = feats[0]
            if skill_ratio >= 0.7 and exp_years >= job["min_experience_years"] and has_edu:
                label = 2
            elif skill_ratio >= 0.4 and (has_edu or exp_years >= job["min_experience_years"] * 0.5):
                label = 1
            else:
                label = 0
            X.append(feats)
            y.append(label)

    return np.array(X), np.array(y), tfidf


# ---------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------
def train_models(force=False):
    ml_path = os.path.join(MODELS_DIR, "logreg_model.joblib")
    dl_path = os.path.join(MODELS_DIR, "mlp_model.joblib")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf.joblib")

    if not force and all(os.path.exists(p) for p in [ml_path, dl_path, tfidf_path]):
        return (joblib.load(ml_path), joblib.load(dl_path), joblib.load(tfidf_path))

    X, y, tfidf = _generate_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    ml_acc = accuracy_score(y_test, logreg.predict(X_test))
    ml_f1 = f1_score(y_test, logreg.predict(X_test), average="macro")

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", max_iter=800,
        random_state=42, early_stopping=True,
    )
    mlp.fit(X_train, y_train)
    dl_acc = accuracy_score(y_test, mlp.predict(X_test))
    dl_f1 = f1_score(y_test, mlp.predict(X_test), average="macro")

    joblib.dump(logreg, ml_path)
    joblib.dump(mlp, dl_path)
    joblib.dump(tfidf, tfidf_path)

    metrics = {
        "logreg_accuracy": round(ml_acc, 3), "logreg_f1": round(ml_f1, 3),
        "mlp_accuracy": round(dl_acc, 3), "mlp_f1": round(dl_f1, 3),
    }
    joblib.dump(metrics, os.path.join(MODELS_DIR, "metrics.joblib"))
    return logreg, mlp, tfidf


def get_metrics():
    path = os.path.join(MODELS_DIR, "metrics.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


LABELS = {0: "Weak Fit", 1: "Moderate Fit", 2: "Strong Fit"}


def score_resume(resume: dict, job: dict, logreg, mlp, tfidf) -> dict:
    feats = build_features(resume, job, tfidf).reshape(1, -1)

    ml_pred = int(logreg.predict(feats)[0])
    ml_proba = logreg.predict_proba(feats)[0]

    dl_pred = int(mlp.predict(feats)[0])
    dl_proba = mlp.predict_proba(feats)[0]

    # Blend both models into a single 0-100 fit score for display
    ml_score = float(np.dot(ml_proba, [0, 50, 100]))
    dl_score = float(np.dot(dl_proba, [0, 50, 100]))
    blended = round((ml_score + dl_score) / 2, 1)

    required = set(job["required_skills"])
    preferred = set(job["preferred_skills"])
    resume_skills = set(resume.get("skills", []))

    return {
        "features": {
            "skill_match_ratio": round(float(feats[0][0]), 2),
            "preferred_match_ratio": round(float(feats[0][1]), 2),
            "experience_ratio": round(float(feats[0][2]), 2),
            "education_match": bool(feats[0][3]),
            "tfidf_similarity": round(float(feats[0][4]), 2),
        },
        "ml_model": {"label": LABELS[ml_pred], "score": round(ml_score, 1)},
        "dl_model": {"label": LABELS[dl_pred], "score": round(dl_score, 1)},
        "final_score": blended,
        "final_label": LABELS[max(ml_pred, dl_pred) if ml_pred != dl_pred else ml_pred],
        "matched_required": sorted(resume_skills & required),
        "missing_required": sorted(required - resume_skills),
        "matched_preferred": sorted(resume_skills & preferred),
        "missing_preferred": sorted(preferred - resume_skills),
    }


if __name__ == "__main__":
    logreg, mlp, tfidf = train_models(force=True)
    print("Training complete. Metrics:", get_metrics())
