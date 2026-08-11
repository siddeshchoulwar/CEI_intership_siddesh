"""
Retrieval-Augmented Feedback Generator
---------------------------------------
Implements a lightweight retrieval-based text generation pipeline:

  1. KNOWLEDGE BASE: a bank of advice "snippets" tagged by skill/topic
     (e.g. what to say about a missing 'docker' skill, how to phrase
     an experience gap, etc).
  2. RETRIEVAL: given a candidate's score breakdown, retrieve the most
     relevant snippets using TF-IDF + cosine similarity over the
     snippet bank, keyed off the actual missing/matched skills.
  3. GENERATION: the retrieved snippets are stitched into a coherent,
     personalized, explainable feedback report using templated
     natural-language generation (grounded entirely in retrieved
     content, not hallucinated).

This mirrors a real RAG architecture (retriever + generator) without
requiring an external LLM API.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Knowledge base of advice snippets, keyed by generic topic tags ---
KNOWLEDGE_BASE = [
    {"tag": "skill_gap", "skill": "python",
     "text": "Python is the backbone of most data workflows; build a couple of small "
             "end-to-end scripts or a portfolio project to demonstrate fluency."},
    {"tag": "skill_gap", "skill": "machine learning",
     "text": "Strengthen machine learning fundamentals with a project that covers the "
             "full pipeline: data cleaning, feature engineering, model training, and evaluation."},
    {"tag": "skill_gap", "skill": "deep learning",
     "text": "Deep learning experience stands out when you can point to a trained model "
             "(CNN, RNN, or transformer-based) with clear metrics and a short write-up."},
    {"tag": "skill_gap", "skill": "sql",
     "text": "SQL is expected for almost every data role; practicing joins, window "
             "functions, and query optimization will close this gap quickly."},
    {"tag": "skill_gap", "skill": "docker",
     "text": "Containerizing even one project with Docker signals you understand "
             "reproducible deployment, which recruiters specifically screen for."},
    {"tag": "skill_gap", "skill": "aws",
     "text": "Cloud experience (AWS/GCP/Azure) is increasingly a baseline requirement; "
             "a free-tier project deployed to the cloud is a strong, low-cost signal."},
    {"tag": "skill_gap", "skill": "nlp",
     "text": "NLP shows up heavily in this role; a project involving text classification, "
             "embeddings, or a RAG pipeline would directly address this gap."},
    {"tag": "skill_gap", "skill": "pytorch",
     "text": "Familiarity with a deep learning framework like PyTorch or TensorFlow is "
             "expected for hands-on model building beyond scikit-learn."},
    {"tag": "skill_gap", "skill": "system design",
     "text": "For engineering-heavy roles, being able to reason about system design "
             "trade-offs (scalability, latency, storage) matters as much as coding ability."},
    {"tag": "experience_gap",
     "text": "The experience level is a bit below what's typically expected; internships, "
             "freelance work, or substantial open-source contributions can help bridge this."},
    {"tag": "education_mismatch",
     "text": "The educational background listed doesn't clearly map to the role's usual "
             "requirements; highlighting relevant coursework or certifications can help."},
    {"tag": "strong_match",
     "text": "The skill overlap with this role is strong, which is the single biggest "
             "predictor of getting past an initial resume screen."},
    {"tag": "general_tip",
     "text": "Quantify impact wherever possible (e.g. 'reduced processing time by 30%') "
             "since concrete numbers make a resume far more persuasive than skill lists alone."},
    {"tag": "general_tip",
     "text": "Tailoring the resume's top section to mirror language from the job "
             "description improves both ATS keyword matching and recruiter skim-readability."},
]

_tfidf = TfidfVectorizer(stop_words="english")
_corpus_texts = [item["text"] for item in KNOWLEDGE_BASE]
_tfidf_matrix = _tfidf.fit_transform(_corpus_texts)


def retrieve(query: str, top_k: int = 3):
    """TF-IDF cosine-similarity retrieval over the knowledge base."""
    q_vec = _tfidf.transform([query])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return [KNOWLEDGE_BASE[i] for i in ranked[:top_k] if sims[i] > 0]


def generate_feedback(resume: dict, job: dict, score_result: dict) -> dict:
    """
    Retrieval + generation: pulls relevant snippets for each missing
    skill / gap, then assembles a structured, explainable feedback
    report (strengths, gaps, and prioritized action items).
    """
    strengths, gaps, action_items = [], [], []

    if score_result["matched_required"]:
        strengths.append(
            "Strong alignment on core required skills: "
            + ", ".join(score_result["matched_required"]) + "."
        )
    if score_result["matched_preferred"]:
        strengths.append(
            "Good bonus coverage on preferred skills: "
            + ", ".join(score_result["matched_preferred"]) + "."
        )
    if score_result["features"]["education_match"]:
        strengths.append("Education background aligns with what this role typically expects.")
    if not strengths:
        hits = retrieve("strong match skill overlap", top_k=1)

    # Missing required skills -> retrieve targeted advice for each, but
    # only from snippets actually tagged as skill_gap advice for that
    # skill (otherwise fall back to a generic message instead of an
    # unrelated high-similarity snippet).
    skill_gap_kb = [item for item in KNOWLEDGE_BASE if item["tag"] == "skill_gap"]
    for skill in score_result["missing_required"]:
        direct_hit = next((item for item in skill_gap_kb if item["skill"] == skill), None)
        if direct_hit:
            gaps.append(f"Missing required skill — {skill}: {direct_hit['text']}")
        else:
            gaps.append(f"Missing required skill: {skill}. Consider a small project "
                        f"or course to build demonstrable experience here.")
        action_items.append(f"Build proof-of-work for '{skill}' (small project or certification).")

    # Missing preferred skills -> lighter-weight mention
    if score_result["missing_preferred"]:
        gaps.append(
            "Preferred-but-missing skills that would strengthen the profile: "
            + ", ".join(score_result["missing_preferred"]) + "."
        )

    if score_result["features"]["experience_ratio"] < 0.5:
        hits = retrieve("experience gap internship", top_k=1)
        if hits:
            gaps.append(hits[0]["text"])

    if not score_result["features"]["education_match"]:
        hits = retrieve("education mismatch certification", top_k=1)
        if hits:
            gaps.append(hits[0]["text"])

    general_tips = retrieve("quantify impact tailor resume keywords", top_k=2)
    for tip in general_tips:
        action_items.append(tip["text"])

    summary = (
        f"Overall fit for {job['title']}: {score_result['final_label']} "
        f"({score_result['final_score']}/100). "
        f"{len(score_result['matched_required'])}/{len(job['required_skills'])} required "
        f"skills matched."
    )

    return {
        "summary": summary,
        "strengths": strengths or ["No strong standout matches found yet — see gaps below."],
        "gaps": gaps or ["No major gaps detected against this job posting."],
        "action_items": action_items[:5],
    }
