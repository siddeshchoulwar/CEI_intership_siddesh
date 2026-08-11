"""
Conversational Interface
-------------------------
A lightweight intent-classification chatbot that lets a candidate ask
natural-language questions about their evaluation ("why was my score
low?", "what skills am I missing?", "how do I improve?") and get
answers grounded in that candidate's actual score_result + feedback,
instead of a generic canned response.

Intent detection uses TF-IDF + cosine similarity against a small set
of labeled example utterances per intent (fast, no external API,
fully explainable).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENT_EXAMPLES = {
    "score_explanation": [
        "why did i get this score", "explain my score", "how was my score calculated",
        "what does my fit score mean", "why is my score low", "why is my score high",
    ],
    "missing_skills": [
        "what skills am i missing", "what should i learn", "which skills do i lack",
        "what are my skill gaps", "what do i need to add to my resume",
    ],
    "strengths": [
        "what are my strengths", "what did i do well", "what matched well",
        "what are my good points",
    ],
    "improvement_tips": [
        "how can i improve", "what should i do next", "give me tips",
        "how do i get a better score", "what should i focus on",
    ],
    "matched_skills": [
        "what skills matched", "which of my skills are relevant",
        "what skills do i already have that fit",
    ],
    "greeting": ["hi", "hello", "hey", "good morning"],
}

_intent_texts, _intent_labels = [], []
for label, examples in INTENT_EXAMPLES.items():
    for ex in examples:
        _intent_texts.append(ex)
        _intent_labels.append(label)

_vectorizer = TfidfVectorizer()
_intent_matrix = _vectorizer.fit_transform(_intent_texts)


def classify_intent(message: str, threshold: float = 0.15) -> str:
    vec = _vectorizer.transform([message.lower()])
    sims = cosine_similarity(vec, _intent_matrix)[0]
    best_idx = sims.argmax()
    if sims[best_idx] < threshold:
        return "unknown"
    return _intent_labels[best_idx]


def respond(message: str, score_result: dict, feedback: dict, job_title: str) -> str:
    intent = classify_intent(message)

    if intent == "greeting":
        return "Hi! Ask me about your score, missing skills, strengths, or how to improve."

    if intent == "score_explanation":
        f = score_result["features"]
        return (
            f"Your overall fit for {job_title} is **{score_result['final_label']}** "
            f"({score_result['final_score']}/100). This blends an ML model (Logistic "
            f"Regression: {score_result['ml_model']['score']}/100) and a neural network "
            f"(MLP: {score_result['dl_model']['score']}/100), driven by required-skill "
            f"match ({f['skill_match_ratio']*100:.0f}%), preferred-skill match "
            f"({f['preferred_match_ratio']*100:.0f}%), experience ratio "
            f"({f['experience_ratio']*100:.0f}%), education match "
            f"({'yes' if f['education_match'] else 'no'}), and resume-JD text similarity "
            f"({f['tfidf_similarity']*100:.0f}%)."
        )

    if intent == "missing_skills":
        missing = score_result["missing_required"] + score_result["missing_preferred"]
        if not missing:
            return "You're not missing any tracked skills for this role — nice work."
        return "You're missing: " + ", ".join(missing) + ". " + (
            feedback["gaps"][0] if feedback["gaps"] else ""
        )

    if intent == "matched_skills":
        matched = score_result["matched_required"] + score_result["matched_preferred"]
        if not matched:
            return "No tracked skills matched this posting yet."
        return "Skills that matched this role: " + ", ".join(matched) + "."

    if intent == "strengths":
        return " ".join(feedback["strengths"])

    if intent == "improvement_tips":
        return " ".join(f"- {tip}" for tip in feedback["action_items"])

    return (
        "I can answer questions about your score, matched skills, missing skills, "
        "strengths, or how to improve — try asking one of those."
    )
