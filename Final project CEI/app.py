"""
AI-Powered Intelligent Hiring Assistant
-----------------------------------------
Streamlit front-end tying together:
  - resume_parser  : extracts structured signal from an uploaded resume
  - scoring_model   : ML (Logistic Regression) + DL (MLP) fit scoring
  - rag_feedback    : retrieval-based, explainable feedback generation
  - chatbot         : conversational interface over the candidate's result

Run with:  streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import resume_parser
import scoring_model
import rag_feedback
import chatbot
from job_requirements import list_jobs, get_job

st.set_page_config(page_title="AI Hiring Assistant", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------
# Load / train models once, cache across reruns
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    return scoring_model.train_models()


logreg, mlp, tfidf = load_models()
metrics = scoring_model.get_metrics()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🧠 Hiring Assistant")
st.sidebar.caption(
    "Evaluates a resume against a job's requirements using ML + DL models, "
    "then generates explainable feedback through a retrieval-based pipeline."
)

jobs = list_jobs()
job_key = st.sidebar.selectbox(
    "Job posting", options=list(jobs.keys()), format_func=lambda k: jobs[k]
)
job = get_job(job_key)

with st.sidebar.expander("Model performance (on synthetic eval set)"):
    if metrics:
        st.write(f"Logistic Regression — accuracy {metrics['logreg_accuracy']}, "
                 f"F1 {metrics['logreg_f1']}")
        st.write(f"Neural Net (MLP) — accuracy {metrics['mlp_accuracy']}, "
                 f"F1 {metrics['mlp_f1']}")

with st.sidebar.expander("Job requirements"):
    st.write(f"**Required:** {', '.join(job['required_skills'])}")
    st.write(f"**Preferred:** {', '.join(job['preferred_skills'])}")
    st.write(f"**Min. experience:** {job['min_experience_years']} yrs")

st.title("AI-Powered Intelligent Hiring Assistant")
st.write(job["description"])

# ---------------------------------------------------------------------
# Resume input
# ---------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Candidate resume")
    uploaded = st.file_uploader("Upload resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    pasted_text = st.text_area("...or paste resume text directly", height=200)

run = st.button("Evaluate resume", type="primary")

if run:
    if uploaded is not None:
        raw_bytes = uploaded.read()
        text = resume_parser.extract_text_from_bytes(raw_bytes, uploaded.name)
    elif pasted_text.strip():
        text = pasted_text
    else:
        st.warning("Upload a resume or paste resume text first.")
        st.stop()

    parsed = resume_parser.parse_resume(text)
    result = scoring_model.score_resume(parsed, job, logreg, mlp, tfidf)
    feedback = rag_feedback.generate_feedback(parsed, job, result)

    # stash in session state so the chat interface can reference it
    st.session_state["parsed"] = parsed
    st.session_state["result"] = result
    st.session_state["feedback"] = feedback
    st.session_state["job_title"] = job["title"]
    st.session_state["chat_history"] = []

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]
    feedback = st.session_state["feedback"]
    parsed = st.session_state["parsed"]

    st.divider()
    st.subheader("2. Evaluation")

    m1, m2, m3 = st.columns(3)
    m1.metric("Final fit score", f"{result['final_score']}/100", result["final_label"])
    m2.metric("ML model (Logistic Regression)", f"{result['ml_model']['score']}/100",
               result["ml_model"]["label"])
    m3.metric("DL model (Neural Net)", f"{result['dl_model']['score']}/100",
               result["dl_model"]["label"])

    st.progress(min(result["final_score"] / 100, 1.0))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Matched required skills**")
        st.write(", ".join(result["matched_required"]) or "None")
        st.markdown("**✅ Matched preferred skills**")
        st.write(", ".join(result["matched_preferred"]) or "None")
    with c2:
        st.markdown("**❌ Missing required skills**")
        st.write(", ".join(result["missing_required"]) or "None")
        st.markdown("**❌ Missing preferred skills**")
        st.write(", ".join(result["missing_preferred"]) or "None")

    st.divider()
    st.subheader("3. Explainable, retrieval-generated feedback")
    st.info(feedback["summary"])

    st.markdown("**Strengths**")
    for s in feedback["strengths"]:
        st.markdown(f"- {s}")

    st.markdown("**Gaps**")
    for g in feedback["gaps"]:
        st.markdown(f"- {g}")

    st.markdown("**Recommended next steps**")
    for a in feedback["action_items"]:
        st.markdown(f"- {a}")

    with st.expander("Extracted resume data"):
        st.json({
            "email": parsed["email"],
            "phone": parsed["phone"],
            "skills": parsed["skills"],
            "experience_years": parsed["experience_years"],
            "education": parsed["education"],
        })

    # -------------------------------------------------------------
    # Chatbot
    # -------------------------------------------------------------
    st.divider()
    st.subheader("4. Ask the assistant about your evaluation")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    user_msg = st.chat_input("e.g. 'why is my score low?' or 'what should I improve?'")
    if user_msg:
        st.session_state["chat_history"].append(("user", user_msg))
        reply = chatbot.respond(
            user_msg, result, feedback, st.session_state["job_title"]
        )
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()
else:
    st.info("Upload or paste a resume, then click **Evaluate resume** to get started.")
