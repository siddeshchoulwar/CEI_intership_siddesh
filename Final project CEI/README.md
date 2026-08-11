# AI-Powered Intelligent Hiring Assistant

An end-to-end system that evaluates a candidate resume against a job's
requirements using **ML and DL models**, generates **personalized, explainable
feedback** through a **retrieval-based text generation pipeline**, and offers a
**conversational interface** for the candidate to ask about their result.

## Architecture

```
Resume (PDF/DOCX/TXT)
        │
        ▼
 resume_parser.py ──► skills, experience, education, contact info
        │
        ▼
 scoring_model.py ──► feature engineering (skill overlap, experience ratio,
        │              education match, TF-IDF resume↔JD similarity)
        │
        ├──► ML model:  Logistic Regression  (classical)
        └──► DL model:  MLP neural network    (deep learning)
        │
        ▼
 rag_feedback.py ───► RETRIEVAL: TF-IDF search over an advice knowledge base
        │              GENERATION: templated, grounded feedback report
        ▼
 chatbot.py ────────► intent-classification chatbot answering questions
                       about the score, using the same result + feedback
        │
        ▼
   app.py (Streamlit UI)
```

### Why this counts as ML *and* DL
`scoring_model.py` trains two models side by side on the same engineered
feature vector — a `LogisticRegression` (classical ML) and an
`MLPClassifier` feed-forward neural network (a compact DL model) — and blends
their outputs into the final fit score. Both are trained on a synthetically
generated but rule-grounded dataset (`_generate_synthetic_dataset`), so the
project is fully self-contained with no external data or API dependency.

### Why this counts as retrieval-based generation (RAG-style)
`rag_feedback.py` keeps a small knowledge base of advice snippets tagged by
skill/topic. For each gap in a candidate's profile, it **retrieves** the most
relevant snippet(s) via TF-IDF cosine similarity, then **generates** a
feedback report by assembling those snippets with candidate-specific details
— i.e. retriever + generator, without needing an external LLM.

## Project structure

```
hiring_assistant/
├── app.py                    # Streamlit UI (entry point)
├── requirements.txt
├── models/                   # trained model artifacts (.joblib)
├── sample_data/               # example resumes to try
└── src/
    ├── job_requirements.py    # job posting definitions
    ├── resume_parser.py       # PDF/DOCX/TXT parsing + skill/entity extraction
    ├── scoring_model.py       # feature engineering + ML/DL models
    ├── rag_feedback.py        # retrieval-based feedback generation
    └── chatbot.py             # conversational Q&A interface
```

## Setup & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run trains and caches the ML/DL models (a few seconds); later
runs load the cached `.joblib` files from `models/`.

Try it with the sample resumes in `sample_data/` (upload the `.txt` files,
or open and paste their contents) against the "Data Scientist" posting.

## Extending it

- **More jobs**: add entries to `JOB_POSTINGS` in `src/job_requirements.py`.
- **More skills**: extend `SKILL_VOCAB` in `src/resume_parser.py`.
- **More feedback snippets**: extend `KNOWLEDGE_BASE` in `src/rag_feedback.py`.
- **Real LLM generation**: swap the templated generator in
  `rag_feedback.generate_feedback` for a call to an LLM API, using the
  retrieved snippets as grounding context (the retrieval step doesn't
  need to change).
- **Real training data**: replace `_generate_synthetic_dataset` in
  `scoring_model.py` with a labeled dataset of real resume/JD/outcome
  triples once available.
