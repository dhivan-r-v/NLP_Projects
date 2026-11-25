"""
FastAPI backend for the Customer Feedback Analysis project.

Loads the pre-trained TF-IDF vectorizer + Logistic Regression model from
`models/` (no training happens here) and exposes:

    GET  /api/health         liveness + model status
    GET  /api/model          model metadata (classes, vocabulary size, ...)
    POST /api/predict        analyze one piece of feedback
    POST /api/predict/batch  analyze up to 500 feedback items at once

It also serves the frontend from `frontend/` at `/`.

Run from the project root:
    .venv\\Scripts\\python -m uvicorn src.api.app:app --reload --port 8000
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.data.preprocess import clean_text, clean_batch

# --------------------------------------------------------------------------
# Paths (absolute, so the server works no matter where it is launched from)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"

MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

MAX_TEXT_LEN = 5000
MAX_BATCH = 500

# --------------------------------------------------------------------------
# Load artifacts once at startup
# --------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

CLASSES: List[str] = [str(c) for c in model.classes_]
FEATURE_NAMES = vectorizer.get_feature_names_out()
COEF = model.coef_  # shape: (n_classes, n_features)

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class Review(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class Keyword(BaseModel):
    term: str
    weight: float


class Token(BaseModel):
    token: str
    weight: float | None  # None = word not in the model's vocabulary


class Prediction(BaseModel):
    text: str
    clean_text: str
    tokens: List[Token]
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    supporting_terms: List[Keyword]
    opposing_terms: List[Keyword]
    low_signal: bool


class BatchResponse(BaseModel):
    count: int
    summary: dict[str, int]
    average_confidence: float
    top_terms: dict[str, List[Keyword]]
    results: List[Prediction]
    elapsed_ms: float


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------


def _explain(row, class_idx: int, k: int = 6):
    """Per-term contribution = tfidf value * class coefficient."""
    row = row.tocsr()
    idx = row.indices
    if idx.size == 0:
        return [], []
    contrib = row.data * COEF[class_idx, idx]
    order = np.argsort(contrib)
    supporting = [
        Keyword(term=str(FEATURE_NAMES[idx[i]]), weight=round(float(contrib[i]), 4))
        for i in order[::-1][:k]
        if contrib[i] > 0
    ]
    opposing = [
        Keyword(term=str(FEATURE_NAMES[idx[i]]), weight=round(float(contrib[i]), 4))
        for i in order[:k]
        if contrib[i] < 0
    ]
    return supporting, opposing


def _token_weights(row, clean: str, class_idx: int) -> List[Token]:
    """Contribution of each unigram in the cleaned text toward the predicted class."""
    row = row.tocsr()
    values = dict(zip(row.indices, row.data))
    vocab = vectorizer.vocabulary_
    out = []
    for tok in clean.split():
        j = vocab.get(tok)
        w = None if j is None else round(float(values.get(j, 0.0) * COEF[class_idx, j]), 4)
        out.append(Token(token=tok, weight=w))
    return out


def _analyze(raw_texts: List[str], cleaned: List[str]) -> List[Prediction]:
    X = vectorizer.transform(cleaned)
    probas = model.predict_proba(X)
    out: List[Prediction] = []
    for i, (raw, clean) in enumerate(zip(raw_texts, cleaned)):
        p = probas[i]
        ci = int(np.argmax(p))
        row = X[i]
        supporting, opposing = _explain(row, ci)
        out.append(
            Prediction(
                text=raw,
                clean_text=clean,
                tokens=_token_weights(row, clean, ci),
                sentiment=CLASSES[ci],
                confidence=round(float(p[ci]), 4),
                probabilities={c: round(float(v), 4) for c, v in zip(CLASSES, p)},
                supporting_terms=supporting,
                opposing_terms=opposing,
                # No known vocabulary terms -> the model is effectively guessing
                low_signal=row.nnz == 0,
            )
        )
    return out


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(
    title="Customer Feedback Analysis API",
    version="1.0.0",
    description="Sentiment analysis for customer reviews using a TF-IDF + Logistic Regression model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/api/model")
def model_info():
    return {
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "classes": CLASSES,
        "vocabulary_size": int(len(FEATURE_NAMES)),
        "ngram_range": list(getattr(vectorizer, "ngram_range", (1, 1))),
        "max_text_length": MAX_TEXT_LEN,
        "max_batch_size": MAX_BATCH,
    }


@app.post("/api/predict", response_model=Prediction)
def predict(review: Review):
    return _analyze([review.text], [clean_text(review.text)])[0]


@app.post("/api/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    start = time.perf_counter()
    texts = [t for t in (str(x).strip() for x in req.texts) if t]
    if not texts:
        raise HTTPException(status_code=422, detail="No non-empty texts provided.")
    texts = [t[:MAX_TEXT_LEN] for t in texts]

    # Same normalisation as clean_text, then spaCy in batch mode (much faster)
    pre = [re.sub(r"[^a-zA-Z ]", "", re.sub(r"http\S+", "", t.lower())) for t in texts]
    cleaned = clean_batch(pre)

    results = _analyze(texts, cleaned)
    summary = Counter(r.sentiment for r in results)

    # Most influential terms per class across the whole batch
    term_scores: dict[str, Counter] = {c: Counter() for c in CLASSES}
    for r in results:
        for kw in r.supporting_terms:
            term_scores[r.sentiment][kw.term] += kw.weight
    top_terms = {
        c: [Keyword(term=t, weight=round(w, 4)) for t, w in term_scores[c].most_common(10)]
        for c in CLASSES
    }

    return BatchResponse(
        count=len(results),
        summary={c: summary.get(c, 0) for c in CLASSES},
        average_confidence=round(float(np.mean([r.confidence for r in results])), 4),
        top_terms=top_terms,
        results=results,
        elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
    )


# Backwards-compatible route from the original app
@app.post("/predict")
def predict_legacy(review: Review):
    r = predict(review)
    return {"sentiment": r.sentiment, "confidence": r.confidence}


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
