"""
Spam / Ham classification API + web frontend.

Run from the project root:
    .venv\\Scripts\\python -m uvicorn src.api.app:app --reload --port 8001
Then open http://localhost:8001
"""

import time
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data.preprocess import clean_text
from src.quarantine.store import clear_quarantine, quarantine_email, read_quarantine

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
MAX_CHARS = 20_000

app = FastAPI(title="Spam Shield API", version="1.0.0")

model = joblib.load(ROOT / "models" / "spam_model.pkl")
vectorizer = joblib.load(ROOT / "models" / "count_vectorizer.pkl")

CLASSES = list(model.classes_)
SPAM_IDX = CLASSES.index("spam")
HAM_IDX = CLASSES.index("ham")
FEATURES = vectorizer.get_feature_names_out()
# Per-term evidence: log P(term|spam) - log P(term|ham). Positive = spammy.
TERM_WEIGHT = model.feature_log_prob_[SPAM_IDX] - model.feature_log_prob_[HAM_IDX]


class Email(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)


class EmailBatch(BaseModel):
    texts: List[str] = Field(..., max_length=100)


def analyse(text: str, quarantine: bool = True) -> dict:
    start = time.perf_counter()
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    spam_p = float(proba[SPAM_IDX])
    label = CLASSES[int(np.argmax(proba))]

    # Which known terms pushed the decision, and how hard
    row = X.tocoo()
    signals = [
        {"term": FEATURES[j], "count": int(c), "weight": round(float(c * TERM_WEIGHT[j]), 3)}
        for j, c in zip(row.col, row.data)
    ]
    signals.sort(key=lambda s: abs(s["weight"]), reverse=True)

    if label == "spam" and quarantine:
        quarantine_email(text, label, spam_p)

    return {
        "label": label,
        "spam_probability": round(spam_p, 4),
        "confidence": round(float(proba.max()), 4),
        "cleaned_text": cleaned,
        "known_terms": int(X.nnz),
        "signals": signals[:12],
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }


# ---------- API ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "model": type(model).__name__, "classes": CLASSES}


@app.post("/predict")
def predict_legacy(email: Email):
    """Original endpoint, kept for backward compatibility."""
    return {"label": analyse(email.text)["label"]}


@app.post("/api/predict")
def predict(email: Email):
    if not email.text.strip():
        raise HTTPException(422, "Email text is empty.")
    return analyse(email.text)


@app.post("/api/predict/batch")
def predict_batch(batch: EmailBatch):
    texts = [t for t in batch.texts if t.strip()]
    if not texts:
        raise HTTPException(422, "No non-empty emails supplied.")
    results = []
    for t in texts:
        r = analyse(t[:MAX_CHARS])
        results.append({"text": t, "label": r["label"], "spam_probability": r["spam_probability"]})
    return {"count": len(results), "spam": sum(r["label"] == "spam" for r in results), "results": results}


@lru_cache(maxsize=1)
def _model_info():
    """Evaluate on the same held-out split used in training (random_state=42)."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ROOT / "data" / "processed" / "email_spam_processed.csv")
    df["clean_email"] = df["clean_email"].fillna("")
    X = vectorizer.transform(df["clean_email"])
    y = df["label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pred = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, pos_label="spam", average="binary")
    cm = confusion_matrix(y_test, pred, labels=["spam", "ham"]).tolist()

    order = np.argsort(TERM_WEIGHT)
    return {
        "model": type(model).__name__,
        "vectorizer": f"{type(vectorizer).__name__} (ngram {vectorizer.ngram_range}, {len(FEATURES)} features)",
        "dataset": {k: int(v) for k, v in y.value_counts().items()},
        "test_size": int(len(y_test)),
        "metrics": {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
        },
        "confusion_matrix": {"labels": ["spam", "ham"], "matrix": cm},
        "top_spam_terms": [FEATURES[i] for i in order[::-1][:15]],
        "top_ham_terms": [FEATURES[i] for i in order[:15]],
    }


@app.get("/api/model-info")
def model_info():
    return _model_info()


@app.get("/api/quarantine")
def quarantine_list(limit: int = 50):
    return {"items": read_quarantine(limit)}


@app.delete("/api/quarantine")
def quarantine_clear():
    clear_quarantine()
    return {"cleared": True}


# ---------- Frontend ----------

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")
