# Email Spam / Ham Classification — Spam Shield

Classifies emails / SMS messages as **spam** or **ham** with a Multinomial Naive Bayes model, served through a FastAPI backend and a web frontend.

## Results (held-out test set, 1,031 messages)

| Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|
| 96.7% | 83.0% | 93.1% | 87.8% |

## Pipeline

1. **Preprocess** (`src/data/process_dataset.py`) — lowercase, strip URLs and punctuation, spaCy stop-word removal + lemmatization.
2. **Vectorize** (`src/features/vectorizer.py`) — `CountVectorizer` with unigrams + bigrams, 5,000 features.
3. **Train** (`src/models/train.py`) — `MultinomialNB`, training split balanced with `RandomOverSampler`.
4. **Evaluate** (`src/models/evaluate.py`) — classification report + confusion matrix.
5. **Serve** (`src/api/app.py`) — FastAPI API + web UI; messages predicted as spam are written to `data/quarantine/quarantine.csv`.

## Web app

- **Analyze** — verdict, spam score, and the words that pushed the decision toward spam or ham.
- **Batch** — classify up to 100 messages at once (paste or upload .txt/.csv), export to CSV.
- **Quarantine** — every message flagged as spam.
- **Model** — test metrics, confusion matrix, strongest spam / ham terms.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m uvicorn src.api.app:app --port 8001
```

Then open http://localhost:8001 (or double-click `run_app.bat` on Windows).

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/predict` | `{"text": "..."}` → label, spam probability, top signals |
| POST | `/api/predict/batch` | `{"texts": ["...", "..."]}` → results for each message |
| GET | `/api/model-info` | Test metrics, confusion matrix, top terms |
| GET / DELETE | `/api/quarantine` | List / clear quarantined messages |
| POST | `/predict` | Original endpoint, returns `{"label": ...}` |
