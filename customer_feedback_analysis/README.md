# Customer Feedback Analysis

Sentiment classification (positive / neutral / negative) for customer product reviews, served through a FastAPI backend with a web frontend.

- **Data:** Amazon product reviews (`data/raw/7817_1.csv`), labelled with VADER and cleaned with spaCy (`data/processed/cleaned_reviews.csv`).
- **Model:** TF-IDF (10,000 terms, 1–2-grams) + logistic regression, saved in `models/`.
- **App:** FastAPI (`src/api/app.py`) serves the model and the frontend (`frontend/`).

## Project layout

```
customer_feedback_analysis/
├── data/              raw and processed review CSVs
├── models/            sentiment_model.pkl, tfidf_vectorizer.pkl
├── src/
│   ├── data/          loading, VADER labelling, text cleaning
│   ├── features/      TF-IDF vectorizer
│   ├── models/        train, evaluate, CLI predict
│   └── api/app.py     FastAPI backend
├── frontend/          index.html, styles.css, app.js
├── main.py            rebuilds data/processed from data/raw
└── run_app.bat        starts the web app on Windows
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the web app

```bash
python -m uvicorn src.api.app:app --port 8000
```

Open http://127.0.0.1:8000. API docs are at `/docs`. On Windows you can double-click `run_app.bat` instead.

The app has three views:

- **Single:** classify one review, see class probabilities, which words pushed the decision, and the cleaned text the model actually read.
- **Batch:** paste up to 500 reviews or load a CSV/TXT file, get the sentiment split, top words per class, a filterable table and CSV export.
- **History:** recent single analyses, stored in the browser.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | – | server status |
| GET | `/api/model` | – | model type, classes, vocabulary size |
| POST | `/api/predict` | `{"text": "..."}` | sentiment, confidence, probabilities, per-word weights |
| POST | `/api/predict/batch` | `{"texts": ["...", "..."]}` | per-review results plus summary |
| POST | `/predict` | `{"text": "..."}` | sentiment and confidence (original endpoint) |

## Retraining (optional)

The app uses the saved models as they are. To rebuild them:

```bash
python main.py                    # clean data/raw -> data/processed
python -m src.models.train        # fit TF-IDF + logistic regression, save to models/
```

## Known limitations

Labels come from VADER scores rather than human ratings, so the model inherits VADER's blind spots: mixed reviews ("great sound but the app crashes") lean positive, and short complaints or plain factual statements are sometimes misclassified.
