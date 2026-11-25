import joblib
from src.data.preprocess import clean_text

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Type 'exit' or 'quit' to stop.")

while True:
    review = input("\nEnter Review: ").strip()

    if review.lower() in {"exit", "quit"}:
        break

    if not review:
        continue

    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X).max() * 100

    print(f"Prediction: {prediction} ({confidence:.1f}% confidence)")
