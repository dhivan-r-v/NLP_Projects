import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/cleaned_reviews.csv")

# 1. Drop rows where clean_review or sentiment is missing
df = df.dropna(subset=["clean_review", "sentiment"])

X = df["clean_review"]
y = df["sentiment"]

# 2. Split BEFORE vectorizing to prevent data leakage
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Fit vectorizer ONLY on training text, then transform test text
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 4. Train model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# 5. Print evaluation metrics so you know how well it performed
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save model and vectorizer safely
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Training Completed")
