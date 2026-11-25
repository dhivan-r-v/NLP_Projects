import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed/cleaned_reviews.csv")

# Remove rows where clean_review became empty/NaN during cleaning
df = df.dropna(subset=["clean_review"])

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X = vectorizer.fit_transform(df["clean_review"])

# Ensure the models directory exists before dumping
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print(X.shape)
