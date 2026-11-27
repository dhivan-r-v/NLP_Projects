import pandas as pd
import joblib
 
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from src.models.build_model import build_model
 
 
df = pd.read_csv("data/processed/email_spam_processed.csv")
df["clean_email"] = df["clean_email"].fillna("")
 
# Load the vectorizer prepared in Stage 4 — transform only, never re-fit.
vectorizer = joblib.load("models/count_vectorizer.pkl")
 
X = vectorizer.transform(df["clean_email"])
y = df["label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
# Balance the TRAIN set only (test set stays untouched).
print("Before:", Counter(y_train))
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
print("After:", Counter(y_train_bal))

# Build the untrained model from Stage 5, then fit it here.
model = build_model()
 
model.fit(X_train_bal, y_train_bal)
 
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "models/spam_model.pkl")
 
print("Training Completed")
