import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("data/processed/email_spam_processed.csv")
df["clean_email"] = df["clean_email"].fillna("")

X = df["clean_email"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Before:", Counter(y_train))

# vectorize first (e.g. TF-IDF), then oversample the TRAIN part only
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_vec, y_train)
print("After:", Counter(y_train_bal))