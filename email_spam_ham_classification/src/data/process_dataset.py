import pandas as pd
from src.data.preprocess import clean_batch
 
def process_dataset():
 
    print("Script Started")
    df = pd.read_csv("data/raw/email_spam.csv", encoding="cp1252")[["v1", "v2"]]
    df.columns = ["label", "email_text"]
    df = df.drop_duplicates(subset="email_text").dropna()

    print("Dataset Loaded")
 
    # Lowercasing + regex are fast as vectorized pandas ops,
    # so we do them here and leave only tokenizing/lemmatizing to spaCy.
 
    texts = df["email_text"].astype(str).str.lower()
    texts = texts.str.replace(r"http\S+|www\S+", " ", regex=True)
    texts = texts.str.replace(r"[^a-z\s]", " ", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()
    
    df["clean_email"] = clean_batch(texts)
    df = df[df["clean_email"].str.len() > 0]
    df = df[["label", "clean_email"]]
    
    print("Cleaning Completed")
 
    df.to_csv("data/processed/email_spam_processed.csv", index=False)
 
    print("File Saved")
    print(df["label"].value_counts())
