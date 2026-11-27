import csv
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUARANTINE_DIR = os.path.join(ROOT, "data", "quarantine")
QUARANTINE_FILE = os.path.join(QUARANTINE_DIR, "quarantine.csv")


def quarantine_email(email_text, label, spam_probability=None):

    os.makedirs(QUARANTINE_DIR, exist_ok=True)

    new_file = not os.path.exists(QUARANTINE_FILE)

    with open(QUARANTINE_FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        if new_file:
            writer.writerow(["timestamp", "label", "spam_probability", "email_text"])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            label,
            "" if spam_probability is None else f"{spam_probability:.4f}",
            email_text,
        ])


def read_quarantine(limit=50):

    if not os.path.exists(QUARANTINE_FILE):
        return []

    with open(QUARANTINE_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return list(reversed(rows))[:limit]


def clear_quarantine():

    if os.path.exists(QUARANTINE_FILE):
        os.remove(QUARANTINE_FILE)
