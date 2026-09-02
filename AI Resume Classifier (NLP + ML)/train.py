"""
Trains the resume classifier and writes the model and its scores to disk.

Run: python train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import clean_text

HERE = Path(__file__).parent
DATA = HERE / "data" / "resumes.csv"
MODEL_DIR = HERE / "model"
MODEL_PATH = MODEL_DIR / "resume_classifier.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            f"{DATA} is missing. Run: python data/generate_dataset.py"
        )

    df = pd.read_csv(DATA)
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["category"]

    # stratify keeps every category represented in both splits. It also fails
    # loudly on a dataset too small to split, which is the correct behaviour —
    # a "model" trained on one example per class is not a model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # A single split on a small corpus is noisy, so the cross-validated mean is
    # reported alongside it — that is the number worth quoting.
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"rows              {len(df)}")
    print(f"categories        {sorted(y.unique())}")
    print(f"holdout accuracy  {accuracy:.3f}")
    print(f"5-fold accuracy   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print()
    print(classification_report(y_test, y_pred))
    print("confusion matrix (rows = actual, cols = predicted)")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=sorted(y.unique()),
        columns=sorted(y.unique()),
    ))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Written out so the README can quote a number that came from a run rather
    # than from memory.
    METRICS_PATH.write_text(
        json.dumps(
            {
                "rows": int(len(df)),
                "categories": sorted(y.unique().tolist()),
                "holdout_accuracy": round(float(accuracy), 4),
                "cv5_mean_accuracy": round(float(cv_scores.mean()), 4),
                "cv5_std": round(float(cv_scores.std()), 4),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nsaved model   -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
