"""
Classifies pasted resume text with the trained model.

Run: python predict.py
"""

from pathlib import Path

import joblib

from preprocess import clean_text

MODEL_PATH = Path(__file__).parent / "model" / "resume_classifier.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        # The model is a build artifact, not committed. Saying so beats a bare
        # FileNotFoundError, which was what this did before.
        raise SystemExit(
            f"No model at {MODEL_PATH}.\n"
            "Train one first:\n"
            "  python data/generate_dataset.py\n"
            "  python train.py"
        )

    model = joblib.load(MODEL_PATH)

    print("Paste resume text and press enter. Type 'exit' to quit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue

        cleaned = clean_text(text)
        if not cleaned:
            print("  nothing usable in that input\n")
            continue

        prediction = model.predict([cleaned])[0]

        # The probability matters as much as the label: on text unlike anything
        # in the training data the model still returns its best guess, and a low
        # confidence is the only signal that the guess is weak.
        confidence = max(model.predict_proba([cleaned])[0])
        print(f"  {prediction}  ({confidence:.0%} confidence)\n")


if __name__ == "__main__":
    main()
