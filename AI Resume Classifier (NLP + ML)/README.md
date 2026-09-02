# Resume Classifier

Sorts resume text into one of five job families using TF-IDF features and
logistic regression.

```
> performed penetration testing and hardened server configurations
  Cybersecurity  (62% confidence)

> built regression models and tuned hyperparameters with cross validation
  Data Science  (60% confidence)
```

## Run it

```bash
pip install -r requirements.txt
python data/generate_dataset.py   # builds the corpus
python train.py                   # trains, scores, saves the model
python predict.py                 # classify pasted text
```

## The data is synthetic, and that matters

Real resumes are personal data, so this trains on a generated corpus rather
than scraped ones. `data/generate_dataset.py` is committed alongside its
output — a CSV of unknown provenance is not evidence of anything.

The generator deliberately makes the task harder than it would naturally be:

- Every category shares filler ("collaborated with stakeholders", "worked in an
  agile team"), so common resume language carries no signal.
- Tools overlap across fields — Python appears under Data Science, Software
  Engineering and Cybersecurity; SQL under four of the five.
- **45% of resumes borrow a line from a different field**, because career
  changers and hybrid roles are normal and a corpus without them is separable
  by keyword alone.

That last point was added after the fact. The first version of the generator
gave each role its own private vocabulary and the model scored a **perfect
1.00** — which measured the dataset, not the model.

## Results

From the most recent run of `train.py`, written to `model/metrics.json`:

| | |
|---|---|
| Rows | 300 across 5 categories |
| Holdout accuracy | **0.973** |
| 5-fold cross-validated | **0.983** (± 0.011) |

**Read that number carefully.** ~98% on generated text says the pipeline works
end to end — cleaning, vectorising, fitting, scoring, persisting. It says
nothing about accuracy on real resumes, which are longer, messier, full of
formatting noise, and not drawn from a phrase bank. Treat it as a sanity check,
not a benchmark.

The more interesting behaviour is confidence. Given text from a field it
knows, the model returns roughly 60–65%. Given something unrelated:

```
> I enjoy long walks on the beach and baking sourdough bread
  Cybersecurity  (22% confidence)
```

It still returns a label, because logistic regression always does — but at 22%
against a 20% chance baseline for five classes, it is effectively saying it has
no idea. `predict.py` prints the confidence for exactly this reason: the label
alone would look confident and be meaningless.

## What is here

| File | |
|---|---|
| `data/generate_dataset.py` | Builds the corpus. Seeded, so it reproduces |
| `preprocess.py` | Lowercase, strip non-letters, collapse whitespace |
| `train.py` | Trains, prints a report and confusion matrix, saves model and metrics |
| `predict.py` | Interactive classification with confidence |

`model/` is a build artifact and is not committed — `train.py` creates it.

## What would make this real

Honest list, in order:

1. A real dataset. That means resumes with consent, or a public licensed
   corpus, and revisiting every number above.
2. Handling documents rather than sentences — PDF and DOCX extraction, section
   detection, and the formatting noise that comes with them.
3. A confidence floor that returns "unclear" instead of a label, rather than
   leaving the caller to interpret the percentage.
4. More classes. Five job families is a coarse split; real hiring taxonomies
   run to dozens and overlap far more.
