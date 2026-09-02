"""
Builds data/resumes.csv.

Real resumes are personal data, so this generates a synthetic corpus instead.
The generator is committed rather than just its output so the dataset can be
regenerated and inspected — a CSV of unknown provenance is not evidence of
anything.

The categories deliberately SHARE vocabulary. Every role writes "collaborated
with stakeholders", several use Python, and Data Science and Software
Engineering overlap heavily on tooling. Without that overlap each class would
have its own private words, TF-IDF would separate them perfectly, and the
reported accuracy would say nothing about the model.
"""

import csv
import random
from pathlib import Path

random.seed(20260802)  # reproducible corpus

# Phrases that genuinely distinguish a role.
SPECIFIC = {
    "Data Science": [
        "built regression models to forecast quarterly demand",
        "trained a gradient boosting classifier on customer churn data",
        "ran A/B tests and reported statistical significance",
        "cleaned and imputed missing values across large datasets",
        "used pandas and numpy for exploratory data analysis",
        "tuned hyperparameters with cross validation",
        "presented model performance to non technical stakeholders",
        "engineered features from raw transaction logs",
    ],
    "Software Engineering": [
        "designed and shipped REST APIs consumed by mobile clients",
        "refactored a legacy monolith into separate services",
        "wrote unit and integration tests to raise coverage",
        "reviewed pull requests and mentored junior developers",
        "implemented caching to reduce database load",
        "built responsive interfaces in React and TypeScript",
        "debugged production incidents and wrote postmortems",
        "designed relational schemas and wrote migrations",
    ],
    "Cybersecurity": [
        "performed penetration testing on internal web applications",
        "triaged vulnerability scan findings and tracked remediation",
        "hardened server configurations and rotated credentials",
        "investigated phishing reports and contained incidents",
        "wrote detection rules for the SIEM platform",
        "led a tabletop exercise for incident response",
        "reviewed code for injection and authentication flaws",
        "implemented least privilege access across environments",
    ],
    "Marketing": [
        "planned and ran paid social campaigns within budget",
        "improved organic search ranking through on page SEO",
        "wrote email sequences and measured open and click rates",
        "managed the content calendar across three channels",
        "briefed designers on campaign creative",
        "reported campaign performance against acquisition targets",
        "ran customer interviews to refine positioning",
        "coordinated a product launch with sales and support",
    ],
    "IT Support": [
        "resolved hardware and software tickets within SLA",
        "imaged and deployed laptops for new starters",
        "administered user accounts and group permissions",
        "documented common fixes in the internal knowledge base",
        "supported video conferencing and meeting room equipment",
        "escalated network faults to the infrastructure team",
        "managed the asset inventory and licence renewals",
        "walked non technical staff through remote troubleshooting",
    ],
}

# Deliberately shared across roles — this is what makes the task non-trivial.
SHARED = [
    "collaborated with cross functional stakeholders",
    "worked in an agile team with two week sprints",
    "communicated progress in daily stand ups",
    "documented work for future maintainers",
    "used git for version control",
    "took part in retrospectives and planning",
    "balanced competing priorities under deadline",
    "wrote python scripts to automate repetitive work",
]

# Tools that appear in more than one field, so no single token gives the answer.
AMBIGUOUS = {
    "Data Science": ["python", "sql", "jupyter", "tableau"],
    "Software Engineering": ["python", "sql", "docker", "typescript"],
    "Cybersecurity": ["python", "linux", "docker", "wireshark"],
    "Marketing": ["excel", "google analytics", "hubspot", "sql"],
    "IT Support": ["windows", "linux", "active directory", "excel"],
}

PER_CATEGORY = 60

# How often a resume borrows a line from a different field. Career changers,
# hybrid roles and generalists are normal, and a corpus without them is
# separable by keyword alone — the first version of this generator scored a
# perfect 1.00, which measured the dataset rather than the model.
CROSSOVER_RATE = 0.45


def build_resume(category: str) -> str:
    """One resume: a little role-specific signal, buried in shared filler."""
    # Deliberately few distinctive lines. Give a resume five of them and the
    # classifier only has to spot one.
    specific = random.sample(SPECIFIC[category], k=random.randint(1, 3))
    shared = random.sample(SHARED, k=random.randint(2, 4))
    tools = random.sample(AMBIGUOUS[category], k=random.randint(2, 3))

    parts = specific + shared + [f"experienced with {', '.join(tools)}"]

    # A line from a neighbouring field, the way a real resume carries traces of
    # adjacent work.
    if random.random() < CROSSOVER_RATE:
        other = random.choice([c for c in SPECIFIC if c != category])
        parts.append(random.choice(SPECIFIC[other]))

    random.shuffle(parts)
    return ". ".join(parts) + "."


def main() -> None:
    rows = [
        (build_resume(category), category)
        for category in SPECIFIC
        for _ in range(PER_CATEGORY)
    ]
    random.shuffle(rows)

    out = Path(__file__).parent / "resumes.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows across {len(SPECIFIC)} categories -> {out}")


if __name__ == "__main__":
    main()
