# Advanced Machine Learning — Assignment 1

This repository is the starter project for the NMF robustness assignment on
face-image reconstruction. It intentionally contains project structure and
environment setup only; algorithm implementations, experiments, results, and
the final report are not included.

## Project layout

```text
.
├── code/
│   ├── algorithm/       # NMF implementations will be added here
│   ├── data/            # Place datasets here locally; dataset files are not committed
│   ├── tests/           # Tests for the implementation
│   └── README.md
├── requirements.txt
└── .gitignore
```

## Environment setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The assignment permits the Python standard library, NumPy, and SciPy for
implementing NMF algorithms. Scikit-learn is included only for the optional
evaluation metrics/clustering workflow described in the assignment, and
Matplotlib is included for experiment visualizations.

## Data

Do not commit the ORL or Extended YaleB datasets. Download or obtain them
separately and place them under `code/data/`. Dataset-loading and preprocessing
code will be added as part of the assignment implementation.

## Implementation status

This is a setup scaffold. Add the algorithms, data pipeline, experiments,
and tests in subsequent work. The report will be written and submitted
externally.
