# Advanced Machine Learning — Assignment 1

This repository contains an NMF robustness experiment for face-image
reconstruction. It includes two NumPy-based NMF implementations, ORL and
Extended YaleB data loaders, block-occlusion generation, reproducible
evaluation, automated tests, and report-ready result plotting.

The assignment permits the Python standard library, NumPy, and SciPy for
implementing NMF algorithms. Scikit-learn is included only for the optional
evaluation metrics/clustering workflow described in the assignment, and
Matplotlib is included for experiment visualizations.

## Data

Do not commit the ORL or Extended YaleB datasets. Download or obtain them
separately and place them under `code/data/`. Dataset-loading and preprocessing
code is included under `code/experiments/`.

## Running the code

Install the dependencies from `requirements.txt`, then follow the commands in
`code/README.md`. Run those commands from the `code/` directory. Generated
outputs and local dataset files are excluded from version control.
