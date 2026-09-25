# Code

This directory contains the assignment source code.

## Required structure

- `algorithm/`: implementations of at least two NMF algorithms.
- `data/`: local copies of the ORL and Extended YaleB datasets. Dataset files
  must not be committed.
- `tests/`: automated tests for algorithms and supporting utilities.

The implementation must use Python 3 and may use the standard library, NumPy,
and SciPy. Scikit-learn may be used for evaluation only, not for NMF
implementations.

## Evaluation pipeline

Run commands from this `code/` directory. The data loaders expect the datasets
at `data/ORL/` and `data/CroppedYaleB/`.

Install dependencies:

```bash
python -m pip install -r ../requirements.txt
```

Run all self-contained unit tests:

```bash
python -m unittest discover -s tests -v
```

Alternatively, from the repository root:

```bash
python -m pytest -q
```

Run a small real-data smoke experiment before committing to the full grid:

```bash
python run_experiments.py \
  --datasets ORL \
  --runs 1 \
  --sample-frac 0.1 \
  --max-iter 5 \
  --block-sizes 5 \
  --block-counts 1
```

Run the required repeated ORL and YaleB experiments:

```bash
python run_experiments.py \
  --runs 5 \
  --sample-frac 0.9 \
  --max-iter 200 \
  --block-sizes 10 15 20 \
  --size-sweep-count 1 \
  --block-counts 1 2 3 \
  --count-sweep-size 10
```

Non-overlapping placement is enforced by default. An impossible placement raises
a clear error instead of silently overlapping blocks. Add `--allow-overlap` only
for a separately documented experiment. The runner records both the requested
and actual covered fractions. Use `--clustering-metrics` to add the optional
Accuracy and NMI evaluation.

The runner checks relative loss convergence with a conservative tolerance of
`1e-5`. Use `--no-early-stopping` to force all `--max-iter` updates, or adjust
`--tol`, `--min-iter`, and `--patience` when documenting another convergence
rule.

After the experiment finishes, generate aggregate CSV/LaTeX tables, RRE plots,
and reconstruction panels:

```bash
python plot_results.py \
  --results outputs/evaluation/results.csv \
  --output-dir outputs/evaluation
```

The experiment runner checkpoints `results.csv`, metadata, and sampled column
indices after every condition so a long YaleB run retains completed work. Use
`--resume` with the same arguments to continue a matching checkpoint. Use
`--overwrite` only when intentionally replacing an existing run.
