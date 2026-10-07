# Advanced Machine Learning — Assignment 1

Group 39's comparison of Standard NMF and smoothed L1-loss NMF for face-image
reconstruction under white block occlusion. Both algorithms are implemented
from scratch using NumPy. The robust method minimises a smoothed entry-wise
absolute reconstruction loss; it does not use an explicit sparse-error matrix
or L1 regularisation on the factors.

The assignment permits the Python standard library, NumPy, and SciPy for
implementing NMF algorithms. Scikit-learn is included only for the optional
evaluation metrics/clustering workflow described in the assignment, and
Matplotlib is included for experiment visualizations.

## Repository contents

- `code/algorithm/`: the two NMF implementations.
- `code/experiments/`: image loading, preprocessing, noise and visualisation.
- `code/evaluation.py`: clean-reference RRE and supplementary clustering metrics.
- `code/run_experiments.py`: matched trials, deterministic seeds and checkpoints.
- `code/plot_results.py`: result aggregation, tables and figures.
- `code/tests/`: self-contained algorithm, utility and pipeline tests.
- `code/final_results/`: retained summary CSV, LaTeX table and report figures.
- `code/data/`: local dataset location; dataset files are excluded from Git.

The full report PDF and its main LaTeX source are not stored in this repository.
`code/final_results/rre_summary_table.tex` is only a generated table fragment.

## Setup and data

From this repository's root, using Python 3 (the reported environment used
Python 3.12.2):

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Do not commit the ORL or Extended YaleB datasets. Download or obtain them
separately and place the subject directories under `code/data/ORL/` and
`code/data/CroppedYaleB/`. See [code/README.md](code/README.md) for the exact
layout, a dataset-free demo, a small smoke run and the full reproduction command.

## Reported experiment

The final experiment contains 120 fits: two datasets, five matched trials, six
unique conditions (including clean data), and two algorithms. Each trial samples
90% of the images using stratified sampling. Both algorithms receive identical
samples, corruption and initial factors within each condition.

The reported settings are 200 updates with early stopping disabled, base seed
4328, robust smoothing 0.001, and ranks 40 (ORL) and 38 (YaleB). Blocks are
non-overlapping and filled with 255 before normalisation. The size sweep uses
10, 15 and 20 pixel edges with one block; the count sweep uses one, two and three
10-pixel blocks. Tables report means and sample standard deviations over five
trials. Accuracy and NMI follow the assignment tutorial's majority-label mapping.

Standard NMF gives lower clean-data RRE. Smoothed L1-loss NMF reduces mean RRE
by 26.5–60.7% for the 10-pixel block-count conditions, but loses its advantage
for a single 20-pixel block on both datasets and a 15-pixel block on ORL.
Clustering quality does not consistently follow reconstruction quality.

The original saved experiment records about 2 hours 21 minutes of factorisation
time. A fresh full run with one numerical-library thread completed in **30 minutes
36 seconds including plotting** in the tested environment. It loaded the original
PGM images and ran all 120 fits with clustering enabled and 200 updates per fit.
Runtime depends on hardware and numerical-library configuration.

The complete experiment and result plotting finished within one hour. Its local outputs are under
`code/outputs/evaluation_fast/`; `timing.json`, `plotting_timing.json` and
`verification.json` record the measurements and checks.

The command-line runner now defaults to one numerical-library thread in a fresh
Python process, unless explicitly overridden. On a short YaleB benchmark this
was 6.3–7.1 times faster than eight threads without changing the NMF updates.
See [runtime instructions](code/README.md#thread-settings-and-runtime) for the
benchmark scope, a fresh-output command and full-run timing verification.

### Runtime history

| Version | Complete-run time | Evidence and change |
| --- | --- | --- |
| Initial development version | More than 4 hours, approximate | Team-reported estimate; no original full-run timer retained. |
| After earlier pipeline improvements | About 2 hours 29 minutes | Saved workbook's wall-time estimate; CSV records exactly 2 hours 21 minutes 19.4 seconds of fitting. Earlier commits reused matrix reconstructions and removed a duplicate noise-generation call. |
| Current version, one numerical-library thread | **30 minutes 36 seconds** | Fresh measured run of all 120 fits, including loading, clustering, checkpoints and result plotting. |

The earlier code changes are verified in Git, but the initial timing was not a
controlled benchmark that isolates each change. The current run uses the same
scientific settings as the saved intermediate run. Comparing recorded fitting
times gives a **4.73x speedup**; comparing complete-run times with the workbook's
approximate 2 hours 29 minutes gives about **4.9x**.

All RRE table means and standard deviations stayed unchanged at four decimal
places. One YaleB condition's optional clustering summary changed slightly.
All four RRE plots and all 12 reconstruction panels have identical pixels.
The original report figures and summaries in `code/final_results/` are retained;
use the [page-by-page report corrections](code/README.md#exact-report-corrections)
when updating the report's runtime claims and clustering table.

## Report formatting

The report uses a **10 pt body** and an approximately **14 pt title**.
The Ed discussion clarification ("Font size", #74) confirms that either the
template's **10 pt body font or a 12 pt body font is acceptable**.
The final compiled PDF must stay within **20 pages**, including references and
running instructions. Recheck the page count after edits. The group reflection
is in Section 4.9.4.

## Running the code

Follow [code/README.md](code/README.md), running its experiment commands from
`code/`. The report's full command includes both `--no-early-stopping` and
`--clustering-metrics`; omitting them changes the experiment.

Raw per-fit results, metadata, sample indices and reconstruction snapshots are
written under `code/outputs/evaluation/`, which is ignored by Git. The retained
files in `code/final_results/` are summaries and figures, not a resumable
checkpoint. Preserve the original raw results locally when reproducing plots.

## Submission checklist

- Include the final LaTeX-generated report PDF, with group ID, every member's
  name/student ID/unikey, accurate contributions, and running instructions in
  its appendix.
- Preserve the report's 10 pt body and 14 pt title. Keep the final PDF within
  the 20-page maximum after updating its results.
- Put `report.pdf`, `code/`, `requirements.txt`, `pytest.ini` and this README
  at the archive root. Include `code/README.md`, the algorithms and tests.
- Keep `code/data/` empty in the ZIP. Exclude local datasets, virtual
  environments, `.git/`, caches and `code/outputs/` (which may contain copies of
  the image data). Keep the report figures in `code/final_results/`.
- Name the ZIP using the group ID and all member student IDs, separated by
  underscores, following the assignment's naming convention. One member submits.
- Extract the ZIP into a fresh directory and run the tests; then place the
  supplied datasets in `code/data/` and run the documented smoke experiment.

Adding the PDF to a submission ZIP is a separate step: an archive of this
repository alone does not contain the report.
