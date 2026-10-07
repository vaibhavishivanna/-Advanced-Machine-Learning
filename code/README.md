# Code

This directory contains Group 39's NMF implementations and experiment pipeline.

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

```text
data/
  ORL/
    s1/1.pgm ... 10.pgm
    ...
    s40/
  CroppedYaleB/
    yaleB01/*.pgm
    ...
    yaleB39/*.pgm
```

YaleB subject 14 is absent from the supplied dataset. The loader excludes
`*Ambient.pgm` and non-PGM files. ORL is resized to 37 x 30 pixels and YaleB to
48 x 42 (height x width), then intensities are divided by 255 after corruption.
Use `--orl-root` and `--yaleb-root` with the experiment runner if the datasets
are elsewhere. Actual dataset files are not needed for the automated tests.

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

Run the dataset-free algorithm demonstration from `code/`:

```bash
python demo_algorithms.py
```

Run a small real-data smoke experiment before committing to the full grid:

```bash
python run_experiments.py \
  --datasets ORL \
  --runs 1 \
  --sample-frac 0.1 \
  --max-iter 5 \
  --block-sizes 5 \
  --block-counts 1 \
  --count-sweep-size 5 \
  --output-dir outputs/smoke
```

The smoke run uses a separate output directory so it cannot occupy the final
experiment checkpoint. It tests the workflow, not the report's numerical results.

Reproduce the report's five-trial ORL and YaleB experiment:

```bash
python run_experiments.py \
  --datasets ORL YaleB \
  --runs 5 \
  --sample-frac 0.9 \
  --sampling stratified \
  --max-iter 200 \
  --no-early-stopping \
  --block-sizes 10 15 20 \
  --size-sweep-count 1 \
  --block-counts 1 2 3 \
  --count-sweep-size 10 \
  --smoothing 0.001 \
  --seed 4328 \
  --clustering-metrics \
  --output-dir outputs/evaluation
```

This produces 120 result rows: 2 datasets x 5 trials x 6 unique conditions x
2 algorithms. The 90% samples contain 360 ORL and 2173 YaleB images. Default
ranks are the subject counts, 40 and 38 respectively. Both algorithms receive
identical samples, corrupted matrices and initial factors within a condition.

Non-overlapping placement is enforced by default. An impossible placement raises
a clear error instead of silently overlapping blocks. Add `--allow-overlap` only
for a separately documented experiment. The runner records both the requested
and actual covered fractions. The full command includes optional Accuracy and
NMI to match the report. Both use majority-mapped labels, following the supplied
assignment notebook; Accuracy is purity-style and NMI is calculated after mapping.

The runner's default relative-loss stopping tolerance is `1e-5`. The reported
command disables early stopping, so each fit performs all 200 updates. Enabling
early stopping or changing the iteration budget creates a different experiment;
`--tol`, `--min-iter` and `--patience` apply when early stopping is enabled.
The robust objective uses smoothing 0.001, distinct from the
floating-point denominator floor.

The saved full run took about 2 hours 21 minutes for factorisation alone on the
reported machine. Clustering, loading and plotting add time. Runtime varies
with hardware and BLAS configuration.

The subsequent full run described below completed within one hour, including
result plotting, with one numerical-library thread.

### Thread settings and runtime

When launched as a script in a fresh Python process, `run_experiments.py` now
defaults OpenBLAS, OpenMP, MKL and Accelerate thread environment settings to one
before importing NumPy. Explicit environment settings are respected. Importing
the runner as a module does not change the caller's thread settings.

In the tested environment (NumPy 1.26.4, OpenBLAS 0.3.21), a YaleB benchmark with
2173 images, rank 38 and 30 updates gave:

| Algorithm | 8 threads | 1 thread | Speedup |
| --- | ---: | ---: | ---: |
| Standard NMF | 12.69 s | 1.79 s | 7.1x |
| Smoothed L1-loss NMF | 32.60 s | 5.18 s | 6.3x |

This is a short benchmark, not a full-grid timing guarantee. The final losses
matched to floating-point precision. Thread coordination was expensive for
these matrix shapes; more threads were slower. This change preserves the NMF
updates, five trials, sample sizes, ranks, seeds and 200-update budget.

Changing thread counts can change floating-point reduction order. The complete
120-fit run preserved all RRE summaries to four decimal places, but one severe
YaleB condition produced different optional clustering scores, detailed below.

### What changed in the earlier optimisation

The initial version was reported to take more than four hours. The saved
`final_results.xlsx` workbook records approximately 2 hours 29 minutes for the
later complete run, including clustering, data handling, saving and plotting.
Its raw CSV contains 8479.41 seconds of fitting. The workbook and original
report PDF are stored beside this repository, not inside it.

Git history identifies two earlier efficiency changes:

- Commit `00b4713` (`Improve NMF convergence and robustness tests`) reused the
  current `W @ H` reconstruction in `algorithm/robust_l1_nmf.py`. Previously the
  loop calculated this large product four times per update. It now calculates
  it twice, after updating H and after updating W, and reuses it for the residual,
  weighted denominator and loss where the factors have not changed. The cached
  reconstruction is refreshed after each factor update.
- Commit `175ad11` (`Harden data and experiment pipeline`) made
  `add_occlusion_noise` return its mask with the corrupted data. The runner no
  longer calls the noise function a second time on an empty array merely to
  recover the occlusion footprint.

Dataset loading was already outside the trial loop in the original runner;
the historical changes above removed repeated calculations and a duplicate
function call. The same commits also introduced validation and optional early
stopping, but **early stopping is disabled in both compared final runs**.
The initial four-hour estimate has no retained timer, and these commits include
other changes, so their individual contributions cannot be assigned exact
speedups. The latest change configures numerical-library threading; it changes
neither NMF update equations nor the experiment size.

To explicitly select one thread on macOS/Linux and preserve the original results,
run from `code/` in a terminal:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python run_experiments.py \
  --runs 5 --sample-frac 0.9 --max-iter 200 \
  --no-early-stopping --clustering-metrics \
  --output-dir outputs/evaluation_fast
```

Other scientific settings use the report defaults shown in the full command
above. The datasets must still be present under `data/`. In a notebook or an
existing interpreter, restart Python with these settings before NumPy is imported.

Generate the matching results for this destination with:

```bash
python plot_results.py \
  --results outputs/evaluation_fast/results.csv \
  --output-dir outputs/evaluation_fast
```

The runner saves `timing.json`, measuring its invocation including data loading,
fitting, metrics and checkpoints, but excluding the separate plotting command.
It also records package versions and configured thread environment variables.
A resumed invocation records only the work performed in that invocation, with
`new_result_rows` distinguishing newly computed fits from saved ones. Use a fresh
output directory and all 120 fits for a complete runtime measurement.

### Verified full run and report updates

On 8 October 2026, the complete command above was run from scratch on the
original ORL and YaleB image files. All **120 fits** completed in **1828.50 seconds
(30 minutes 28.5 seconds)** including Python startup, data loading, clustering
and checkpoints. Plotting and table generation took another **7.60 seconds**,
giving **30 minutes 36 seconds total**. The runner's internal timer excludes
interpreter/import startup and records 1826.78 seconds.

This used Python 3.12.2, NumPy 1.26.4 and OpenBLAS 0.3.21 with one thread, on
macOS 13.7.8. Five 90% trials, all six conditions, ranks 40/38, seed 4328,
smoothing 0.001 and all 200 updates were retained. Fitting alone took 1792.49
seconds versus 8479.41 seconds in the original saved run (about 4.73x faster).

Updated factorisation times across 30 fits per dataset/algorithm:

| Dataset | Standard NMF, mean ± sample SD | Smoothed L1, mean ± sample SD | Robust/standard |
| --- | ---: | ---: | ---: |
| ORL | 1.28 ± 0.41 s | 3.44 ± 0.81 s | 2.68x |
| YaleB | 14.65 ± 1.23 s | 40.38 ± 1.82 s | 2.76x |

Verification matched all sample indices, seeds, ranks and iteration counts to
the original run. Every RRE summary mean and sample SD is unchanged at the
report's four-decimal precision; the largest per-fit RRE difference was
8.11e-8. One fit's optional clustering assignments changed, affecting the YaleB
`b=20, k=1` robust-method summary: Accuracy is now **0.084 ± 0.004** and NMI
**0.079 ± 0.009**. Other clustering summaries are unchanged at the report's
three-decimal precision.

The fresh results, summaries and figures are under `outputs/evaluation_fast/`.
`timing.json`, `plotting_timing.json` and `verification.json` retain the local
verification evidence. These are generated outputs and are not committed.
The original `outputs/evaluation/` and retained `final_results/` files have not
been replaced.

When updating the report to describe the optimised run, replace its runtime
table and related totals/ratios in the abstract, discussion and conclusion;
record the one-thread configuration and updated operating-system version;
and update the one affected clustering row. Use the fresh five-trial summaries,
not the short 30-update benchmark. Choose another unused output directory for
an additional full timing check; do not resume completed results to measure
the full computation.

### Exact report corrections

These locations refer to the current 20-page
`Assignment_1_COMP4328_5328_8328.pdf`; pagination may move after editing.
Its main LaTeX source is not present, so the PDF has not been edited. The report
uses a 10 pt body and approximately 14 pt title; retain those sizes.
The Ed discussion clarification ("Font size", #74) accepts either the template's
10 pt body or a 12 pt body. Check that the final compiled PDF stays within
20 pages after editing; the original PDF's page count does not verify a revision.

| Location | Required replacement |
| --- | --- |
| Abstract, page 1 | Replace the old `1.69–2.22` runtime range with **2.68–2.76 times the factorisation time**. |
| Section 4.2.3, page 13 | Change macOS **13.5.2** to **13.7.8** for the new run. Add **OpenBLAS 0.3.21, one numerical-library thread**. Distinguish per-fit factorisation timings from the separately measured complete-run wall time. |
| Section 4.7, page 15 | For YaleB, change “higher Accuracy and NMI in nearly every condition” to **“higher mean Accuracy and NMI in every tested condition”**. Differences remain small under strong corruption. |
| Section 4.8, pages 15–16 | Replace its timing paragraph with the text below. Retain the statement that all 120 fits used 200 updates. |
| Table 4, page 16, YaleB `b=20, k=1` row | Robust Accuracy: **0.085 ± 0.003 → 0.084 ± 0.004**. Robust NMI: **0.081 ± 0.006 → 0.079 ± 0.009**. The Standard NMF entries and every other row are unchanged at displayed precision. |
| Table 5, page 16 | Replace both rows with the updated table above: ORL **1.28 ± 0.41**, **3.44 ± 0.81**, **2.68x**; YaleB **14.65 ± 1.23**, **40.38 ± 1.82**, **2.76x**. Times are seconds, with sample SD across 30 fits per dataset/algorithm. |
| Section 4.9.1, page 16 | Replace `1.69–2.22` with **2.68–2.76** and use “times the factorisation time”. |
| Section 4.9.3, page 17 | Replace the final runtime limitation sentence with: **“Runtime was measured on one machine. Table 5 covers factorisation only; the complete-run wall time additionally includes loading, corruption, clustering, checkpointing and result plotting. These measurements do not establish performance on other hardware.”** |
| Section 5.1, page 18 | Replace `1.69–2.22 times more factorisation time` with **2.68–2.76 times the factorisation time**. |
| Section 5.2, page 18 | Refine the parallel-execution suggestion: **“Future work could test early stopping, accelerated updates and parallel execution of independent fits with controlled BLAS thread counts. Increasing BLAS threads alone slowed the present workload.”** |
| Appendix A.2, page 20 | Use `outputs/noise_size` for the first noise-demo command and `outputs/noise_count` for the second, as below. Their current shared destination overwrites the first set of illustrations. |
| Appendix A.3, page 20 | Prefix the full experiment command with the four one-thread environment settings shown above. Keep every scientific argument. Use a fresh output directory and the same destination for plotting; existing completed directories are not fresh timing runs. |
| Appendix A.4, page 20 | Add: **“On completion, the runner writes timing.json with invocation duration, package/platform information, configured thread counts and the number of newly computed fits. Plotting runs separately.”** |

Replacement timing paragraph for Section 4.8:

> Table 5 reports the mean factorisation time per fit across the 30 fits for each
> dataset–algorithm pair. The robust method required 2.68 times the factorisation
> time of Standard NMF on ORL and 2.76 times on YaleB. Across all 120 fits, Standard
> NMF required 477.9 seconds and the robust method required 1314.6 seconds, giving
> 1792.5 seconds of factorisation time in total (29 minutes 52.5 seconds). These
> per-fit timings exclude data loading, corruption, clustering, checkpointing and
> plotting. Separately, the full experiment process took 1828.50 seconds and result
> plotting took 7.60 seconds, giving 1836.10 seconds overall (30 minutes 36.1
> seconds). The run used one numerical-library thread and retained all five
> trials, six conditions and 200 updates per fit. Totals use unrounded values.

If describing the development history in Section 4.8 or the existing personal
reflection (Section 4.9.4, pages 17–18), distinguish the team's initial **over
four-hour estimate**, the workbook's **approximately 2 hours 29 minutes**, and
the new **measured 30 minutes 36 seconds**. Explain the verified reconstruction
reuse and duplicate-noise-call removal before the latest threading change.
The old and new saved factorisation totals support a 4.73x speedup. The approximate
complete-run comparison supports about 4.9x, not an exact controlled measurement
of each historical change. Attribute personal contributions only to the person
who actually performed them.

**Numbers and figures that do not need changing:** Table 3's RRE means and SDs
are unchanged at four decimals, as are the stated RRE percentage improvements.
All four regenerated RRE sweep plots and all 12 reconstruction panels are
pixel-for-pixel identical to `final_results/figures/`. These saved files also
match the images embedded in the PDF, so Figures 2–4 need no replacement.
For Figure 1, regenerating all four noise illustrations produced slightly
different plot spacing/image bounds. Sampling the centres of all original
37 x 30 or 48 x 42 image pixels in each panel confirmed identical face and
occlusion content. This is a rendering-layout difference; the existing Figure 1
remains valid. The largest individual RRE change was about 8.10e-8, while one fit's
clustering output changed enough to alter the Table 4 entries above; the new run
is not bit-for-bit identical numerically. Local figure comparisons are recorded
in `outputs/evaluation_fast/figure_verification.json`,
`pdf_figure_verification.json` and `noise_figure_verification.json`.

An additional existing wording correction is in Section 3.7.1, step 6, page 11:
the runner saves **initial and final training losses**, rather than full loss
histories. Full histories are returned by the algorithms but are not written
to the experiment CSV. The reflection already exists; check its named author
and contribution descriptions against Section 6 before submission.

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

Resume requires the same argument values, including dataset and output paths.
The original local checkpoint used `../data/ORL` and `../data/CroppedYaleB`;
the submission layout above uses `data/ORL` and `data/CroppedYaleB`. Use a fresh
output directory when reproducing with a different layout.

## Noise illustrations

Run these from `code/`. Separate destinations retain both sweeps, because the
demo uses the same figure filenames for each invocation:

```bash
python -m experiments.run_demo --block_sizes 10 15 20 \
  --num_blocks 1 --out_dir outputs/noise_size

python -m experiments.run_demo --block_sizes 10 \
  --num_blocks 1 2 3 --out_dir outputs/noise_count
```

Each destination contains `figures/ORL_occlusion_comparison.png` and
`figures/YaleB_occlusion_comparison.png`. The demo also saves image arrays under
`arrays/`; exclude these from the submission. Demo path flags use underscores
(`--orl_root`, `--yaleb_root`), while the experiment runner uses hyphens.

## Outputs and retained results

- `results.csv`: per-fit RRE, optional metrics, runtime, seeds and endpoint losses.
- `metadata.json`: experiment arguments, conditions and dataset dimensions.
- `timing.json`: elapsed time and environment for the latest completed invocation.
- `sample_indices.json`: exact sampled image columns for every dataset/trial.
- `reconstructions/*.npz`: first-trial image snapshots for each condition.
- `summary.csv` and `rre_summary_table.tex`: means and sample standard deviations
  (`ddof=1`), generated by `plot_results.py`.
- `figures/`: RRE plots and reconstruction panels. Error bars are one sample
  standard deviation, not confidence intervals.

Full loss histories are returned by the algorithms but are not saved by the
experiment runner. The CSV retains only their first and last values.

`final_results/` retains the report's summaries and figures in Git. Raw
checkpoints under `outputs/` are local and ignored. The generated LaTeX table
is a fragment requiring `booktabs` in the containing report, not a standalone
document. The full report source is not included in this repository.

When regenerating all reconstruction panels, keep the matching `reconstructions/`
snapshot directory inside the selected `--output-dir`; the plotter looks there
for the arrays. Supplying the CSV alone regenerates the tables and RRE plots.
