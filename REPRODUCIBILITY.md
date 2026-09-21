# Reproducing the submission

The primary paper uses frozen neural predictions and the same discrete matrix for
truth, residuals, corrections and validation. Numerical values in the manuscript
are generated from retained per-field records, not copied into LaTeX by hand.
All intervals are conditional on the trained networks and the stated field
population. Random geometry draws are shared across contrasts/resolutions;
those settings must not be pooled as independent geometry replicates.

## Three levels of verification

1. **Code and artifact checks:** run the test suite, recompute the reported
   statistics from CSVs, check generated-source and figure hashes, and inspect
   the compiled paper. This does not retrain a model.
2. **Frozen-model replay:** regenerate held-out fields, evaluate saved checkpoints,
   rerun all scores and bounds in a new output directory, and compare per-field
   metrics. This checks the numerical pipeline with the actual learned errors.
3. **Retraining:** train new ensembles with the recorded configuration. Outcomes
   can vary with hardware and numerical libraries, so this tests scientific
   replication rather than byte identity with a historical checkpoint.

The recorded environment is Python3.9.6, NumPy2.0.2, SciPy1.13.1,
PyTorch2.8.0, PyAMG5.3.0 and Matplotlib3.9.4. Exact Python dependencies are in
`requirements-lock.txt`. The final timing replication used an Apple M4 CPU.
PDF builds require a TeX distribution with `latexmk` and PDFLaTeX; PDF checks
require Poppler's `pdfinfo` and `pdftotext`.

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q -W error::RuntimeWarning
.venv/bin/python analysis/build_submission_assets.py --check
.venv/bin/python analysis/audit_submission_citations.py
.venv/bin/python analysis/audit_submission_artifacts.py
.venv/bin/python analysis/audit_submission_splits.py
```

`VECLIB_MAXIMUM_THREADS` controls Apple Accelerate; limiting OpenBLAS alone did
not establish a one-thread benchmark. Timing artifacts retain backend metadata,
source snapshots, randomized method order, raw repetitions, process CPU/wall
ratio and bootstrap intervals. Absolute runtimes depend on the machine.

## Primary artifacts

| Result | Retained directory |
|---|---|
| Original frozen FNO comparisons | `results/submission_32`, `results/submission_64` |
| Fresh frozen FNO comparisons | `results/submission_fresh_32`, `results/submission_fresh_64` |
| Original and fresh Poisson bounds | `results/rank_bounds_32`, `rank_bounds_64`, `rank_bounds_fresh_32`, `rank_bounds_fresh_64` |
| Paired equilibrated-flux bounds | `results/flux_bounds_all` |
| Reference-free adaptive stopping | `results/adaptive_rank_stopping`, `results/adaptive_rank_flux` |
| Final runtime replication | `results/submission_timing_flux` |
| Longer FNO training check | `results/submission_longtrain_fresh_32` |
| Additional U-Net check | `results/unet_robustness` |
| Calibrated/OOD supporting analyses | See `SUPPLEMENT_METHODS.md` |

Within localization directories, `predictions_k*.npz` retains coefficient,
forcing, discrete truth, all three member predictions, and a provenance
signature. `metrics_k*.csv` has one row per field/method, `norms_k*.csv` has
independent energy/flux targets, and `summary.json` records bootstrap statistics.

Original evaluation uses generator seed0 after skipping700 training draws;
fresh evaluation uses seed20260912 without skipping. Training normalization is
reconstructed and checked against checkpoints. Historical80-epoch checkpoints
omit an embedded data seed: seed0 provenance is supported by this reconstruction
and saved protocol, not a complete historical execution trace.

## Frozen-model replay

Use a **new** output directory. Existing caches are immutable evidence; the
strict cache signature intentionally refuses raw-source mismatches. A subsequent
docstring correction to the coefficient generator changed its file hash but no
executable AST nodes. The previous exact file and machine-checked equivalence
record are retained under `reproducibility/`. This changes neither saved arrays
nor scientific results. Replay creates a signature for the current source.

```sh
.venv/bin/python experiments/exp_submission_validation.py \
  --N 32 --kappas 10 100 500 --samples 200 \
  --test-seed 20260912 --skip-fields 0 \
  --out results/replay/submission_fresh_32
.venv/bin/python experiments/exp_submission_validation.py \
  --N 64 --kappas 10 100 500 --samples 200 \
  --test-seed 20260912 --skip-fields 0 \
  --out results/replay/submission_fresh_64
```

For the original evaluation, use `--test-seed 0 --skip-fields 700` and different
output directories. To replay the longer-trained ensemble, add
`--checkpoint-dir results/checkpoints_longtrain --N 32 --kappas 100`.

```sh
.venv/bin/python experiments/exp_rank_certificates.py \
  --cache-dir results/replay/submission_fresh_32 \
  --out results/replay/rank_bounds_fresh_32
.venv/bin/python experiments/exp_rank_certificates.py \
  --cache-dir results/replay/submission_fresh_64 \
  --out results/replay/rank_bounds_fresh_64
.venv/bin/python experiments/exp_flux_bounds.py \
  --cache-dirs results/replay/submission_fresh_32 results/replay/submission_fresh_64 \
  --budgets 20 40 --out results/replay/flux_bounds
.venv/bin/python analysis/check_adaptive_rank_stopping.py \
  --cache-dirs results/replay/submission_fresh_32 results/replay/submission_fresh_64 \
  --bound-kind poisson --out results/replay/adaptive_poisson
.venv/bin/python analysis/check_adaptive_rank_stopping.py \
  --cache-dirs results/replay/submission_fresh_32 results/replay/submission_fresh_64 \
  --bound-kind flux --out results/replay/adaptive_flux
.venv/bin/python analysis/benchmark_submission_solvers.py \
  --cache-dirs results/replay/submission_fresh_32 results/replay/submission_fresh_64 \
  --samples 30 --repeats 3 --out results/replay/timing
```

The full flux audit also includes original caches and
`--ood-bundle results/ood_features_bundle.npz`. The new single-thread timing
replication supersedes the earlier timing directories; the earlier runs are
retained for comparison. Do not time solvers during training or other heavy
computation. Each timing method includes residual formation; fresh AMG includes
hierarchy construction and the first coarse solve. Reused-setup measurements
are a separate same-matrix scenario.

## Training configurations

The baseline FNO has three Fourier layers,12 modes,width24 and three input
channels (normalized log coefficient and x/y coordinates). Each of the3 seeds
uses700 training fields,80epochs,MSE,Adam lr0.001,cosine decay and batch32.
Grid-specific ensembles are trained independently. The longer check changes
only the epoch budget to240 at N32/contrast100, before fresh evaluation.

```sh
.venv/bin/python experiments/exp_real_fno_multi_kappa.py \
  --N 32 --kappas 10 100 500 --seeds 0 1 2 --epochs 80 \
  --checkpoint-dir results/retrained/checkpoints_32 \
  --out results/retrained/fno_32
.venv/bin/python experiments/exp_real_fno_multi_kappa.py \
  --N 64 --kappas 10 100 500 --seeds 0 1 2 --epochs 80 \
  --checkpoint-dir results/retrained/checkpoints_64 \
  --out results/retrained/fno_64
.venv/bin/python experiments/exp_unet_robustness.py \
  --width 16 --epochs 200 --seeds 0 1 2 \
  --out results/retrained/unet_robustness
```

The U-Net check uses a compact two-level encoder/decoder with channels16/32/64,
GroupNorm/GELU and transposed convolutions, the same train/fresh field populations
and training-only normalization. All three members are frozen before loading
fresh reference solutions. Training-only timing pilots informed the compute
budget; no fresh result selected a model, epoch or width.

## Interpretation and verification limits

### Additional smooth-coefficient experiment

`experiments/SMOOTH_ROBUSTNESS_PROTOCOL.md` fixes the design before test
evaluation. The alternate manuscript adds a separately trained smooth-family
FNO ensemble at N32 and contrast100. Gaussian-filtered white noise (filter
standard deviation0.12 domain lengths, wrap mode) is normalized by its sampled
extrema and exponentiated to coefficients in[1,100]. This is a bounded smooth
random family, not an unchanged Gaussian or lognormal law. The fixed forcing,
discretization, architecture, 700 training fields, 80 epochs and three model
seeds match the baseline. Data seeds20260913/20260914 separate training and
200-field evaluation. No test result selects epochs, models or generator knobs.

```sh
.venv/bin/python experiments/exp_smooth_robustness.py \
  --out results/retrained/smooth_robustness
.venv/bin/python analysis/verify_smooth_artifacts.py \
  --directory results/smooth_robustness --producer-log logs/smooth_robustness.log
.venv/bin/python analysis/build_robustness_assets.py --check
```

The first command retrains into a new directory; the second independently
checks the retained submission run. To verify a retraining, substitute its new
directory and captured producer log. `results/smooth_robustness/` retains training/test arrays, checkpoints,
source snapshots, per-field metrics and bound decisions. The bootstrap unit is
the coefficient field and intervals condition on each trained ensemble.
Binary/smooth comparisons are descriptive because the field populations and
trained models differ. This experiment has no new runtime speedup claim.

### Scope of the discrete diagnostics

The identity is `e=A^-1 r` for the same discrete TPFA system. The residual is
flux divergence, not flux magnitude. Cell-energy contributions are nonnegative
and sum to `e^T A e`. Poisson and flux bounds are classical exact-arithmetic
implications for this discrete system; float64 checks are not outward-rounded
certification. The bound-check tolerance is `1e-10*max(|e|)+1e-13` and is not
added to the bounds used to create rank flags. Tiny raw bound exceedances near
solver roundoff are retained in the audit rather than erased.

The adaptive stopping rule uses A, r and coefficient data only. Ground truth is
used afterward to validate the flags. The code handles unresolved ties and
unmet targets explicitly. Fewer iterations with the flux bound do not imply
less runtime: its extra face sums were slower in the tested implementation.

The manuscript's generated numerical files and plots have an input/output hash
manifest. `--check` detects edits to source results, plotting code and figure
bytes. The independent artifact audit recomputes per-field statistics and
checks the official template and PDF. The legacy numerical ledger audit does
not parse the current paper and is not evidence that its claims match artifacts.

## Reference verification

`analysis/audit_submission_citations.py` mechanically checks every cited title,
ordered author list, chosen year, URL and listed identifier/publication field
against `paper/citations_primary_snapshot.json`. The snapshot records the primary
sources checked and explains arXiv versus publication-year choices. This offline
check detects drift from verified metadata; it does not replace reading a source
for relevance or claim to retrieve the source again on every run.

## Recovered historical source versions

A final archive review identified three recorded source hashes whose exact
versions had not initially been retained. Narrow, documented edits were reversed
in separate copies, and the recovered files match the recorded SHA256 hashes
exactly. The files and transparent reconstruction receipt are retained under
`reproducibility/recovered_source_snapshots/`. This recovers exact source bytes;
it does not invent an execution trace or relabel original experiment metadata.
The changes concerned the adaptive wrapper's later flux option, a zero-RHS AMG
validation guard, and an allocation-only `copy=False` change in CG.

A separate full current-code replay of the original Poisson adaptive analysis is
recorded in `results/adaptive_current_replay_comparison.json`, with its exact
source snapshots under `reproducibility/adaptive_current_replay/`. Replay outputs
are compared with retained per-field records and histories; the report states
any differences rather than silently replacing the earlier evidence.
