# From PDE Residuals to Error Rankings in Elliptic Neural Operators

Start with **`paper/draft_ml4ps.pdf`** in the anonymous review archive. The
artifact contains the implementation, retained per-field evidence and checks
behind its two tables and two figures. The alternate-named manuscript files are
equivalent copies for the document checker.

The identity `e = A^-1 r` is classical. The paper asks what information about a
learned predictor's spatial error becomes available before completing that
solve, comparing residual consistency, empirical ranking, certified membership
and actual correction against computational cost.

## Quick reproduction: review archive only

Extract `ml4ps2026_review.zip` and work inside `ml4ps2026-review/`.
Use Python 3.9 (recorded: 3.9.6) with the pinned dependencies:

```sh
python3.9 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python analysis/reproduce_paper_assets.py
```

No GPU, TeX installation or companion archive is needed for this command.
It runs the existing builders in an isolated copy and writes to a **new**
`tmp/reproduced_paper/` directory. Retained data, figures and manuscript PDFs
remain unchanged. Use `--out tmp/another_run` for subsequent runs.

| Paper item | Output below `tmp/reproduced_paper/` |
|---|---|
| Table 1: binary-field localization | `paper/generated/localization_table.tex` |
| Table 2: predictor/coefficient controls | `paper/generated/robustness_table.tex` |
| Figure 1: workflow, maps and membership | `paper/figures/submission_maps_alternate.{pdf,svg,png}` |
| Figure 2: budget and certification curves | `paper/figures/submission_budget_bounds.{pdf,svg,png}` |

Table 1 compares scores on binary coefficient fields. Table 2 checks different
predictors and a separately trained smooth-family model; its smooth row uses a
different field population, so that comparison is descriptive rather than paired.

`REPRODUCTION.json` records input/output hashes, environment, runtime and figure
byte comparisons; `logs/` contains every builder/check transcript. Numerical
LaTeX and representative-field decisions must match the retained outputs
exactly. Figure bytes can vary with renderer/font versions. A test with the
recorded environment took about **14 seconds on an Apple M4 CPU**, using one
CPU thread, with all six figure files byte-identical. Installation is excluded.

This fast path reads retained summaries and recomputes the representative
field's correction and bound. It does **not** independently recompute every
bootstrap interval, replay all predictions or train networks.

## Full frozen-model checks

Extract `ml4ps2026_frozen_inputs.zip` into the same parent directory
to merge its caches and checkpoints into `ml4ps2026-review/`. Use a disposable
extracted copy: verification refreshes audit receipts and derived diagnostics.
Document checks also require Poppler (`pdfinfo`, `pdftotext`).

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest -q -W error::RuntimeWarning
python analysis/audit_submission_artifacts.py
python analysis/verify_smooth_artifacts.py --producer-log logs/smooth_robustness.log
python analysis/build_robustness_assets.py
python analysis/audit_alternate_manuscript.py
```

The core audit checks the binary/U-Net evidence, recomputes statistics and
executes its declared numerical replays. The smooth verifier checks every
retained training/test draw and checkpoint prediction, summaries, identities,
energies and rank decisions. Both share numerical primitives with the experiment;
the verifier documents its independently implemented and shared components.
Neither command replays training optimization. Recorded release runtimes were
about **12 seconds for the core audit and 14 seconds for the smooth verifier**
on the same CPU with one thread; these are environment-specific observations.
The robustness builder refreshes its manifest after the smooth receipt changes.

Without the companion archive, `python analysis/audit_submission_artifacts.py
--artifact-only` checks available records and explicitly reports missing replay
as unverified. It is not a full verification pass. Retraining and fresh-directory
replay require new output directories.

## Retraining examples

For a complete three-member binary FNO example at grid 32, contrast 100, train
and then evaluate fresh fields using **the same new checkpoint directory**:

```sh
python experiments/exp_real_fno_multi_kappa.py \
  --N 32 --kappas 100 --seeds 0 1 2 --epochs 80 --ntrain 700 --ntest 200 \
  --checkpoint-dir results/retrained/binary32_checkpoints \
  --out results/retrained/binary32_training
python experiments/exp_submission_validation.py \
  --N 32 --kappas 100 --samples 200 --test-seed 20260912 --skip-fields 0 \
  --checkpoint-dir results/retrained/binary32_checkpoints \
  --out results/retrained/binary32_fresh
python experiments/exp_smooth_robustness.py \
  --out results/retrained/smooth_robustness
```

The binary training driver also evaluates its original fields; the second
command evaluates fresh fields from its new checkpoints. The smooth command
trains and evaluates the complete fixed protocol. Its recorded run took about
**7.8 minutes**, including **7.5 minutes of training** summed over the three
members' epoch histories (`results/smooth_robustness/summary.json`). This is a
CPU observation, not a runtime guarantee; no comparable complete historical
binary-training time is claimed. Remaining grids, contrasts, bounds, timing and
training controls are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Evidence and seeds

- Binary FNO: checkpoints in `results/checkpoints/`, `checkpoints_64/` and
  `checkpoints_longtrain/`; per-field `metrics_k*.csv`, `norms_k*.csv` and
  summaries in `results/submission_*/` and `submission_fresh_*/`.
- U-Net: checkpoints, `fields.csv`, protocol and summary in
  `results/unet_robustness/`. All ensembles use model seeds **0, 1, 2**.
- Binary training uses data seed **0**; original evaluation skips its first
  700 draws, while fresh evaluation uses **20260912**. Geometries are shared
  across binary contrasts/grids, so those settings are not independent draws.
- Smooth FNO: `results/smooth_robustness/` holds `training.npz`,
  `predictions_k100.npz`, `fno_seed*.pt`, `metrics.csv`, `fields.csv`,
  `bounds.csv`, `members.csv`, training histories and exact source snapshots.
  Train/test seeds are **20260913/20260914**; bootstrap seed is **20260915**.
  The design is in `experiments/SMOOTH_ROBUSTNESS_PROTOCOL.md`.
- Bound/adaptive ledgers: `results/rank_bounds_*/`, `flux_bounds_all/`,
  `flux_bounds_unet/`, `adaptive_rank_stopping/` and `adaptive_rank_flux/`.
  Timing repetitions and source snapshots: `results/submission_timing_flux/`.

## Numerical caveat and scope

The original smooth producer emitted **600 NumPy scalar-matmul floating-status
warnings** (divide-by-zero, overflow and invalid value). They remain in
`logs/smooth_robustness.log`. The independent verifier uses elementwise/face
sums for the energy check: all test-field energies were finite, with maximum
relative disagreement about `1.03e-15`. Its receipt records both these checks
and the original warnings; the producer evidence was not rewritten.

Intervals condition on the trained ensembles. Binary/smooth comparisons are
descriptive, and the smooth control changes both coefficient distribution and
predictor. Results cover one forcing and modest discrete grids. Bounds are
exact-arithmetic implications checked numerically in float64, not outward-rounded
or continuum certificates. Iteration counts do not imply equal runtime.
The retained JSON verification receipts and source manifests document the
numerical checks and provenance used by the reproduction commands above.
