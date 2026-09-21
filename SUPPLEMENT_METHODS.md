# Supplement: OOD audits and discrete flux diagnostics

This document records supporting analyses for a frozen, three-member Fourier
neural operator (FNO) ensemble. The principal claim concerns localization of
error relative to the **same discrete Darcy system** used for reference solutions.
These experiments do not establish continuum error bounds, distribution-free OOD
coverage, or an end-to-end solver speedup. The bounds below are classical energy
and equilibrated-flux majorants; no new theorem is claimed.

## Artifacts and data protocol

All paths below are relative to the repository root. Historical result files
remain unchanged. New analyses live in separate directories.

| Analysis | Authoritative numerical artifact | Reproduction program |
|---|---|---|
| Corrected calibration and deterministic full AMG | `results/ood_audit_fullamg_20260912/audit.json` | `analysis/audit_ood_protocol.py` |
| Source-trained risk ranking with fair controls | `results/ood_audit_fullamg_20260912/id_risk_controls.json` | `analysis/check_id_risk_controls.py` |
| Fixed-budget OOD rank enclosures | `results/rank_bounds_ood/summary.json` | `experiments/exp_rank_bounds_ood.py` |
| Paired flux-majorant comparison | `results/flux_bounds_all/summary.json` | `experiments/exp_flux_bounds.py` |
| Training/test overlap and checkpoint metadata | `results/submission_split_audit.json` | `analysis/audit_submission_splits.py` |
| Fresh-field U-Net robustness | `results/unet_robustness/summary.json` | `experiments/exp_unet_robustness.py` |
| U-Net checkpoint/prediction verification | `results/unet_robustness/artifact_verification.json` | `analysis/verify_unet_artifacts.py` |
| Exploratory frozen U-Net flux check | `results/flux_bounds_unet/summary.json` | `experiments/exp_flux_bounds.py` |

Per-field CSVs accompany the rank and flux summaries; `summary_table.csv` in each
directory provides a compact view. JSON files retain full precision, input and
implementation hashes, numerical versions, and confidence intervals.

The original FNOs use 700 Family-C training fields, 80 epochs, modes 12, width 24,
three layers, and model seeds 0/1/2. Each contrast and grid has its own ensemble.
The N=64 experiment is a separately trained resolution replication. Original
evaluation fields are generator seed 0, draws 700–899; fresh evaluation fields
are seed 20260912, draws 0–199. The original test was used in exploratory work.
Fresh fields use frozen predictors and the fixed comparison protocol in
`experiments/SUBMISSION_VALIDATION_PROTOCOL.md`. Random geometries are shared
across contrasts and resolutions; conditions must not be pooled as independent
geometry replicates.

The split audit reconstructs all original and fresh coefficient arrays exactly,
checks zero exact coefficient overlap with the first 700 seed-0 training draws,
and checks 21 checkpoint payloads against their sidecars. Training-only input
normalization agrees within 1e-12. The current training code computes output
normalization from training targets and uses no test labels in optimization or
epoch selection. Old 80-epoch checkpoints omit an embedded data seed; seed-0
provenance is supported by this reconstruction, normalization, and saved protocol,
rather than a complete historical training log. Exact disjointness does not
prove absence of unrecorded historical model selection.

The saved N=32, nominal-contrast-100 OOD bundle contains 200 fields each for
calibration, ID, geometry shift, roughness shift, and checkerboard. All sets have
zero exact coefficient overlap with reconstructed FNO training fields and with
one another. ID, geometry, and roughness test sets each have 200 unique fields;
the checkerboard test has only **two** unique coefficient fields, the two phases
of block size 4. Its 200 draws are a finite-mixture diagnostic, not 200 distinct
geometries. Roughness fields have actual minima 0.010299–0.160963 and contrasts
77.36–4844.01; using nominal contrast to infer their coefficient minimum would
invalidate the bound.

## Calibration and source-trained field risk

Pooled grid cells are dependent. Pooled-cell conformal curves are descriptive
coverage diagnostics. Unscaled field-maximum split conformal has its usual
guarantee only for exchangeable calibration and test fields with a fixed
predictor. No such OOD guarantee is asserted here.

Legacy weighted conformal uses estimated, clipped classifier ratios and a fixed
target mass of 1. Exact weighted conformal instead requires each target point's
own density ratio on the same scale as calibration weights, plus covariate-shift
and support assumptions. Using coefficient summaries also needs conditional
invariance at the summary level. The geometry shift changes inclusion counts
from 2–5 to 4–6 and circle radii from 0.055–0.14 to 0.07–0.15: matching coefficient
extrema does not prove target support containment. Re-evaluation with individual
estimated target masses makes every roughness/checkerboard interval infinite;
their resulting 100% coverage is vacuous. See the classical weighted-conformal
construction in Tibshirani et al. (2019), listed in the paper bibliography.

Corrections in the new audit preserve infinite score mass, reject corrupt
weights/scores, normalize source and target ratios on one source scale, freeze
calibration scale floors on target batches, assign constant-width AUROC 0.5,
and bootstrap the reported pooled masked-coverage ratio by whole fields.
These repairs do not retroactively turn the legacy protocol into a guarantee.

Historical experiment 10 fits 120 labeled fields **within each target shift**
and evaluates 80; it is supervised whole-field ranking. The separate new risk
experiment fits once on 120 ID fields (60/40 split seed 202606070) and evaluates
80 held-out ID plus all 200 fields of each shift. The high-error threshold is the
ID training-error 85th percentile, 0.18126234244189426, frozen across shifts.
No target error label affects fitting, scores, or threshold; regression tests
explicitly perturb target labels to check this invariance.

Both fair-control models use gradient boosting: 120 trees, depth 2, learning rate
0.05, minimum leaf size 6, and the split seed. The cheap model has 24 features:
six ensemble-spread summaries, six prediction-magnitude summaries, and twelve
coefficient/geometry features. Augmentation adds six AMG-correction summaries.
Features receive signed log1p after clipping to ±1e6. AMG fields were recomputed
for all 1,000 bundle fields with deterministic full PyAMG: four hierarchy levels,
four coarsest unknowns, actual smoothed aggregation, no fallback. Seeds are
20260912 + 1000 × dataset index + field index. The maximum relative same-A
identity discrepancy is 3.83e-14; per-field backend and cost records are saved.

The metric is area under uncaught-high-error fraction versus selected whole-field
fraction, lower being better. CIs are 2,000 paired held-out-field bootstraps,
seed 2026091201 plus dataset offset, conditional on the fixed models, split, and
threshold. Exact risk ties are averaged over uniform within-tie orderings.

| Evaluation | High-error fields | Cheap AUC | Cheap + AMG | Paired reduction, 95% CI |
|---|---:|---:|---:|---:|
| ID | 8/80 | 0.01977 | 0.00625 | 0.01352 [0.00156, 0.02845] |
| Geometry shift | 67/200 | 0.11425 | 0.07165 | 0.04260 [0.02602, 0.06059] |
| Roughness shift | 129/200 | 0.31225 | 0.32250 | −0.01025 [−0.02500, 0.00515] |
| Checkerboard | 200/200 | 0.50000 | 0.50000 | 0 [0, 0] |

The geometry gain at 20% whole-field selection is 0.050 [0.015, 0.090] in
uncaught-high-error fraction. Roughness improvement is unsupported, and all-high
checkerboard labels make ranking uninformative. No cell repair or actual solver
handoff is executed in this experiment.

## Classical pointwise bounds and strict top-set decisions

Let A be the symmetric positive definite, cell-centered TPFA Darcy matrix with
harmonic interior coefficients and homogeneous Dirichlet walls. Set B=A(1),
e=u_hat−u, r=A u_hat−f, and d=r−Az for any approximate correction z. Then
e−z=A⁻¹d. With the actual coefficient minimum a_min, A ≥ a_min B. Energy
Cauchy–Schwarz and inverse ordering give the baseline

    |(e−z)_i| ≤ sqrt((A⁻¹)_ii dᵀA⁻¹d)
               ≤ sqrt((B⁻¹)_ii dᵀB⁻¹d) / a_min = beta_i.

`solver/poisson.py` computes B⁻¹ by the orthonormal DST-II matching the
cell-centered Dirichlet stencil. `uncertainty/rank_bounds.py` computes its
inverse diagonal from separable eigenvectors, with O(N³) work once per grid and
O(N²) storage. Neither operation requires a variable-coefficient inverse.

For the tighter flux majorant, include grounded boundary faces in the incidence
matrix D and write A=D W_A Dᵀ, B=D W_B Dᵀ. Interior W_B=N² and each wall-face
W_B=2N². Solve p=B⁻¹d once and let q=W_B Dᵀp, so Dq=d. For δ=A⁻¹d,

    dᵀδ = (Dᵀδ)ᵀq ≤ sqrt(δᵀAδ) sqrt(qᵀW_A⁻¹q),
    hence dᵀA⁻¹d ≤ E_eq := qᵀW_A⁻¹q.

Consequently

    beta_flux_i = sqrt((B⁻¹)_ii E_eq / a_min) ≤ beta_i,

because W_A ≥ a_min W_B implies E_eq ≤ dᵀB⁻¹d/a_min. The implemented face sum is

    E_eq = N² Σ_interior (p_i−p_j)² / harmonic(a_i,a_j)
           + 2N² Σ_wall_faces p_i²/a_i.

Every incident wall face is included, including two at each corner. Both
majorants share **one** Poisson solve; the tighter variant adds weighted face
sums. This does not assert identical runtime. It is the classical equilibrated
flux/Thomson argument specialized to the discrete operator.

The reverse triangle inequality gives magnitude intervals
L=max(|z|−beta,0), U=|z|+beta. For m cells and k=ceil(0.1m), a cell is certainly
in the top k if its lower endpoint exceeds the (k+1)-st largest upper endpoint;
it is certainly outside if its upper endpoint is below the k-th largest lower
endpoint. Strict comparisons leave ambiguous ties unresolved. Certified top-10%
recall is the number flagged inside divided by k; resolved fraction counts both
inside and outside flags. Corrected prediction is u_hat−z, not u_hat+z.

## Fixed-budget flux results and numerical limits

Poisson-preconditioned CG uses zero start, 20/40 iteration caps, rtol=8 times
float64 epsilon, and atol=0. Actual executed counts and convergence status are
saved. No reference error enters correction, majorant, or flag construction.
The flux sweep includes original/fresh N=32/64 × contrasts 10/100/500, plus the
four OOD test sets: 3,200 fields total. It performs 13,926,400 pointwise checks
per majorant. Bootstrap CIs use 5,000 paired whole-field resamples, seed 20260912
plus the source index in the saved configuration. They condition on the frozen
predictors and sampled field distribution, and are not simultaneous intervals.

| Fields, PCG budget | Baseline top-10% recall | Flux top-10% recall | Paired gain, 95% CI |
|---|---:|---:|---:|
| Fresh N=32, contrast 500, 20 | 0.18728 | 0.25709 | 0.06981 [0.05913, 0.08136] |
| Fresh N=64, contrast 500, 20 | 0.10912 | 0.16304 | 0.05391 [0.04412, 0.06432] |
| OOD geometry, 20 | 0.41568 | 0.55617 | 0.14049 [0.12961, 0.15097] |
| OOD roughness, 20 | 0.00782 | 0.06058 | 0.05277 [0.03529, 0.07107] |
| OOD roughness, 40 | 0.34369 | 0.58398 | 0.24029 [0.21184, 0.26879] |

Fresh N=64, contrast-500 resolved-cell fraction at 20 iterations rises from
0.21444 to 0.30986. Roughness resolved fraction at 40 rises from 0.54971 to
0.79106. At 20, its correction still has relative prediction L2 error 0.007721;
at 40, 0.000796. Thus strong ranks or certification may require substantial
solution correction. These are informativeness results, not speedup claims.

There are zero observed false high/low flags, zero lost baseline flags under
the tighter bound, and zero bound exceedances beyond the declared evaluation
tolerance

    1e-10 * max(max(abs(u_hat−u)), 1e-12) + 1e-13.

This tolerance is **never added to the analytic radius or used to form flags**.
Raw positive float64 exceedances occur near convergence: 732,475 for the baseline
and 913,764 for flux, with maximum excess divided by max absolute reference error
2.35e-12 and 2.48e-12 respectively. These checks compare floating-point solves;
the implementation is not outward rounded or a formal roundoff enclosure.
Saying “zero bound violations” without this qualification would be misleading.
The separate OOD-only baseline run has six raw exceedances, all below tolerance.

Independent tests cover the incidence factorization and equilibrium, constant
coefficients, dense/sparse inverse comparisons, boundary scaling including N=1,
pointwise bounds, strict flags and ties, invalid inputs, and deliberately false
bounds. The numerical implementation and experiment protocol received separate
read-only reviews.

## Additional U-Net evaluation control

A separate convolutional surrogate checks whether the localization pattern is
confined to the original FNO implementation. This is an additional evaluation
control, not an equal-capacity or equal-compute architecture benchmark. The
two-level U-Net has channels 16/32/64, 117,681 parameters, GroupNorm/GELU blocks,
max-pooling, transpose-convolution upsampling, and skip concatenation. It has no
Fourier layers; **GroupNorm uses global spatial statistics within each group**,
so the complete predictor is not strictly local.

Each initialization seed 0/1/2 trains for exactly 200 epochs on the same 700
seed-0 fields at N=32, contrast 100, with Adam, initial learning rate 0.001,
cosine decay, batch size 32, and normalized training MSE. All normalization uses
training arrays; the four fitted constants exactly reproduce the archived FNO
constants. All three checkpoints are frozen before fresh references are loaded.
No fresh label chooses epochs, checkpoints, seeds, or hyperparameters. Three-epoch
training-only timing pilots at widths 16 and 12 are preserved; width 12 was
excluded without fresh evaluation, and the originally proposed width 16 was
retained. CPU thread limits include explicit `VECLIB_MAXIMUM_THREADS=1`.

Evaluation uses the identical 200 fresh seed-20260912 fields, with 5,000 paired
whole-field bootstrap draws from seed 20260912. The ensemble's mean relative L2
error is 0.04407 [0.03747, 0.05148], versus 0.10541 for the original 80-epoch FNO
on those fields. Capacity and training-duration differences prevent attributing
this accuracy difference to architecture alone. The localization result persists:

| U-Net score | Mean per-field Spearman, 95% CI |
|---|---:|
| Raw residual | 0.23089 [0.21132, 0.25046] |
| Prediction magnitude | 0.42090 [0.39143, 0.45050] |
| Ensemble spread | 0.41618 [0.39300, 0.43959] |
| Diagonal-PCG20 | 0.64069 [0.61727, 0.66456] |
| Full AMG cycle | 0.87917 [0.86183, 0.89572] |
| Poisson-PCG20 | 0.999973 [0.999965, 0.999981] |

The paired prediction-magnitude minus raw-residual gain is 0.19001
[0.14968, 0.22974]; Poisson-PCG20 minus AMG is 0.12080 [0.10425, 0.13813].
Poisson-PCG20 also reduces mean relative prediction L2 error to 8.42e-5, so the
ranking improvement accompanies substantial solution correction. The same-A
identity's largest relative discrepancy is 1.22e-11. Eleven model/protocol tests
pass, and an independent verifier reloads every checkpoint and reproduces all
600 member-field predictions exactly using a separately written physical
prediction loop. Source/cache hashes remain unchanged; the figure was checked.
Complete settings, per-field metrics and hashes are in
`results/unet_robustness/TRAINING_PROTOCOL.md`, `summary.json` and the corresponding CSV files.

An explicitly exploratory, fit-free application of the existing flux evaluator
to this frozen U-Net cache gives certified top-10% recall 0.64820 → 0.75252 at
20 iterations, paired gain 0.10432 [0.09282, 0.11597]. Resolved-cell fraction is
0.83567 → 0.91637. At 40 iterations both majorants give recall 0.999951 and
resolved fraction 0.999990. There are zero false high/low flags, zero lost
baseline flags, and zero exceedances beyond the same evaluation-only tolerance
defined above. At 20 there are no raw floating-point exceedances; at 40 there
are 523 for the baseline and 747 for flux, with maximum relative excess
2.64e-12 and 2.92e-12. These 200 fields are a separate supporting analysis and
are **not included** in the earlier 3,200-field flux-sweep counts. No formal
floating-point, architecture-independent, or solver-speedup claim follows.

## Replay and preserved provenance

Run from the repository root; use new output directories when preserving these
audits. The saved scripts default to the settings reported above.

```bash
.venv/bin/python analysis/audit_submission_splits.py
.venv/bin/python analysis/audit_ood_protocol.py --refresh-amg --out-dir results/ood_audit_fullamg_20260912 --bootstrap 2000
.venv/bin/python analysis/check_id_risk_controls.py
.venv/bin/python experiments/exp_rank_bounds_ood.py
.venv/bin/python experiments/exp_flux_bounds.py --cache-dirs results/submission_32 results/submission_64 results/submission_fresh_32 results/submission_fresh_64 --ood-bundle results/ood_features_bundle.npz --budgets 20 40 --out results/flux_bounds_all
.venv/bin/python experiments/exp_unet_robustness.py --out results/unet_robustness_replay
.venv/bin/python analysis/verify_unet_artifacts.py
.venv/bin/python experiments/exp_flux_bounds.py --cache-dirs results/unet_robustness --kappas 100 --budgets 20 40 --out results/flux_bounds_unet
.venv/bin/python -m pytest tests/test_conformal.py tests/test_conformal_protocol.py tests/test_rank_bounds.py tests/test_rank_bounds_ood.py tests/test_flux_bounds.py -q
.venv/bin/python -m pytest tests/test_unet_robustness.py -q
```

U-Net training refuses to overwrite existing checkpoints. The explicit new
output directory above is for a retraining replay; the verifier defaults to the
original completed run and accepts `--directory` for another run. Both the
U-Net JSON configuration and NPZ signature contain the original local output
path, which must be transformed only in anonymous package copies as described
below.

The primary immutable OOD bundle SHA256 is
`17941221fabba420c672a78279f3d8ed284887f61767634d89b2985087ca7137`.
The independently refreshed AMG bundle SHA256 is
`3dbdb6a1f71ba7f3f6a0dd17c84d03f8e111b7145677e22de46b2b82c1c02ccb`.
All other exact input/source digests are retained in the listed JSON files and
`results/ood_audit_fullamg_20260912/handoff_provenance.json`. Executable audits
verify that source and input files remain unchanged during their run.

For anonymous distribution, remove machine-specific absolute path prefixes in
**package copies** of audit metadata and record hashes of those derived copies.
Preserve original local audit artifacts and their original hashes. A prose-only
generator edit changes its source digest even if arrays do not change; do not
silently relabel old cached signatures as produced by the new source. The
documentation-only change is recorded in `reproducibility/generator_documentation_update.json`,
with the prior source retained under `reproducibility/source_snapshots/` and a
machine-checked comparison showing that executable abstract syntax is unchanged.
