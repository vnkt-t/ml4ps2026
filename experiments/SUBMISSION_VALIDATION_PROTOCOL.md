# Submission validation protocol, fixed before fresh-field evaluation

2026-09-12. The original test set has been used for exploratory development; it
must not be described as a pristine final test. We therefore distinguish:

- **Original fields:** generator seed 0, skip the 700 training draws, then use
  the next 200 fields. Reproduces the published-in-repository pilot evaluation.
- **Fresh fields:** generator seed 20260912, no skipped draws, 200 fields. Use
  unchanged model checkpoints, normalization, methods, iteration budgets and
  statistics. No model selection or calibration uses these fresh error labels.

Both sets are evaluated at N=32 and N=64, contrasts 10/100/500, with three-member
FNO ensembles (training seeds 0/1/2). The N=64 models were trained separately;
this is a resolution replication, not zero-shot resolution transfer. Prediction
and field hashes are retained. Each contrast uses the same random geometry draws.

Comparators fixed before drawing fresh fields: ensemble standard deviation;
prediction magnitude; distance to nearest domain boundary; absolute residual;
Gaussian smoothing of absolute residual (sigma 1 cell); undamped Jacobi at
1/5/10/20/40/80 sweeps; unpreconditioned, diagonal-preconditioned and
Poisson-preconditioned CG at 5/10/20/40/80 iterations; constant-coefficient
Poisson inverse via DST-II; one full-hierarchy PyAMG V-cycle; sparse-direct oracle.
The primary table uses 20 iterations; larger budgets diagnose the dependence on
that choice and must not be presented as tuned stopping rules.

Targets: pointwise absolute solution error; nonnegative face energy allocation
whose sum is e^T A e including Dirichlet walls; RMS face flux magnitude. Primary
metric: mean per-field spatial Spearman correlation. Secondary: per-field
top-10% AUROC, overlap and absolute-error capture. Correction diagnostics also
record relative L2 and A-norm errors, which answer different questions from ranks.

Statistics: percentile intervals from 5,000 bootstrap resamples of independent
coefficient fields, RNG seed 20260912. Method differences are paired within field.
Intervals condition on the trained networks; they do not represent uncertainty
over all possible training runs. Larger-grid fields are not pooled as extra cells.

No filtering based on favorable localization is allowed. All requested fields
are retained. The same-A identity and energy allocation must pass; an actual AMG
backend is required for an AMG claim. The pointwise error target is the discrete
reference solution, not the continuum PDE error.

Additional prespecified robustness run: train the identical kappa=100 FNO for
240 epochs (versus original 80), same 700 fields and seeds 0/1/2, with all other
training settings fixed. This isolates training-duration sensitivity. Do not
select the best epoch from test errors or overwrite the original checkpoints.

Frozen-error contrast control is explicitly algebraic: hold the kappa=100 learned
error and binary geometry fixed, vary only A's contrast over 1/2/5/10/50/100/500,
then score |A e| against |e|. This does not test the learned model on a new forcing.

Timing is a separate isolated experiment after training stops. Label the shared
starting state (A assembled, prediction available), count coefficient-dependent
setup per field, and distinguish cached constant-coefficient setup and warm-cycle
diagnostics. PyAMG cycle complexity is a structural estimate, not elapsed cost.

## Subsequent exploratory analyses

The deterministic Poisson/flux enclosures, adaptive membership stopping rule,
and associated OOD diagnostics were developed after the primary fresh-field
score comparisons. They are exploratory analyses, not a prospectively registered
blind validation. They fit no parameters to true errors. The top fraction0.1,
recall target0.9, checks every5steps and maximum80steps were fixed before their
respective sweeps. Flux bounds were evaluated at budgets20/40, not selected by
an optimum on fresh labels. All results, including conservative/unresolved cases
and the extra runtime of flux checks, are retained.

The additional U-Net replication uses a fixed200-epoch budget and seeds0/1/2.
Training-only timing pilots determine feasible width; they do not load fresh
labels. The final width16 run freezes all3members before fresh evaluation, and
its complete protocol is retained with the results. This is an architecture
sensitivity check, not an architecture-independent or state-of-the-art claim.
