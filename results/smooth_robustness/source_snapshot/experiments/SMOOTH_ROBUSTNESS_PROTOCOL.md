# Smooth coefficient robustness check

Protocol fixed September 12, 2026, before training or examining new test results.
This is a subsequent exploratory robustness experiment, not a preregistered
study or a replacement for the frozen binary benchmark.

## Question and controlled settings

Does poor residual localization also occur for a predictor trained and tested
on smoothly varying, non-binary coefficients? Testing a binary-trained network
only on smooth fields would confound this question with distribution shift.

Use a separately trained three-member FNO ensemble, at N=32 and contrast100,
with the original architecture (12 modes, width24, three layers), 700 training
fields, 80 fixed epochs, batch32, Adam lr0.001 and cosine decay. Model seeds
are0,1,2. Retain the original forcing, zero Dirichlet boundary conditions,
harmonic-face TPFA operator and physical normalization fitted on training only.
There is no validation selection, early stopping or test-driven tuning.

For each coefficient draw, Gaussian-filter independent normal noise with
standard deviation0.12 domain lengths and wrap boundary mode. Scale the filtered
field by its sampled minimum and maximum to t in[0,1], then set a=exp(log(100)t).
This fixes the sampled extrema at1 and100 while retaining intermediate values.
It is a bounded smooth random family, not an unchanged Gaussian/lognormal law.
Use data seed20260913 for training and20260914 for200 independent test fields.
Freeze all checkpoints before generating the test solutions. Check coefficient
array hashes for exact train/test duplicates; distinct seeds alone are not the
verification. A training-only timing pilot may assess runtime but must not alter
the fixed scientific configuration using test outcomes.

Changing the coefficient family also changes its histogram and effective
conductivity. This is evidence about robustness beyond binary media, not a
causal estimate of interface sharpness and not a test of resolution transfer.

## Outcomes fixed before evaluation

Primary outcomes are mean within-field Spearman correlation of |r| with |e|,
and the paired difference between Poisson-PCG20 and |r|. Report the result
regardless of its direction or size. Secondary outcomes are prediction relative
L2 error, top10% overlap, remaining correction norm, and comparisons with
prediction magnitude, boundary distance, ensemble spread, CG20, diagonal-PCG20
and one full AMG cycle. Retain Poisson-PCG budgets5,10,20,40,80 and actual
iterations executed. Compare classical Poisson and equilibrated-flux enclosures
at20 and40 iterations; do not assume that their informativeness will match the
binary case. Retain per-member prediction error and raw-residual correlations.

Intervals use5,000 paired whole-field percentile bootstrap draws with
seed20260915. They are conditional on the trained ensemble, not uncertainty
over training datasets. Binary/smooth comparisons are descriptive, unpaired
comparisons between independently trained models and sampled field families.
Iteration counts are work descriptors, not equal-runtime comparisons. Existing
wall-clock claims continue to concern the original binary benchmark.

## Verification

Retain source snapshots, configuration, training histories, checkpoints,
normalization, train/test arrays, per-field metrics, bootstrap summaries and
source/output hashes. Reload checkpoints and independently reproduce prediction
arrays. Reconstruct test fields from their seed and compare exact arrays.
Check the same-operator identity with a sparse direct solve (relative tolerance
1e-8), nonnegative cell energies summing to e^T A e, and bound enclosures and
inside/outside flags. The unchanged numerical enclosure tolerance is
1e-10*max(max(|e|),1e-12)+1e-13; it is used only for evaluation and never added
to bounds or rank decisions. Check code and numerical results independently
before integrating any claim. Do not discard inconvenient fields or rerun seeds
to improve the result.
