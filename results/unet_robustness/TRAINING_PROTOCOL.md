# Fixed U-Net robustness protocol

This supporting run checks one convolutional architecture on the same coefficient family and fresh fields as the FNO benchmark. It is not a model leaderboard or an isolated causal comparison of architectures.

The chosen model is the repository's established two-level U-Net, with width 16 (channels 16/32/64), GroupNorm/GELU blocks, max-pooling, transpose-convolution upsampling and skip concatenation. It has no Fourier layers; GroupNorm does use spatial statistics, so the complete predictor is not strictly local.

Each of initialization seeds 0/1/2 trains for exactly 200 epochs on 700 Family-C fields at N=32 and contrast 100, using generator seed 0. Adam uses initial learning rate 0.001, cosine decay over 200 epochs, batch 32 and MSE on training-normalized targets. Input/output normalization uses only training arrays. All three trained checkpoints are saved, reloaded and checked before fresh evaluation labels are opened. No validation or fresh-test error chooses epochs, checkpoints, hyperparameters or seeds.

A width 16, three-training-epoch timing pilot projected 35 minutes for the full run. A width 12 timing-only pilot projected 22 minutes; it did not load test labels and was excluded. The original width 16 capacity was retained when the available time allowed 35 minutes. Both pilot logs/metadata are preserved. Main-run settings remain the original width 16 proposal.

Fresh evaluation uses all 200 cached generator-seed 20260912 fields, whose exact coefficient hashes match the FNO benchmark. Scores are raw residual magnitude, prediction magnitude, ensemble spread, unpreconditioned/diagonal/Poisson CG20 and one current full AMG cycle. Reference errors enter after score construction. Same-A residual identity is a mandatory gate. Every method uses the same fields; 5,000 paired field-bootstrap resamples give percentile confidence intervals conditional on the three trained networks. No end-to-end speedup is inferred.

The initial U-Net checkpoint reproduces all four archived N=32, contrast 100 FNO training normalization constants exactly: log-coefficient mean/std and solution mean/std. This provides additional evidence for the declared shared training draws. It does not reconstruct unrecorded historical model-selection decisions.

Final numerical results and complete source/checkpoint hashes are saved to `summary.json` when the fixed run finishes. The original U-Net historical artifacts and FNO checkpoints remain unchanged.
