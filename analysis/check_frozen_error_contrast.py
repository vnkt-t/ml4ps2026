"""Separate changing operator contrast from changing learned error fields.

This is an algebraic control, not a new PDE generalization experiment. Freeze
each learned error and binary geometry, change only the operator's high/low
coefficient ratio, and recompute r_kappa=A_kappa e. Prespecified contrasts include
the homogeneous operator. No network is retrained or recalibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from solver.assemble_fd import assemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/submission_v1_32/predictions_k100.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/frozen_error_contrast"))
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        a = data["a"]
        e = data["members"].astype(np.float64).mean(axis=0) - data["u"]
    kappas = [1, 2, 5, 10, 50, 100, 500]
    rows = []
    for i, (coeff, error) in enumerate(zip(a, e)):
        low, high = float(coeff.min()), float(coeff.max())
        if low == high or not np.all((coeff == low) | (coeff == high)):
            raise ValueError("this control requires binary coefficient geometry")
        mask = coeff == high
        for kappa in kappas:
            operator = assemble(np.where(mask, float(kappa), 1.0))
            residual = operator @ error.reshape(-1)
            rows.append({"sample": i, "operator_kappa": kappa,
                         "spearman": float(spearmanr(np.abs(residual), np.abs(error).reshape(-1)).statistic)})
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_suffix(".csv"), index=False)
    wide = frame.pivot(index="sample", columns="operator_kappa", values="spearman")
    draws = np.random.default_rng(20260912).integers(0, len(a), size=(5000, len(a)))

    def summary(values):
        v = np.asarray(values)
        low, high = np.quantile(v[draws].mean(axis=1), [.025, .975])
        return {"mean": float(v.mean()), "ci95_low": float(low), "ci95_high": float(high)}

    result = {
        "experiment": "frozen_error_contrast", "n_fields": len(a), "grid_N": int(a.shape[1]),
        "source_predictions": str(args.predictions),
        "source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "protocol": "Freeze the learned error and binary inclusion/channel geometry; vary only A. Algebraic diagnostic, not forcing or coefficient generalization of the learned predictor.",
        "bootstrap": "5000 paired field resamples, seed 20260912",
        "per_operator_kappa": {str(k): summary(wide[k]) for k in kappas},
        "paired_k500_minus_k10": summary(wide[500] - wide[10]),
        "paired_k500_minus_k100": summary(wide[500] - wide[100]),
        "fraction_fields_monotonically_nonincreasing": float(np.mean(np.all(np.diff(wide[kappas], axis=1) <= 0, axis=1))),
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
