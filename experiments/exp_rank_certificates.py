"""Evaluate analytic pointwise error bounds on frozen learned errors.

The bounds use only A, r, an approximate correction, and a constant-Poisson
inverse. Ground truth is used solely to evaluate correctness and informativeness.
No fitted parameters, conformal calibration, or learned thresholds are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from solver.assemble_fd import assemble
from solver.poisson import poisson_preconditioner
from uncertainty.rank_bounds import poisson_error_bound, magnitude_envelope, separate_top_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    if args.samples < 2 or any(k < 1 for k in args.budgets):
        parser.error("samples >=2 and positive budgets required")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "discrete_rank_enclosures", "per_kappa": {},
                "bound": "sqrt(diag(B^-1) * d^T B^-1 d)/a_min, B=A(1), d=r-Az",
                "scope": "Analytic exact-arithmetic bound for this discrete TPFA system; float64 implementation independently checked, not interval-arithmetic certification.",
                "top_fraction": .1, "bootstrap_replicates": 5000, "bootstrap_seed": 20260912,
                "source_hashes": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [REPO / "uncertainty/rank_bounds.py", REPO / "solver/poisson.py"]}}
    for path in sorted(args.cache_dir.glob("predictions_k*.npz")):
        kappa = path.stem.split("_k")[-1]
        with np.load(path, allow_pickle=False) as data:
            a, f, u = (data[name][:args.samples] for name in ("a", "f", "u"))
            prediction = data["members"][:, :args.samples].astype(np.float64).mean(axis=0)
        n, N, _ = a.shape
        if n != args.samples:
            raise ValueError(f"requested {args.samples} fields but {path} has {n}")
        M = poisson_preconditioner(N)
        rows = []
        for i in range(n):
            A = assemble(a[i])
            error = prediction[i] - u[i]
            rhs = A @ prediction[i].reshape(-1) - f[i].reshape(-1)
            truth_top = np.zeros(N*N, dtype=bool)
            n_top = int(np.ceil(.1 * N*N))
            truth_top[np.argpartition(np.abs(error).reshape(-1), -n_top)[-n_top:]] = True
            reference_scale = max(float(np.max(np.abs(error))), 1e-12)
            for budget in args.budgets:
                z, info = cg(A, rhs, M=M, x0=np.zeros_like(rhs), rtol=8*np.finfo(float).eps,
                             atol=0., maxiter=budget)
                if info < 0 or not np.isfinite(z).all():
                    raise RuntimeError(f"Poisson-PCG failed on {path}, field{i}")
                defect = (rhs - A @ z).reshape(N, N)
                correction = z.reshape(N, N)
                bound = poisson_error_bound(defect, float(a[i].min()))
                lower, upper = magnitude_envelope(correction, bound)
                high, low = separate_top_set(lower, upper, .1)
                signed_miss = np.abs(error - correction) - bound
                # Distinguish exact-arithmetic theory from numeric evaluation.
                # This tolerance compares two double-precision sparse solves;
                # it is NOT added to the bound or used to create rank flags.
                numeric_tolerance = 1e-10 * reference_scale + 1e-13
                false_high = int(np.sum(high.reshape(-1) & ~truth_top))
                false_low = int(np.sum(low.reshape(-1) & truth_top))
                violations = int(np.sum(signed_miss > numeric_tolerance))
                if false_high or false_low or violations:
                    raise AssertionError(f"bound/rank failure kappa{kappa} field{i} budget{budget}: "
                                         f"high={false_high},low={false_low},bound={violations}")
                rows.append({
                    "sample": i, "iterations": budget,
                    "resolved_fraction": float(np.mean(high | low)),
                    "certified_high_recall": float(high.sum() / n_top),
                    "certified_high_count": int(high.sum()), "certified_low_count": int(low.sum()),
                    "mean_bound_over_mean_abs_error": float(bound.mean() / np.mean(np.abs(error))),
                    "relative_l2_correction_error": float(np.linalg.norm(error - correction) / np.linalg.norm(error)),
                    "max_bound_excess_over_error_scale": float(max(0., signed_miss.max()) / reference_scale),
                    "false_high": false_high, "false_low": false_low, "bound_violations": violations,
                })
            if (i+1) % 50 == 0:
                print(f"rank bounds N={N} kappa={kappa}: {i+1}/{n}", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(args.out / f"fields_k{kappa}.csv", index=False)
        draws = np.random.default_rng(20260912).integers(0, n, size=(5000, n))
        summaries = {}
        for budget, block in frame.groupby("iterations"):
            summaries[str(budget)] = {}
            for metric in ["resolved_fraction", "certified_high_recall", "mean_bound_over_mean_abs_error", "relative_l2_correction_error"]:
                values = block[metric].to_numpy()
                low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
                summaries[str(budget)][metric] = {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}
        manifest["per_kappa"][kappa] = {
            "grid_N": N, "n_fields": n, "budgets": summaries,
            "prediction_cache": str(path), "prediction_cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "false_high_total": int(frame.false_high.sum()), "false_low_total": int(frame.false_low.sum()),
            "bound_violation_total": int(frame.bound_violations.sum()),
            "max_numeric_bound_excess_relative": float(frame.max_bound_excess_over_error_scale.max()),
        }
        (args.out / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(f"kappa{kappa}: " + ", ".join(f"k={b} resolved={v['resolved_fraction']['mean']:.3f}, top-recall={v['certified_high_recall']['mean']:.3f}"
                                            for b, v in summaries.items()), flush=True)
    if not manifest["per_kappa"]:
        raise FileNotFoundError(f"no prediction caches in {args.cache_dir}")
    print("PASS: no observed bound violations or false rank flags", flush=True)


if __name__ == "__main__":
    main()
