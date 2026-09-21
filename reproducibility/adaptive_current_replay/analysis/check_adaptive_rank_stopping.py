"""Evaluate a declared rank-driven stopping rule on cached learned errors.

Protocol fixed before these trials: q=0.1, certified top-set recall target=0.9,
check every five Poisson-PCG steps, maximum 80 steps. Ground truth is used only
for evaluation after the adaptive solver has returned. No training, tuning,
calibration, or oracle is used to choose the stopping point.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from solver.adaptive import adaptive_poisson_cg
from solver.assemble_fd import assemble


def mean_ci(values, draws):
    arr = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(arr[draws].mean(axis=1), [0.025, 0.975])
    return {"mean": float(arr.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "summary.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed adaptive evaluation: {output}")
    source_paths = [REPO / "solver/adaptive.py", REPO / "solver/poisson.py",
                    REPO / "uncertainty/rank_bounds.py", REPO / "uncertainty/flux_bounds.py", Path(__file__).resolve()]
    payload = {
        "experiment": "adaptive_rank_stopping",
        "protocol": {"q": 0.1, "target_certified_high_recall": 0.9,
                     "check_every": 5, "maxiter": 80,
                     "bound_kind": args.bound_kind,
                     "declaration": "defaults fixed before evaluation; no outcome tuning",
                     "stopping_inputs": "A, residual, coefficient minimum; no true error or oracle",
                     "uncertainty": "field-bootstrap, conditional on frozen trained networks and selected family",
                     "bootstrap_replicates": args.bootstrap, "bootstrap_seed": 20260912,
                     "fft_count": "forward or inverse 2D DST calls, two per Poisson solve",
                     "scope": "exact-arithmetic rank-bound theorem, float64 implementation without outward rounding"},
        "source_hashes": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "results": {},
    }
    for directory in args.cache_dirs:
        paths = sorted(directory.glob("predictions_k*.npz"))
        if not paths:
            raise FileNotFoundError(f"no prediction caches in {directory}")
        for path in paths:
            kappa = path.stem.split("_k")[-1]
            with np.load(path, allow_pickle=False) as cache:
                a, f, u = [cache[name][:args.samples] for name in ("a", "f", "u")]
                prediction = cache["members"][:, :args.samples].astype(np.float64).mean(axis=0)
                cache_signature = json.loads(str(cache["signature"].item()))
            count, n, _ = a.shape
            if count != args.samples:
                raise ValueError(f"requested {args.samples} fields but {path} contains {count}")
            records, histories = [], []
            for i in range(count):
                A = assemble(a[i])
                rhs = (A @ prediction[i].reshape(-1) - f[i].reshape(-1)).reshape(n, n)
                result = adaptive_poisson_cg(A, rhs, float(a[i].min()),
                                             bound_kind=args.bound_kind, coefficient=a[i])
                # True error enters only here, after the correction and decision.
                error = prediction[i] - u[i]
                top = np.zeros(n*n, dtype=bool)
                top[np.argsort(np.abs(error).reshape(-1))[-result.n_top:]] = True
                false_high = int(np.sum(result.high.reshape(-1) & ~top))
                false_low = int(np.sum(result.low.reshape(-1) & top))
                if false_high or false_low:
                    raise AssertionError(f"false adaptive rank flag at {path}, field {i}")
                if result.target_met != (result.high.sum() >= result.target_high_count):
                    raise AssertionError("stopping result is inconsistent with strict certified count")
                denominator = float(np.linalg.norm(error))
                if denominator == 0:
                    raise ValueError("evaluation encountered zero true error; top-error ranking is undefined")
                records.append({
                    "sample": i, "iterations": result.iterations, "status": result.status,
                    "target_met": int(result.target_met),
                    "certified_high_recall": result.certified_high_recall,
                    "certified_high_count": int(result.high.sum()),
                    "resolved_fraction": float(np.mean(result.high | result.low)),
                    "relative_l2_correction_error": float(np.linalg.norm(error - result.correction) / denominator),
                    "cg_operator_applications": result.cg_operator_applications,
                    "extra_operator_applications": result.extra_operator_applications,
                    "preconditioner_applications": result.preconditioner_applications,
                    "bound_poisson_solves": result.bound_poisson_solves,
                    "fft_transforms": result.fft_transforms,
                    "false_high": false_high, "false_low": false_low,
                })
                histories.append({"sample": i, "history": list(result.history)})
                if (i + 1) % 50 == 0 or i + 1 == count:
                    print(f"adaptive {directory.name}, N={n}, kappa={kappa}: {i+1}/{count}", flush=True)
            frame = pd.DataFrame(records)
            key = f"{directory.name}_k{kappa}"
            frame.to_csv(args.out / f"fields_{key}.csv", index=False)
            (args.out / f"histories_{key}.json").write_text(json.dumps(histories, indent=2) + "\n")
            draws = np.random.default_rng(20260912).integers(0, count, size=(args.bootstrap, count))
            metrics = ["iterations", "target_met", "certified_high_recall", "resolved_fraction",
                       "relative_l2_correction_error", "cg_operator_applications", "extra_operator_applications",
                       "preconditioner_applications", "bound_poisson_solves", "fft_transforms"]
            payload["results"][key] = {
                "grid_N": n, "kappa": float(kappa), "n_fields": count,
                "cache": str(path), "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cache_signature": cache_signature,
                "metrics": {name: mean_ci(frame[name], draws) for name in metrics},
                "iteration_distribution": dict(sorted(Counter(frame.iterations.tolist()).items())),
                "status_counts": dict(Counter(frame.status.tolist())),
                "iteration_median": float(frame.iterations.median()),
                "iteration_p90": float(frame.iterations.quantile(0.9)),
                "false_high_total": int(frame.false_high.sum()), "false_low_total": int(frame.false_low.sum()),
            }
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(output)
            print(f"N={n} k={kappa}: target met {frame.target_met.mean():.3f}, "
                  f"mean iterations {frame.iterations.mean():.2f}, p90 {frame.iterations.quantile(.9):.0f}, "
                  f"mean correction L2 {frame.relative_l2_correction_error.mean():.4g}", flush=True)
    print("PASS: all adaptive flags independently correct; unmet targets reported explicitly", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bound-kind", choices=("poisson", "flux"), default="poisson")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.bootstrap < 1:
        parser.error("samples >=2 and bootstrap >=1 required")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
