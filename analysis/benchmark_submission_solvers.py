"""Paired wall-time benchmark on cached submission-validation prediction fields.

Run when the laptop is otherwise idle. Every fresh solve starts from an assembled
A and an available neural prediction, charges residual construction, and computes
the absolute correction score. AMG rebuilds its hierarchy and lazy coarse solve;
the direct oracle refactorizes. No vcycle instrumentation is timed as algorithm
work. Method order is randomized within field/repetition and fields are the
resampling unit. Frozen scientific-result artifacts are never overwritten.

Example:
  .venv/bin/python analysis/benchmark_submission_solvers.py \
    --cache-dirs results/submission_v1_32 results/submission_v1_64 \
    --samples 30 --repeats 3 --out results/submission_timing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

# NumPy uses Apple Accelerate on this machine while SciPy uses OpenBLAS.
# threadpoolctl detects only the latter: set the vecLib limit before importing
# NumPy as well, and record the actual configured backends in the artifact.
for thread_setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_setting] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import pyamg
import scipy
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr
from threadpoolctl import threadpool_info, threadpool_limits

from solver.assemble_fd import assemble
from solver.adaptive import adaptive_poisson_cg
from solver.multigrid import _residual_reduced
from solver.poisson import poisson_preconditioner, solve_poisson
from solver.relaxation import cg_correction, jacobi_sweep


def _one_amg_cycle(hierarchy, rhs):
    correction = np.asarray(hierarchy.solve(
        rhs, x0=np.zeros_like(rhs), maxiter=1, cycle="V", tol=0.0, accel=None
    ), dtype=np.float64)
    if not _residual_reduced(hierarchy.levels[0].A, correction, rhs):
        raise FloatingPointError("AMG benchmark cycle did not pass the ladder acceptance check")
    return correction


def _amg_once(A, rhs):
    return _one_amg_cycle(pyamg.smoothed_aggregation_solver(A), rhs)


def _poisson_cg(A, rhs, M, k):
    correction, info = spla.cg(
        A, rhs, x0=np.zeros_like(rhs), M=M, maxiter=k,
        atol=0.0, rtol=8.0 * np.finfo(np.float64).eps,
    )
    if info < 0 or not np.all(np.isfinite(correction)):
        raise FloatingPointError(f"Poisson-PCG benchmark failed with info={info}")
    return correction


def benchmark_field(a, prediction, source, truth, rng, repeats, field_seed, budgets):
    n = a.shape[0]
    A = assemble(a)
    uh, f = prediction.reshape(-1), source.reshape(-1)
    error = prediction.reshape(-1) - truth.reshape(-1)
    # Constant Poisson setup is reusable for every coefficient field at this N.
    poisson_M = poisson_preconditioner(n)
    solve_poisson(np.zeros((n, n)))  # initialize O(N²) eigenvalue cache outside timings

    corrections = {
        "raw": lambda r: r,
        "jacobi20": lambda r: jacobi_sweep(A, r, np.zeros_like(r), 20),
        "cg20": lambda r: cg_correction(A, r, 20)[0],
        "poisson": lambda r: solve_poisson(r.reshape(n, n)).reshape(-1),
        "amg": lambda r: _amg_once(A, r),
        "oracle": lambda r: spla.spsolve(A.tocsc(), r),
    }
    adaptive_results = {}

    def adaptive_correction(rhs, bound_kind):
        result = adaptive_poisson_cg(A, rhs, float(a.min()), bound_kind=bound_kind,
                                     coefficient=a if bound_kind == "flux" else None)
        adaptive_results[bound_kind] = result
        return result.correction

    corrections["adaptive_rank90"] = lambda r: adaptive_correction(r, "poisson")
    corrections["adaptive_flux90"] = lambda r: adaptive_correction(r, "flux")
    for k in budgets:
        corrections[f"pcg{k}"] = lambda r, k=k: cg_correction(A, r, k, diagonal_preconditioner=True)[0]
        corrections[f"poisson_pcg{k}"] = lambda r, k=k: _poisson_cg(A, r, poisson_M, k)

    # Optional reuse scenarios: setup is excluded ONLY from these explicitly
    # labeled rows, and separately recorded. This is not the fresh-A experiment.
    initial_rhs = A @ uh - f
    np.random.seed(field_seed)
    t0 = time.perf_counter()
    hierarchy = pyamg.smoothed_aggregation_solver(A)
    _one_amg_cycle(hierarchy, initial_rhs)  # initializes lazy coarse solver
    amg_setup_and_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    factor = spla.splu(A.tocsc())
    direct_factor_s = time.perf_counter() - t0
    corrections["amg_reused"] = lambda r: _one_amg_cycle(hierarchy, r)
    corrections["oracle_reused"] = factor.solve
    rows = []
    for repeat in range(repeats):
        for name in rng.permutation(list(corrections)):
            # Fix AMG setup's randomized spectral estimate across repetitions.
            # Seed-setting is experiment infrastructure and outside timed work.
            np.random.seed(field_seed)
            start = time.perf_counter()
            residual = A @ uh - f
            x = np.asarray(corrections[name](residual)).reshape(-1)
            score = np.abs(x)
            elapsed = time.perf_counter() - start
            if not np.all(np.isfinite(score)):
                raise FloatingPointError(f"non-finite {name} benchmark output")
            rho = float(spearmanr(score, np.abs(error)).statistic)
            row = {"method": name, "repeat": repeat, "time_s": elapsed, "spearman": rho}
            if name in ("adaptive_rank90", "adaptive_flux90"):
                result = adaptive_results["flux" if name == "adaptive_flux90" else "poisson"]
                row.update({"iterations": result.iterations, "target_met": int(result.target_met),
                            "bound_checks": result.bound_poisson_solves,
                            "certified_high_recall": result.certified_high_recall})
            rows.append(row)
    return rows, {"amg_setup_plus_first_cycle_s": amg_setup_and_first,
                  "direct_factorization_s": direct_factor_s,
                  "amg_n_levels": len(hierarchy.levels),
                  "amg_coarsest_unknowns": hierarchy.levels[-1].A.shape[0]}


def summarize(frame, bootstrap, seed):
    out = {}
    for (n, kappa), group in frame.groupby(["N", "kappa"]):
        by_field = group.groupby(["sample", "method"], sort=False).time_s.median().unstack()
        draws = np.random.default_rng(seed).integers(0, len(by_field), (bootstrap, len(by_field)))
        methods = {}
        for method in by_field.columns:
            values = by_field[method].to_numpy()
            ratios = values / by_field["oracle"].to_numpy()
            low, high = np.quantile(np.median(ratios[draws], axis=1), [0.025, 0.975])
            methods[method] = {
                "median_time_ms": float(1000.0 * np.median(values)),
                "mean_time_ms": float(1000.0 * values.mean()),
                "median_paired_ratio_to_fresh_oracle": float(np.median(ratios)),
                "ratio_ci95_low": float(low), "ratio_ci95_high": float(high),
                "mean_spearman": float(group[group.method == method].spearman.mean()),
            }
            if method in ("adaptive_rank90", "adaptive_flux90"):
                block = group[group.method == method]
                methods[method].update({"mean_iterations": float(block.iterations.mean()),
                                       "target_met_fraction": float(block.target_met.mean()),
                                       "mean_bound_checks": float(block.bound_checks.mean())})
        flux_ratio = by_field["adaptive_flux90"].to_numpy() / by_field["adaptive_rank90"].to_numpy()
        flux_difference = 1000.0 * (by_field["adaptive_flux90"] - by_field["adaptive_rank90"]).to_numpy()
        ratio_low, ratio_high = np.quantile(np.median(flux_ratio[draws], axis=1), [0.025, 0.975])
        difference_low, difference_high = np.quantile(flux_difference[draws].mean(axis=1), [0.025, 0.975])
        out[f"N{n}_k{kappa:g}"] = {
            "N": int(n), "kappa": float(kappa), "n_fields": len(by_field), "methods": methods,
            "adaptive_flux_vs_poisson": {
                "median_paired_time_ratio": float(np.median(flux_ratio)),
                "ratio_ci95_low": float(ratio_low), "ratio_ci95_high": float(ratio_high),
                "mean_paired_time_difference_ms": float(flux_difference.mean()),
                "difference_ci95_low_ms": float(difference_low), "difference_ci95_high_ms": float(difference_high),
            },
        }
    return out


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "benchmark.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite completed benchmark: {output_path}")
    rng = np.random.default_rng(args.seed)
    frames, setups, provenance = [], [], []
    load_start = os.getloadavg()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    source_paths = [Path(__file__).resolve(), REPO / "solver/adaptive.py",
                    REPO / "solver/assemble_fd.py", REPO / "solver/relaxation.py",
                    REPO / "solver/poisson.py", REPO / "solver/multigrid.py",
                    REPO / "uncertainty/rank_bounds.py", REPO / "uncertainty/flux_bounds.py"]
    source_hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    for path in source_paths:
        snapshot = args.out / "source_snapshot" / path.relative_to(REPO)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    for directory in args.cache_dirs:
        for kappa in args.kappas:
            cache = directory / f"predictions_k{kappa:g}.npz"
            with np.load(cache, allow_pickle=False) as data:
                a, f, u = (data[name] for name in ("a", "f", "u"))
                prediction = data["members"].astype(np.float64).mean(axis=0)
                signature = str(data["signature"].item())
            if a.shape[0] < args.samples:
                raise ValueError(f"{cache} has fewer than {args.samples} fields")
            n = a.shape[1]
            provenance.append({"cache": str(cache), "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                               "cache_signature": json.loads(signature)})
            # Discard one complete field trial to warm imported numerical kernels.
            benchmark_field(a[0], prediction[0], f[0], u[0], rng, 1, args.seed, args.budgets)
            for i in range(args.samples):
                rows, setup = benchmark_field(a[i], prediction[i], f[i], u[i], rng,
                                               args.repeats, args.seed + i, args.budgets)
                frames.append(pd.DataFrame(rows).assign(N=n, kappa=kappa, sample=i))
                setups.append({"N": n, "kappa": kappa, "sample": i, **setup})
            pd.concat(frames, ignore_index=True).to_csv(args.out / "raw_times.csv", index=False)
            print(f"timed N={n}, kappa={kappa:g}, {args.samples} fields x {args.repeats} repeats", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    payload = {
        "schema_version": 2,
        "protocol": {"scope": "assembled A and neural prediction already available; residual included for every method",
                     "fresh_amg": "fresh hierarchy and lazy coarse solve per field and repeat; no benchmark instrumentation",
                     "fresh_oracle": "CSC conversion, sparse factorization, and solve per field and repeat",
                     "reused_rows": "same-A amortized scenario only; setup excluded, not the fresh-field experiment",
                     "constant_poisson": "DST-II; grid eigenvalues cached once per N and reused across coefficient fields",
                     "adaptive_rank90": "declared defaults q=.1, recall target=.9, check every5, maximum80; includes all bound checks; Green diagonal reusable per grid",
                     "adaptive_flux90": "same stopping protocol with equilibrated-flux majorant; includes its extra coefficient-weighted face sums",
                     "field_selection": "first n_fields in original IID held-out order; no outcome selection",
                     "repeats": args.repeats, "method_order": "seeded random permutation per field and repeat",
                     "aggregation": "median repeats per field, then median paired field ratio to fresh oracle",
                     "bootstrap_unit": "coefficient field, conditional on fixed machine and trained models",
                     "bootstrap_replicates": args.bootstrap, "seed": args.seed,
                     "thread_limit_requested": 1,
                     "thread_environment": {name: os.environ[name] for name in
                                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                     "runtime_warmup": "one discarded field per N/kappa; fresh-A setup remains timed"},
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                     "pyamg": pyamg.__version__, "platform": platform.platform(), "machine": platform.machine(),
                     "numpy_blas": np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}),
                     "detected_thread_pools": threadpool_info()},
        "load_average_start": load_start, "load_average_end": os.getloadavg(),
        "process_cpu_s": time.process_time() - cpu_start,
        "benchmark_wall_s": time.perf_counter() - wall_start,
        "provenance": provenance, "results": summarize(frame, args.bootstrap, args.seed),
        "reused_setup_per_field": setups,
        "source_sha256": source_hashes,
    }
    for name, digest in source_hashes.items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"benchmark source changed during evaluation: {name}")
    for record in provenance:
        if hashlib.sha256(Path(record["cache"]).read_bytes()).hexdigest() != record["cache_sha256"]:
            raise RuntimeError(f"prediction cache changed during benchmark: {record['cache']}")
    payload["input_and_source_hashes_unchanged"] = True
    payload["average_cpu_cores_used"] = payload["process_cpu_s"] / payload["benchmark_wall_s"]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("PASS: finite solver outputs; fresh/reused timing scopes recorded separately", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--kappas", nargs="+", type=float, default=[10, 100, 500])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.repeats, args.bootstrap, *args.budgets) < 1:
        parser.error("samples, repeats, bootstrap, and budgets must be positive")
    # Same guard as the production AMG implementation; output acceptance is
    # independently checked and every benchmark score must be finite.
    with threadpool_limits(limits=1), warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        run(args)


if __name__ == "__main__":
    main()
