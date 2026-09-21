"""Untuned submission checks on frozen FNO checkpoints.

Compare Jacobi, CG, diagonal-preconditioned CG and one AMG cycle; retain
per-field metrics for paired bootstrap inference. Unlike the old norm diagnostic,
all errors come from trained networks and energy is a nonnegative face allocation.
No training or model selection occurs here. Original result files are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import torch

from data.generate_contrast import family_C_field, family_C_sample
from experiments.exp_real_fno_multi_kappa import load_ensemble
from analysis.metrics import localization_metrics_top10
from solver.assemble_fd import assemble
from solver.diagnostics import cell_energy, cell_flux_magnitude
from solver.multigrid import vcycle, last_vcycle_backend, last_vcycle_cost
from solver.relaxation import jacobi_sweep, cg_correction
from solver.poisson import solve_poisson, poisson_preconditioner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def mean_ci(values, draws) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not values.size:
        raise ValueError("bootstrap requires a nonempty finite vector")
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def correction_metrics(A, correction, error) -> dict:
    delta = np.asarray(correction).reshape(-1) - error
    q_error = float(np.dot(error, A @ error))
    q_delta = float(np.dot(delta, A @ delta))
    if q_error <= 0 or q_delta < -1e-12 * q_error:
        raise ValueError("invalid SPD energy")
    return {
        "relative_l2_correction_error": float(np.linalg.norm(delta) / np.linalg.norm(error)),
        "relative_A_correction_error": float(np.sqrt(max(q_delta, 0.0) / q_error)),
    }


def dataset_and_predictions(args, kappa):
    ckpt_dir = args.checkpoint_dir or (REPO / ("results/checkpoints" if args.N == 32 else "results/checkpoints_64"))
    ckpt_dir = ckpt_dir.resolve()
    paths = [ckpt_dir / f"fno_kappa{kappa:g}_seed{s}.pt" for s in [0, 1, 2]]
    provenance = {
        "grid_N": args.N, "kappa": kappa, "n_test": args.samples,
        "test_seed": args.test_seed, "skip_fields": args.skip_fields,
        "checkpoint_training_seed": 0,
        "checkpoint_hashes": {str(p.relative_to(REPO)): sha256(p) for p in paths},
        "generator_sha256": sha256(REPO / "data/generate_contrast.py"),
        "solver_sha256": sha256(REPO / "solver/assemble_fd.py"),
        "model_sha256": sha256(REPO / "models/tiny_fno.py"),
    }
    cache_path = args.out / f"predictions_k{kappa:g}.npz"
    signature = json.dumps(provenance, sort_keys=True)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as d:
            if str(d["signature"].item()) != signature:
                raise ValueError(f"stale prediction cache: {cache_path}")
            return [d[k] for k in ("a", "f", "u", "members")], provenance
    rng = np.random.default_rng(args.test_seed)
    # Solving does not consume randomness, so field-only draws reproduce the old
    # 700-training-field offset without solving unused training targets again.
    for _ in range(args.skip_fields):
        family_C_field(N=args.N, kappa=kappa, rng=rng)
    samples = [family_C_sample(N=args.N, kappa=kappa, rng=rng) for _ in range(args.samples)]
    a, f, u = [np.stack([getattr(s, name) for s in samples]) for name in ("a", "f", "u_true")]
    ensemble = load_ensemble(kappa, seeds=[0, 1, 2], checkpoint_dir=ckpt_dir,
                             grid_N=args.N, n_train=700, n_test=200, data_seed=0)
    members = np.concatenate([
        ensemble.predict_members(a[start:start + args.batch_size])
        for start in range(0, args.samples, args.batch_size)
    ], axis=1)
    np.savez_compressed(cache_path, a=a, f=f, u=u, members=members, signature=signature)
    return [a, f, u, members], provenance


def run_kappa(args, kappa):
    start = time.perf_counter()
    (a, f, u, members), provenance = dataset_and_predictions(args, kappa)
    provenance["test_field_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
    u_hat = members.astype(np.float64).mean(axis=0)
    sigma = members.astype(np.float64).std(axis=0)
    rows, norm_rows, costs, identity, energy_identity, relative_l2 = [], [], [], [], [], []
    distance_1d = np.minimum(np.arange(args.N) + 0.5, args.N - 0.5 - np.arange(args.N)) / args.N
    boundary_distance = np.minimum.outer(distance_1d, distance_1d).reshape(-1)
    poisson_M = poisson_preconditioner(args.N)
    for i in range(args.samples):
        A = assemble(a[i])
        error = (u_hat[i] - u[i]).reshape(-1)
        rhs = A @ u_hat[i].reshape(-1) - f[i].reshape(-1)
        truth = spla.spsolve(A, rhs)
        identity.append(float(np.linalg.norm(truth - error) / np.linalg.norm(error)))
        relative_l2.append(float(np.linalg.norm(error) / np.linalg.norm(u[i])))
        if identity[-1] > 1e-8:
            raise AssertionError(f"same-A identity failed: {identity[-1]}")
        target = np.abs(error)
        energy = cell_energy(A, error)
        energy_identity.append(float(abs(energy.sum() - np.dot(error, A @ error)) /
                                     np.dot(error, A @ error)))
        norm_targets = {
            "abs_error": target,
            "face_energy": energy,
            "face_flux_magnitude": cell_flux_magnitude(a[i], error.reshape(args.N, args.N)),
        }
        for name, values in norm_targets.items():
            rho = float(spearmanr(np.abs(rhs), np.asarray(values).reshape(-1)).statistic)
            norm_rows.append({"sample": i, "target": name, "spearman": rho})
        raw = np.abs(rhs).reshape(args.N, args.N)
        scores = {"ensemble": sigma[i].reshape(-1), "raw": np.abs(rhs),
                  "smoothed": gaussian_filter(raw, 1.0, mode="nearest").reshape(-1),
                  "prediction_magnitude": np.abs(u_hat[i]).reshape(-1),
                  "boundary_distance": boundary_distance}
        corrections = {}
        for k in (1, 5, 10, 20, 40, 80):
            corrections[f"jacobi{k}"] = jacobi_sweep(A, rhs, np.zeros_like(rhs), k)
        for k in (5, 10, 20, 40, 80):
            for name, preconditioned in (("cg", False), ("pcg", True)):
                x, info = cg_correction(A, rhs, k, diagonal_preconditioner=preconditioned)
                corrections[f"{name}{k}"] = x
            x, info = spla.cg(A, rhs, x0=np.zeros_like(rhs), rtol=8 * np.finfo(float).eps,
                              atol=0.0, maxiter=k, M=poisson_M)
            if info < 0 or not np.isfinite(x).all():
                raise RuntimeError(f"Poisson-PCG{k} failed on field {i}")
            corrections[f"poisson_pcg{k}"] = x
        corrections["poisson"] = solve_poisson(rhs.reshape(args.N, args.N)).reshape(-1)
        # PyAMG estimates its smoother spectrum using an RNG. Fix it per field
        # to make results invariant to preceding experiment/control flow.
        np.random.seed(20260912 + i)
        corrections["amg"] = np.asarray(vcycle(A, rhs)).reshape(-1)
        backend = last_vcycle_backend()
        if backend != "pyamg_smoothed_aggregation":
            raise RuntimeError(f"AMG comparison requires actual AMG, got {backend}")
        costs.append({"sample": i, **last_vcycle_cost()})
        corrections["oracle"] = truth
        scores.update({name: np.abs(x) for name, x in corrections.items()})
        for name, score in scores.items():
            row = {"sample": i, "method": name, **localization_metrics_top10(score, target)}
            if name in corrections:
                row.update(correction_metrics(A, corrections[name], error))
            rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == args.samples:
            print(f"N={args.N} kappa={kappa:g}: {i + 1}/{args.samples} fields", flush=True)
    frame = pd.DataFrame(rows)
    norm_frame = pd.DataFrame(norm_rows)
    frame.to_csv(args.out / f"metrics_k{kappa:g}.csv", index=False)
    norm_frame.to_csv(args.out / f"norms_k{kappa:g}.csv", index=False)
    draws = np.random.default_rng(args.bootstrap_seed).integers(0, args.samples,
                                                              size=(args.bootstrap, args.samples))
    summary = {}
    for name, block in frame.groupby("method", sort=False):
        summary[name] = {metric: mean_ci(block[metric].to_numpy(), draws)
                         for metric in ("spearman", "auroc", "error_capture_top10", "top10_overlap")}
        if name in corrections:
            for metric in ("relative_l2_correction_error", "relative_A_correction_error"):
                summary[name][metric] = mean_ci(block[metric].to_numpy(), draws)
    wide = frame.pivot(index="sample", columns="method", values="spearman")
    pairs = (("ensemble", "raw"), ("amg", "raw"), ("amg", "jacobi20"),
             ("amg", "cg20"), ("amg", "pcg20"), ("pcg20", "cg20"),
             ("amg", "poisson_pcg20"), ("prediction_magnitude", "raw"),
             ("boundary_distance", "raw"), ("amg", "prediction_magnitude"))
    paired = {f"{left}_minus_{right}": mean_ci((wide[left] - wide[right]).to_numpy(), draws)
              for left, right in pairs}
    payload = {
        "provenance": provenance, "n_fields": args.samples,
        "relative_l2_prediction_error": mean_ci(relative_l2, draws), "methods": summary,
        "paired_spearman_differences": paired,
        "raw_residual_norm_targets": {name: mean_ci(block.spearman.to_numpy(), draws)
                                      for name, block in norm_frame.groupby("target", sort=False)},
        "identity_relative_error_max": max(identity),
        "energy_allocation_relative_error_max": max(energy_identity),
        "amg_cost_per_field": costs, "wall_time_s": time.perf_counter() - start,
    }
    write_json(args.out / f"summary_k{kappa:g}.json", payload)
    print("Spearman: " + ", ".join(f"{name}={summary[name]['spearman']['mean']:.4f}"
                                   for name in ("ensemble", "raw", "prediction_magnitude", "jacobi20", "cg20", "pcg20", "poisson", "poisson_pcg20", "amg")), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, choices=(32, 64), default=32)
    parser.add_argument("--kappas", type=float, nargs="+", default=[10, 100, 500])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--skip-fields", type=int, default=700)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.bootstrap < 1 or args.skip_fields < 0 or args.batch_size < 1:
        parser.error("samples >=2, bootstrap >=1, skip-fields >=0, batch-size >=1 required")
    if args.test_seed == 0 and args.skip_fields < 700:
        parser.error("seed 0 draws 0..699 trained the checkpoints; skip at least 700 fields")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "submission_validation", "protocol": {
        "predictor": "frozen three-member FNO ensemble, training seeds 0,1,2",
        "evaluation": "per-field spatial localization; no tuning or retraining",
        "resampling_unit": "independent coefficient fields; paired differences share fields",
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "energy": "nonnegative face contributions summing to e^T A e, including Dirichlet boundary",
        "pcg": "scipy CG, unpreconditioned / diagonal inverse / constant-coefficient Poisson inverse; zero start; 5/10/20/40/80 iterations",
        "poisson_inverse": "exact constant-coefficient TPFA inverse via orthonormal DST-II; coefficient=1; fixed per-grid preconditioner",
        "cheap_baselines": "ensemble standard deviation; prediction magnitude; distance to closest domain boundary; Gaussian-smoothed |r| sigma=1 cell",
        "timing": "AMG complexity is a rough structural estimate, not end-to-end work",
        "claim_scope": "conditional on these trained networks and this coefficient/source family",
    }, "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "per_kappa": {}}
    for kappa in args.kappas:
        manifest["per_kappa"][f"{kappa:g}"] = run_kappa(args, kappa)
        write_json(args.out / "summary.json", manifest)
    print(f"PASS: completed {len(args.kappas)} contrasts; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
