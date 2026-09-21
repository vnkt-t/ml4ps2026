"""Matched FNO training and frozen evaluation on bounded smooth coefficients.

See SMOOTH_ROBUSTNESS_PROTOCOL.md for the fixed design and inference limits.
Existing binary checkpoints and evidence are never changed by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.linalg import LinearOperator, cg, spsolve
from scipy.stats import spearmanr
import torch

from analysis.metrics import localization_metrics_top10
from data.generate_smooth import smooth_sample
from experiments.exp0_pilot_fno import build_inputs
from experiments.exp_flux_bounds import evaluate_field as evaluate_bounds
from experiments.exp_submission_validation import mean_ci, correction_metrics
from experiments.exp_unet_robustness import fit_normalization
from models.tiny_fno import FNO2d
from solver.assemble_fd import assemble
from solver.diagnostics import cell_energy
from solver.multigrid import vcycle, last_vcycle_backend, last_vcycle_cost
from solver.poisson import poisson_preconditioner

PROTOCOL = REPO / "experiments/SMOOTH_ROBUSTNESS_PROTOCOL.md"
SOURCE_NAMES = [
    "experiments/exp_smooth_robustness.py", "experiments/SMOOTH_ROBUSTNESS_PROTOCOL.md",
    "data/generate_smooth.py", "data/generate_contrast.py", "data/generate_grf.py",
    "experiments/exp0_pilot_fno.py", "experiments/exp_flux_bounds.py",
    "experiments/exp_submission_validation.py", "experiments/exp_unet_robustness.py",
    "experiments/exp_real_fno_multi_kappa.py", "models/tiny_fno.py",
    "solver/assemble_fd.py", "solver/solve.py", "solver/diagnostics.py",
    "solver/poisson.py", "solver/multigrid.py", "solver/relaxation.py",
    "uncertainty/rank_bounds.py", "uncertainty/flux_bounds.py", "analysis/metrics.py",
]
BUDGETS = (5, 10, 20, 40, 80)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def dataset(count, seed, args):
    rng = np.random.default_rng(seed)
    draws = [smooth_sample(N=args.N, kappa=args.kappa, rng=rng,
                           lengthscale=args.lengthscale) for _ in range(count)]
    return {key: np.stack([row[j] for row in draws]) for j, key in enumerate(("a", "f", "u"))}


def predict(model, coefficients, norm, batch_size):
    inputs = build_inputs(coefficients, norm["logmean"], norm["logstd"], coefficients.shape[1])
    model.eval()
    blocks = []
    with torch.inference_mode():
        for offset in range(0, len(inputs), batch_size):
            normalized = model(torch.from_numpy(inputs[offset:offset+batch_size])).numpy()
            blocks.append((normalized * norm["ustd"] + norm["umean"]).astype(np.float32))
    result = np.concatenate(blocks)
    if result.shape != coefficients.shape or not np.isfinite(result).all():
        raise FloatingPointError("invalid frozen FNO prediction")
    return result


def train(seed, training, norm, args, epochs=None):
    torch.manual_seed(seed)
    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.layers)
    inputs = torch.from_numpy(build_inputs(training["a"], norm["logmean"], norm["logstd"], args.N))
    targets = torch.from_numpy(((training["u"]-norm["umean"])/norm["ustd"]).astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(args.epochs if epochs is None else epochs):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(inputs))
        total = 0.0
        for offset in range(0, len(inputs), args.batch_size):
            indices = order[offset:offset+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs[indices]), targets[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at seed{seed} epoch{epoch+1}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        scheduler.step()
        history.append({"epoch": epoch+1, "training_mse": total/len(inputs),
                        "wall_time_s": time.perf_counter()-started})
        if epoch < 3 or (epoch+1) % 10 == 0 or epoch+1 == args.epochs:
            print(f"seed={seed} epoch={epoch+1}/{args.epochs} training_mse={history[-1]['training_mse']:.6f} seconds={history[-1]['wall_time_s']:.2f}", flush=True)
    return model, history


def evaluate(arrays, out, args):
    a, f, truth, members = [arrays[name] for name in ("a", "f", "u", "members")]
    prediction = members.astype(np.float64).mean(axis=0)
    sigma = members.astype(np.float64).std(axis=0)
    distance = np.minimum(np.arange(args.N)+.5, args.N-.5-np.arange(args.N))/args.N
    boundary = np.minimum.outer(distance, distance).reshape(-1)
    poisson = poisson_preconditioner(args.N)
    rows, field_rows, bound_rows, member_rows, costs = [], [], [], [], []
    for i in range(len(a)):
        A = assemble(a[i])
        residual = A @ prediction[i].reshape(-1) - f[i].reshape(-1)
        diagonal = A.diagonal()
        diagonal_inverse = LinearOperator(A.shape, matvec=lambda x: x/diagonal, dtype=np.float64)
        corrections, executed = {}, {}
        specifications = [(f"poisson_pcg{k}", k, poisson) for k in BUDGETS]
        specifications += [("cg20", 20, None), ("diagonal_pcg20", 20, diagonal_inverse)]
        for name, budget, preconditioner in specifications:
            counter = [0]
            def callback(_):
                counter[0] += 1
            z, info = cg(A, residual, M=preconditioner, x0=np.zeros_like(residual),
                         rtol=8*np.finfo(float).eps, atol=0., maxiter=budget, callback=callback)
            if info < 0 or not np.isfinite(z).all():
                raise FloatingPointError(f"{name} failed on field{i}")
            corrections[name] = z
            executed[name] = counter[0]
        np.random.seed(20260915+i)
        corrections["amg"] = np.asarray(vcycle(A, residual)).reshape(-1)
        if last_vcycle_backend() != "pyamg_smoothed_aggregation":
            raise RuntimeError("full PyAMG hierarchy required")
        costs.append({"sample": i, **last_vcycle_cost()})
        scores = {"raw": np.abs(residual), "prediction_magnitude": np.abs(prediction[i]).reshape(-1),
                  "boundary_distance": boundary, "ensemble": sigma[i].reshape(-1),
                  **{name: np.abs(z) for name, z in corrections.items()}}
        # All scores are fixed before reference errors enter evaluation.
        error = (prediction[i]-truth[i]).reshape(-1)
        error_norm = np.linalg.norm(error)
        identity = float(np.linalg.norm(spsolve(A, residual)-error)/error_norm)
        energy = cell_energy(A, error)
        total_energy = float(error @ (A @ error))
        energy_check = float(abs(energy.sum()-total_energy)/total_energy)
        if identity > 1e-8 or energy_check > 1e-10:
            raise AssertionError(f"operator/energy consistency failure at field{i}")
        field_rows.append({"sample": i, "relative_l2_prediction_error": float(error_norm/np.linalg.norm(truth[i])),
                           "identity_relative_error": identity, "energy_relative_error": energy_check,
                           "raw_energy_spearman": float(spearmanr(np.abs(residual), energy).statistic),
                           "a_min": float(a[i].min()), "a_max": float(a[i].max()),
                           "unique_coefficients": int(np.unique(a[i]).size),
                           "coefficient_sha256": array_digest(a[i])})
        for name, score in scores.items():
            row = {"sample": i, "method": name, **localization_metrics_top10(score, np.abs(error))}
            if name in corrections:
                row.update(correction_metrics(A, corrections[name], error))
                row["iterations_executed"] = executed.get(name)
            rows.append(row)
        for j, seed in enumerate(args.seeds):
            member = members[j, i].astype(np.float64)
            member_error = (member-truth[i]).reshape(-1)
            member_residual = A @ member.reshape(-1)-f[i].reshape(-1)
            member_rows.append({"sample": i, "seed": seed,
                                "relative_l2_prediction_error": float(np.linalg.norm(member_error)/np.linalg.norm(truth[i])),
                                "raw_spearman": float(spearmanr(np.abs(member_residual), np.abs(member_error)).statistic)})
        bound_rows.extend({"sample": i, **row} for row in evaluate_bounds(a[i], f[i], prediction[i], truth[i], [20, 40]))
        if (i+1) % 25 == 0 or i+1 == len(a):
            print(f"smooth evaluation {i+1}/{len(a)}", flush=True)
    frame, fields, bounds, seeds = [pd.DataFrame(records) for records in (rows, field_rows, bound_rows, member_rows)]
    for name, data in [("metrics", frame), ("fields", fields), ("bounds", bounds), ("members", seeds)]:
        data.to_csv(out/f"{name}.csv", index=False)
    draws = np.random.default_rng(args.bootstrap_seed).integers(0, len(a), size=(args.bootstrap, len(a)))
    methods = {}
    for name, block in frame.groupby("method", sort=False):
        metrics = ["spearman", "auroc", "top10_overlap", "error_capture_top10"]
        if name in corrections:
            metrics += ["relative_l2_correction_error", "relative_A_correction_error"]
        methods[name] = {metric: mean_ci(block[metric].to_numpy(), draws) for metric in metrics}
        if name in executed:
            values = block.iterations_executed.to_numpy()
            methods[name]["iterations_executed"] = {"min": int(values.min()), "max": int(values.max()), "mean": float(values.mean())}
    wide = frame.pivot(index="sample", columns="method", values="spearman")
    bound_summary = {}
    for (budget, kind), block in bounds.groupby(["iterations", "bound"]):
        summary = {metric: mean_ci(block[metric].to_numpy(), draws) for metric in
                   ["certified_top10_recall", "resolved_fraction", "relative_l2_correction_error"]}
        summary.update({f"{metric}_total": int(block[metric].sum()) for metric in
                        ["false_high", "false_low", "bound_violations", "raw_positive_bound_exceedances"]})
        bound_summary.setdefault(str(budget), {})[kind] = summary
    return {"n_fields": len(a), "methods": methods,
            "relative_l2_prediction_error": mean_ci(fields.relative_l2_prediction_error.to_numpy(), draws),
            "raw_energy_spearman": mean_ci(fields.raw_energy_spearman.to_numpy(), draws),
            "paired_spearman_differences": {f"{name}_minus_raw": mean_ci((wide[name]-wide.raw).to_numpy(), draws)
                                            for name in ["poisson_pcg20", "prediction_magnitude", "boundary_distance", "ensemble", "amg"]},
            "bounds": bound_summary, "per_seed": {str(seed): {key: mean_ci(block[key].to_numpy(), draws)
                 for key in ["raw_spearman", "relative_l2_prediction_error"]} for seed, block in seeds.groupby("seed")},
            "identity_relative_error_max": float(fields.identity_relative_error.max()),
            "energy_relative_error_max": float(fields.energy_relative_error.max()), "amg_cost_per_field": costs}


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"use a new output directory; refusing to overwrite {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {"experiment": "matched_smooth_fno_robustness", "created_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": configuration, "protocol_sha256": digest(PROTOCOL),
              "source_sha256": {name: digest(REPO/name) for name in SOURCE_NAMES},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)}}
    for name in SOURCE_NAMES:
        target = args.out/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)
    write_json(args.out/"run_started.json", report)
    training = dataset(args.n_train, args.train_seed, args)
    norm = fit_normalization(training["a"], training["u"])
    report["normalization"] = norm
    report["training_array_sha256"] = {name: array_digest(array) for name, array in training.items()}
    np.savez_compressed(args.out/"training.npz", **training)
    if args.pilot_epochs:
        _, history = train(args.seeds[0], training, norm, args, args.pilot_epochs)
        report.update({"pilot_only_no_test_data_generated": True, "training_history": history,
                       "projected_training_seconds": float(np.mean([row["wall_time_s"] for row in history[1:] or history])*args.epochs*len(args.seeds))})
        write_json(args.out/"pilot.json", report)
        print(f"Training-only pilot projects {report['projected_training_seconds']/60:.1f} minutes for all seeds", flush=True)
        return
    checkpoints = []
    for seed in args.seeds:
        model, history = train(seed, training, norm, args)
        meta = {"seed": seed, "arch": {"modes": args.modes, "width": args.width, "n_layers": args.layers},
                "normalization": norm, "training_data_seed": args.train_seed, "epochs": args.epochs,
                "training_count": args.n_train, "source_sha256": report["source_sha256"],
                "training_array_sha256": report["training_array_sha256"], "protocol_sha256": report["protocol_sha256"]}
        path = args.out/f"fno_seed{seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "metadata": meta}, path)
        restored = FNO2d(**meta["arch"])
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
        if not np.array_equal(predict(model, training["a"][:2], norm, args.batch_size),
                              predict(restored, training["a"][:2], norm, args.batch_size)):
            raise AssertionError("checkpoint roundtrip changed predictions")
        checkpoints.append({"seed": seed, "file": path.name, "sha256": digest(path), "history": history,
                            "checkpoint_roundtrip_exact": True})
        write_json(args.out/"training_progress.json", {"checkpoints": checkpoints, "normalization": norm})
    report["checkpoints"] = checkpoints
    report["all_checkpoints_frozen_before_test_generation_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out/"training_complete.json", report)
    testing = dataset(args.n_test, args.test_seed, args)
    train_hashes = {array_digest(field) for field in training["a"]}
    test_hashes = {array_digest(field) for field in testing["a"]}
    if train_hashes & test_hashes or len(train_hashes) != args.n_train or len(test_hashes) != args.n_test:
        raise AssertionError("duplicate coefficient fields within or across splits")
    predictions = []
    for checkpoint in checkpoints:
        payload = torch.load(args.out/checkpoint["file"], map_location="cpu", weights_only=True)
        model = FNO2d(**payload["metadata"]["arch"])
        model.load_state_dict(payload["model_state_dict"])
        predictions.append(predict(model, testing["a"], norm, args.batch_size))
    testing["members"] = np.stack(predictions)
    report["test_array_sha256"] = {name: array_digest(array) for name, array in testing.items()}
    report["train_test_disjoint_by_coefficient_hash"] = True
    np.savez_compressed(args.out/"predictions_k100.npz", **testing,
                        signature=json.dumps({"configuration": configuration, "checkpoints": [row["sha256"] for row in checkpoints],
                                              "protocol_sha256": report["protocol_sha256"]}, sort_keys=True))
    report["evaluation"] = evaluate(testing, args.out, args)
    for name, expected in report["source_sha256"].items():
        if digest(REPO/name) != expected:
            raise AssertionError(f"source changed during run: {name}")
    report["source_hashes_unchanged"] = True
    report["wall_time_s"] = time.perf_counter()-started
    report["output_sha256"] = {p.name: digest(p) for p in args.out.iterdir() if p.is_file()}
    write_json(args.out/"summary.json", report)
    print("PASS: matched smooth-family training, fresh evaluation, identity, energy and enclosure checks", flush=True)
    print({name: round(row["spearman"]["mean"], 5) for name, row in report["evaluation"]["methods"].items()}, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO/"results/smooth_robustness")
    parser.add_argument("--pilot-epochs", type=int, default=0)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=100.)
    parser.add_argument("--lengthscale", type=float, default=.12)
    parser.add_argument("--n-train", type=int, default=700)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=20260913)
    parser.add_argument("--test-seed", type=int, default=20260914)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    args = parser.parse_args()
    if min(args.N, args.n_train, args.n_test, args.width, args.layers, args.epochs, args.batch_size, args.bootstrap) < 2:
        parser.error("positive sizes >=2 required")
    if not 1 <= args.modes <= args.N//2 or args.train_seed == args.test_seed or len(set(args.seeds)) != len(args.seeds):
        parser.error("invalid Fourier modes or non-distinct seeds")
    if args.pilot_epochs < 0 or args.pilot_epochs > args.epochs or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("invalid pilot duration or learning rate")
    args.out = args.out.resolve()
    return args


if __name__ == "__main__":
    main(parse_args())
