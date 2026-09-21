"""Independently verify a completed smooth-family experiment, without training.

All retained fields and checkpoints are checked, with no numerical subsampling.
Generation is replayed using the generator; FNO input construction, metrics,
bootstrap summaries, face assembly, cell energies and rank decisions are
implemented here without calling the experiment evaluator. SciPy CG and the
project Poisson preconditioner/AMG are shared numerical primitives; an independent
sparse unit-Poisson factorization checks the preconditioner and builds majorants.
The receipt counts executed assertions, not array elements or inferred checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, splu, spsolve
from scipy.stats import spearmanr
import torch

from data.generate_smooth import smooth_sample
from models.tiny_fno import FNO2d
from solver.assemble_fd import assemble
from solver.multigrid import vcycle, last_vcycle_backend, last_vcycle_cost
from solver.poisson import poisson_preconditioner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def path_label(path):
    """Prefer portable repository-relative provenance labels."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.failures = []

    def require(self, condition, category, label):
        self.counts[category] += 1
        if not bool(condition):
            self.failures.append({"category": category, "check": label})
            raise AssertionError(f"{category}: {label}")

    def close(self, actual, expected, category, label, rtol=2e-9, atol=2e-12):
        self.require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False),
                     category, f"{label}: actual={actual!r}, expected={expected!r}")


def face_system(a):
    """Build A=D diag(w) D.T by visiting each face exactly once."""
    n, cells = a.shape[0], a.size
    rr, cc, vv, wa, wb, owners = [], [], [], [], [], []
    for i in range(n):
        for j in range(n):
            left = i * n + j
            for ni, nj in ((i + 1, j), (i, j + 1)):
                if ni < n and nj < n:
                    right = ni * n + nj
                    face = len(wa)
                    rr.extend((left, right)); cc.extend((face, face)); vv.extend((1., -1.))
                    wa.append(n * n * (2. / (1. / a[i, j] + 1. / a[ni, nj])))
                    wb.append(float(n * n)); owners.append((left, right))
            for wall in (i == 0, i == n - 1, j == 0, j == n - 1):
                if wall:
                    rr.append(left); cc.append(len(wa)); vv.append(1.)
                    wa.append(2. * n * n * a[i, j]); wb.append(2. * n * n)
                    owners.append((left,))
    incidence = sp.csr_matrix((vv, (rr, cc)), shape=(cells, len(wa)))
    wa, wb = np.array(wa), np.array(wb)
    matrix = (incidence @ sp.diags(wa) @ incidence.T).tocsr()
    return matrix, incidence, wa, wb, owners


def independent_energy(error, incidence, weights, owners):
    face_energy = weights * np.square(incidence.T @ error)
    cells = np.zeros(error.size)
    for value, adjacent in zip(face_energy, owners):
        for cell in adjacent:
            cells[cell] += value / len(adjacent)
    return cells


def inverse_diagonal(factor, size, block=32):
    result = np.empty(size)
    for start in range(0, size, block):
        columns = np.arange(start, min(start + block, size))
        rhs = np.zeros((size, len(columns)))
        rhs[columns, np.arange(len(columns))] = 1.
        solved = factor.solve(rhs)
        result[columns] = solved[columns, np.arange(len(columns))]
    return result


def top_weights(values):
    """Explicit sorted-cutoff weights, averaging any cutoff ties."""
    values = np.asarray(values).ravel()
    count = int(np.ceil(.1 * len(values)))
    cutoff = np.sort(values)[-count]
    above, equal = values > cutoff, values == cutoff
    return above.astype(float) + equal * ((count - np.count_nonzero(above)) / np.count_nonzero(equal))


def independent_metrics(score, error):
    score, error = np.asarray(score).ravel(), np.abs(error).ravel()
    chosen_score, chosen_error = top_weights(score), top_weights(error)
    count = int(np.ceil(.1 * len(error)))
    rho = 0. if np.ptp(score) == 0 or np.ptp(error) == 0 else float(spearmanr(score, error).statistic)
    # Direct positive-negative comparisons avoid the production rank-sum AUROC.
    # In these continuous-error fields the true top set has no cutoff ties;
    # the general expected-membership formula also handles ties correctly.
    order = np.argsort(score, kind="stable")
    ordered_score, positive = score[order], chosen_error[order]
    negatives_seen, numerator = 0., 0.
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_score)) + 1, len(score)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        positives = positive[start:end].sum()
        negatives = (end - start) - positives
        numerator += positives * (negatives_seen + .5 * negatives)
        negatives_seen += negatives
    auc = numerator / (count * (len(error) - count))
    return {"spearman": rho, "auroc": float(auc),
            "top10_overlap": float(np.sum(chosen_score * chosen_error) / count),
            "error_capture_top10": float(np.sum(chosen_score * error) / error.sum())}


def independent_flags(lower, upper):
    """Count strict comparisons against other cells, avoiding order statistics."""
    lower, upper = lower.ravel(), upper.ravel()
    count = int(np.ceil(.1 * len(lower)))
    high = np.empty(len(lower), dtype=bool)
    low = np.empty(len(lower), dtype=bool)
    for start in range(0, len(lower), 64):
        stop = start + 64
        high[start:stop] = (lower[start:stop, None] > upper[None, :]).sum(axis=1) >= len(lower) - count
        low[start:stop] = (upper[start:stop, None] < lower[None, :]).sum(axis=1) >= count
    return high, low


def check_sources(directory, summary, checks):
    names = summary["source_sha256"]
    checks.require(len(names) > 0, "provenance", "nonempty source manifest")
    for name, expected in names.items():
        checks.require(digest(REPO / name) == expected, "source_hash", name)
        checks.require(digest(directory / "source_snapshot" / name) == expected, "source_snapshot", name)
    snapshots = {str(p.relative_to(directory / "source_snapshot"))
                 for p in (directory / "source_snapshot").rglob("*") if p.is_file()}
    checks.require(snapshots == set(names), "provenance", "snapshot inventory matches source manifest")
    checks.require(summary["protocol_sha256"] == names["experiments/SMOOTH_ROBUSTNESS_PROTOCOL.md"],
                   "provenance", "protocol hash matches source manifest")
    checks.require(summary["source_hashes_unchanged"] is True, "provenance", "runner source stability recorded")
    expected_outputs = summary["output_sha256"]
    required = {"training.npz", "predictions_k100.npz", "metrics.csv", "fields.csv", "bounds.csv", "members.csv",
                "run_started.json", "training_progress.json", "training_complete.json"}
    required |= {row["file"] for row in summary["checkpoints"]}
    checks.require(required <= set(expected_outputs), "provenance", "complete required output inventory")
    for name, expected in expected_outputs.items():
        checks.require(digest(directory / name) == expected, "output_hash", name)
    for filename in ("run_started.json", "training_complete.json"):
        record = json.loads((directory / filename).read_text())
        for key in ("source_sha256", "protocol_sha256", "configuration"):
            checks.require(record[key] == summary[key], "provenance", f"{filename}:{key}")
        checks.require("test_array_sha256" not in record and "evaluation" not in record,
                       "provenance", f"{filename}: no test artifacts before frozen checkpoint record")
    complete = json.loads((directory / "training_complete.json").read_text())
    checks.require(complete["checkpoints"] == summary["checkpoints"], "provenance", "frozen checkpoint list unchanged")
    for key in ("normalization", "training_array_sha256"):
        checks.require(complete[key] == summary[key], "provenance", f"frozen training record:{key}")
    progress = json.loads((directory / "training_progress.json").read_text())
    for key in ("normalization", "checkpoints"):
        checks.require(progress[key] == summary[key], "provenance", f"final training progress:{key}")


def check_datasets(directory, summary, checks):
    config = summary["configuration"]
    loaded = {}
    split_hashes = {}
    for split, file, count_name, seed_name, manifest in (
        ("training", "training.npz", "n_train", "train_seed", "training_array_sha256"),
        ("test", "predictions_k100.npz", "n_test", "test_seed", "test_array_sha256"),
    ):
        with np.load(directory / file, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files if name != "signature"}
            if split == "test":
                expected_signature = {"configuration": config,
                                      "checkpoints": [x["sha256"] for x in summary["checkpoints"]],
                                      "protocol_sha256": summary["protocol_sha256"]}
                checks.require(json.loads(str(source["signature"].item())) == expected_signature,
                               "provenance", "prediction signature")
        checks.require(set(arrays) == set(summary[manifest]), "arrays", f"{split} array inventory")
        for name, array in arrays.items():
            checks.require(array_digest(array) == summary[manifest][name], "array_hash", f"{split}:{name}")
            checks.require(np.isfinite(array).all(), "arrays", f"{split}:{name} finite")
        shape = (config[count_name], config["N"], config["N"])
        for name in ("a", "f", "u"):
            checks.require(arrays[name].shape == shape, "arrays", f"{split}:{name} expected shape")
        rng = np.random.default_rng(config[seed_name])
        for i in range(config[count_name]):
            regenerated = smooth_sample(N=config["N"], kappa=config["kappa"], rng=rng,
                                        lengthscale=config["lengthscale"])
            for name, value in zip(("a", "f", "u"), regenerated):
                checks.require(np.array_equal(value, arrays[name][i]), "regeneration", f"{split}:{i}:{name}")
        split_hashes[split] = {array_digest(field) for field in arrays["a"]}
        checks.require(len(split_hashes[split]) == config[count_name], "split_separation", f"{split} unique fields")
        loaded[split] = arrays
        print(f"verified all {config[count_name]} {split} draws", flush=True)
    checks.require(not split_hashes["training"] & split_hashes["test"], "split_separation", "train/test disjoint coefficients")
    checks.require(config["train_seed"] != config["test_seed"], "split_separation", "separate generation seeds")
    checks.require(summary["train_test_disjoint_by_coefficient_hash"] is True, "split_separation", "reported separation")
    train = loaded["training"]
    norm = {"logmean": float(np.log(train["a"]).mean()), "logstd": float(np.log(train["a"]).std()),
            "umean": float(train["u"].mean()), "ustd": float(train["u"].std())}
    checks.require(norm == summary["normalization"], "normalization", "training-only reconstruction exact")
    return loaded["test"], norm


def check_predictions(directory, summary, arrays, norm, checks):
    config = summary["configuration"]
    checks.require([x["seed"] for x in summary["checkpoints"]] == config["seeds"], "checkpoint", "seed ordering")
    checks.require(arrays["members"].shape == (len(config["seeds"]), *arrays["a"].shape), "arrays", "prediction member dimensions")
    inputs = np.empty((*arrays["a"].shape, 3), dtype=np.float32)
    inputs[..., 0] = (np.log(arrays["a"]) - norm["logmean"]) / norm["logstd"]
    centers = (np.arange(config["N"]) + .5) / config["N"]
    inputs[..., 1] = centers[None, :, None]
    inputs[..., 2] = centers[None, None, :]
    for member_index, checkpoint in enumerate(summary["checkpoints"]):
        path = directory / checkpoint["file"]
        checks.require(digest(path) == checkpoint["sha256"], "checkpoint_hash", path.name)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected_meta = {"seed": checkpoint["seed"], "arch": {"modes": config["modes"], "width": config["width"], "n_layers": config["layers"]},
                         "normalization": norm, "training_data_seed": config["train_seed"], "epochs": config["epochs"],
                         "training_count": config["n_train"], "source_sha256": summary["source_sha256"],
                         "training_array_sha256": summary["training_array_sha256"], "protocol_sha256": summary["protocol_sha256"]}
        checks.require(payload["metadata"] == expected_meta, "checkpoint", f"{path.name} metadata")
        history = checkpoint["history"]
        checks.require([row["epoch"] for row in history] == list(range(1, config["epochs"] + 1)),
                       "checkpoint", f"{path.name} full fixed-epoch history")
        checks.require(all(np.isfinite(row["training_mse"]) and row["training_mse"] >= 0 and row["wall_time_s"] > 0 for row in history),
                       "checkpoint", f"{path.name} finite history")
        model = FNO2d(**expected_meta["arch"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(inputs), config["batch_size"]):
                stop = start + config["batch_size"]
                output = model(torch.from_numpy(inputs[start:stop])).cpu().numpy()
                physical = (output * norm["ustd"] + norm["umean"]).astype(np.float32)
                checks.require(np.array_equal(physical, arrays["members"][member_index, start:stop]),
                               "checkpoint_predictions", f"seed{checkpoint['seed']}: test fields {start}:{min(stop, len(inputs))}")
    print("independent input/normalization pipeline reproduced every checkpoint prediction exactly", flush=True)


def bootstrap_summary(values, draws):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be finite scalar values")
    means = np.mean(values[draws], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci95_low": float(low), "ci95_high": float(high)}


def check_summaries(directory, summary, checks):
    tables = {name: pd.read_csv(directory / f"{name}.csv", float_precision="round_trip")
              for name in ("fields", "metrics", "bounds", "members")}
    config, result = summary["configuration"], summary["evaluation"]
    n = config["n_test"]
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, n, size=(config["bootstrap"], n))

    def check_ci(values, stored, label):
        computed = bootstrap_summary(values, draws)
        checks.require(set(stored) == set(computed), "bootstrap", f"{label}: summary keys")
        for key in computed:
            checks.close(computed[key], stored[key], "bootstrap", f"{label}:{key}", rtol=1e-13, atol=1e-14)

    def unique_complete(block, label):
        checks.require(sorted(block["sample"].tolist()) == list(range(n)), "csv_inventory", label)
        return block.sort_values("sample")

    fields = unique_complete(tables["fields"], "one field row per test sample")
    for metric in ("relative_l2_prediction_error", "raw_energy_spearman"):
        check_ci(fields[metric], result[metric], metric)
    for metric in ("identity_relative_error", "energy_relative_error"):
        checks.close(fields[metric].max(), result[metric + "_max"], "summary", metric + "_max", rtol=1e-13, atol=1e-18)
    checks.require(result["n_fields"] == n, "summary", "field count")
    checks.require(len(result["amg_cost_per_field"]) == n, "summary", "one AMG cost record per field")
    checks.require(set(tables["metrics"].method) == set(result["methods"]), "csv_inventory", "method inventory")
    for method, stored in result["methods"].items():
        block = unique_complete(tables["metrics"][tables["metrics"].method == method], f"{method} complete samples")
        for metric, values in stored.items():
            if metric == "iterations_executed":
                actual = block[metric].to_numpy()
                checks.require(values == {"min": int(actual.min()), "max": int(actual.max()), "mean": float(actual.mean())},
                               "summary", f"{method} iteration count summary")
            else:
                check_ci(block[metric], values, f"{method}:{metric}")
    wide = tables["metrics"].pivot(index="sample", columns="method", values="spearman").sort_index()
    for name, stored in result["paired_spearman_differences"].items():
        method = name.removesuffix("_minus_raw")
        check_ci(wide[method] - wide["raw"], stored, name)
    expected_bound_groups = {(int(budget), kind) for budget, kinds in result["bounds"].items() for kind in kinds}
    observed_groups = set(zip(tables["bounds"].iterations, tables["bounds"].bound))
    checks.require(expected_bound_groups == observed_groups, "csv_inventory", "bound group inventory")
    for budget, kinds in result["bounds"].items():
        for kind, stored in kinds.items():
            block = unique_complete(tables["bounds"][(tables["bounds"].iterations == int(budget)) & (tables["bounds"].bound == kind)],
                                    f"{kind}:{budget} complete samples")
            for metric, values in stored.items():
                if metric.endswith("_total"):
                    checks.require(int(block[metric.removesuffix("_total")].sum()) == values, "summary", f"{kind}:{budget}:{metric}")
                else:
                    check_ci(block[metric], values, f"{kind}:{budget}:{metric}")
    checks.require(set(result["per_seed"]) == {str(x) for x in config["seeds"]}, "csv_inventory", "per-seed summary inventory")
    checks.require(set(tables["members"].seed) == set(config["seeds"]), "csv_inventory", "per-seed CSV inventory")
    for seed, stored in result["per_seed"].items():
        block = unique_complete(tables["members"][tables["members"].seed == int(seed)], f"seed{seed} complete samples")
        for metric, values in stored.items():
            check_ci(block[metric], values, f"seed{seed}:{metric}")
    return tables


def check_numerics(arrays, summary, tables, checks):
    config = summary["configuration"]
    n = config["N"]
    fields = tables["fields"].set_index("sample")
    metrics = tables["metrics"].set_index(["sample", "method"])
    bounds_table = tables["bounds"].set_index(["sample", "iterations", "bound"])
    members_table = tables["members"].set_index(["sample", "seed"])
    unit, incidence, _, unit_weights, owners = face_system(np.ones((n, n)))
    factor = splu(unit.tocsc())
    inverse_diag = inverse_diagonal(factor, n * n)
    poisson = poisson_preconditioner(n)
    probe = np.random.default_rng(3191).normal(size=n * n)
    checks.close(poisson @ probe, factor.solve(probe), "poisson_independence", "DST preconditioner versus sparse unit operator", rtol=1e-11, atol=1e-14)
    checks.require(np.all(inverse_diag > 0), "poisson_independence", "positive independently solved inverse diagonal")
    prediction = np.mean(arrays["members"].astype(float), axis=0)
    sigma = np.std(arrays["members"].astype(float), axis=0)
    ij = np.indices((n, n))
    boundary = np.minimum.reduce((ij[0] + .5, ij[1] + .5, n - .5 - ij[0], n - .5 - ij[1])).ravel() / n
    correction_names = [name for name in summary["evaluation"]["methods"] if name.startswith("poisson_pcg")]
    budgets = sorted({int(name.removeprefix("poisson_pcg")) for name in correction_names})
    bound_budgets = sorted(int(value) for value in summary["evaluation"]["bounds"])
    numerical_extrema = {"direct_identity_max": 0., "independent_truth_residual_max": 0., "energy_sum_relative_error_max": 0.,
                         "producer_reported_energy_sum_relative_error_max": float(fields["energy_relative_error"].max())}
    for sample, a in enumerate(arrays["a"]):
        independent_A, D, weights, w_unit, face_owners = face_system(a)
        A = assemble(a)
        difference = independent_A - A
        checks.require(not difference.nnz or np.max(np.abs(difference.data)) <= 2e-14 * np.max(np.abs(A.data)),
                       "independent_assembly", f"field{sample}: all matrix entries")
        truth = arrays["u"][sample].ravel()
        forcing = arrays["f"][sample].ravel()
        pred = prediction[sample].ravel()
        error = pred - truth
        residual = A @ pred - forcing
        error_norm, truth_norm = np.linalg.norm(error), np.linalg.norm(truth)
        identity = np.linalg.norm(spsolve(A, residual) - error) / error_norm
        truth_residual = np.linalg.norm(independent_A @ truth - forcing) / np.linalg.norm(forcing)
        checks.require(identity < 1e-8, "identity", f"field{sample}: direct inverse residual/error identity")
        checks.require(truth_residual < 1e-10, "identity", f"field{sample}: independently assembled reference equation")
        energy = independent_energy(error, D, weights, face_owners)
        # Explicit scalar contractions avoid stale macOS BLAS floating-status
        # warnings without suppressing non-finite-value checks.
        total_energy = np.sum(error * (A @ error))
        energy_error = abs(energy.sum() - total_energy) / total_energy
        checks.require(np.all(np.isfinite(energy)) and np.all(energy >= 0) and np.isfinite(total_energy) and energy_error < 1e-10,
                       "energy", f"field{sample}: finite positive face allocation and stable quadratic sum")
        numerical_extrema["direct_identity_max"] = max(numerical_extrema["direct_identity_max"], float(identity))
        numerical_extrema["independent_truth_residual_max"] = max(numerical_extrema["independent_truth_residual_max"], float(truth_residual))
        numerical_extrema["energy_sum_relative_error_max"] = max(numerical_extrema["energy_sum_relative_error_max"], float(energy_error))
        field_expected = {"relative_l2_prediction_error": error_norm / truth_norm, "identity_relative_error": identity,
                          "raw_energy_spearman": float(spearmanr(np.abs(residual), energy).statistic),
                          "a_min": a.min(), "a_max": a.max(), "unique_coefficients": np.unique(a).size}
        for key, expected in field_expected.items():
            checks.close(fields.loc[sample, key], expected, "field_csv", f"field{sample}:{key}")
        checks.require(fields.loc[sample, "energy_relative_error"] < 1e-10, "field_csv", f"field{sample}: stored energy check tolerance")
        checks.require(fields.loc[sample, "coefficient_sha256"] == array_digest(a), "field_csv", f"field{sample}: coefficient hash")
        diagonal = A.diagonal()
        M_diag = LinearOperator(A.shape, matvec=lambda x: x / diagonal, dtype=float)
        corrections, executed, statuses = {}, {}, {}
        specs = [(f"poisson_pcg{k}", k, poisson) for k in sorted(set(budgets + bound_budgets))]
        specs += [("cg20", 20, None), ("diagonal_pcg20", 20, M_diag)]
        for name, budget, preconditioner in specs:
            counter = [0]
            def callback(_):
                counter[0] += 1
            z, info = cg(A, residual, M=preconditioner, x0=np.zeros_like(residual),
                         rtol=8*np.finfo(float).eps, atol=0., maxiter=budget, callback=callback)
            checks.require(info >= 0 and np.isfinite(z).all(), "correction", f"field{sample}:{name} converged or valid budget exit")
            corrections[name], executed[name], statuses[name] = z, counter[0], info
        np.random.seed(20260915 + sample)
        corrections["amg"] = np.asarray(vcycle(A, residual)).ravel()
        checks.require(last_vcycle_backend() == "pyamg_smoothed_aggregation", "amg", f"field{sample}: full AMG backend")
        measured_cost = summary["evaluation"]["amg_cost_per_field"][sample]
        replayed_cost = last_vcycle_cost()
        checks.require(measured_cost["sample"] == sample, "amg", f"field{sample}: cost row ordering")
        for key in ("schema_version", "cycle_complexity", "cycle_complexity_kind", "operator_complexity", "n_levels", "coarsest_unknowns"):
            checks.require(measured_cost[key] == replayed_cost[key], "amg", f"field{sample}:{key} structural replay")
        for key, value in measured_cost.items():
            if key.endswith("_s") or key.endswith("_matvec_equiv"):
                checks.require(np.isfinite(value) and value >= 0., "amg_timing_arithmetic", f"field{sample}:{key} finite nonnegative")
        checks.close(measured_cost["total_s"], measured_cost["setup_s"] + measured_cost["first_cycle_s"] + measured_cost["validation_s"],
                     "amg_timing_arithmetic", f"field{sample}: recorded total includes setup, first cycle and validation")
        for prefix in ("setup", "first_cycle", "cycle", "total"):
            checks.close(measured_cost[prefix + "_matvec_equiv"], measured_cost[prefix + "_s"] / measured_cost["matvec_s"],
                         "amg_timing_arithmetic", f"field{sample}:{prefix} recorded timing ratio")
        scores = {"raw": np.abs(residual), "prediction_magnitude": np.abs(pred), "boundary_distance": boundary,
                  "ensemble": sigma[sample].ravel(), **{name: np.abs(z) for name, z in corrections.items()}}
        for name in summary["evaluation"]["methods"]:
            expected = independent_metrics(scores[name], error)
            if name in corrections:
                delta = error - corrections[name]
                expected.update({"relative_l2_correction_error": np.linalg.norm(delta) / error_norm,
                                 "relative_A_correction_error": np.sqrt(max(0., np.sum(delta * (A @ delta))) / total_energy)})
            if name in executed:
                expected["iterations_executed"] = executed[name]
            for key, value in expected.items():
                checks.close(metrics.loc[(sample, name), key], value, "metric_csv", f"field{sample}:{name}:{key}")
        for member, seed in enumerate(config["seeds"]):
            member_prediction = arrays["members"][member, sample].astype(float).ravel()
            member_error = member_prediction - truth
            member_residual = A @ member_prediction - forcing
            expected = {"raw_spearman": float(spearmanr(np.abs(member_residual), np.abs(member_error)).statistic),
                        "relative_l2_prediction_error": np.linalg.norm(member_error) / truth_norm}
            for key, value in expected.items():
                checks.close(members_table.loc[(sample, seed), key], value, "member_csv", f"field{sample}:seed{seed}:{key}")
        magnitude = np.abs(error)
        top_count = int(np.ceil(.1 * a.size))
        true_top = np.zeros(a.size, dtype=bool)
        true_top[np.argsort(magnitude)[-top_count:]] = True
        error_scale = max(float(magnitude.max()), 1e-12)
        tolerance = 1e-10 * error_scale + 1e-13
        for budget in bound_budgets:
            name = f"poisson_pcg{budget}"
            z = corrections[name]
            defect = residual - A @ z
            potential = factor.solve(defect)
            poisson_energy = np.sum(defect * potential)
            flux = w_unit * (D.T @ potential)
            equilibrated_energy = np.sum(np.square(flux) / weights)
            checks.close(D @ flux, defect, "flux_equilibrium", f"field{sample}:{budget}: divergence equals defect", rtol=1e-8, atol=1e-11)
            computed_bounds = {"baseline": np.sqrt(inverse_diag * poisson_energy) / a.min(),
                               "equilibrated_flux": np.sqrt(inverse_diag * equilibrated_energy / a.min())}
            flags = {}
            for kind, bound in computed_bounds.items():
                lower, upper = np.maximum(np.abs(z) - bound, 0.), np.abs(z) + bound
                high, low = independent_flags(lower, upper)
                flags[kind] = (high, low)
                excess = np.abs(error - z) - bound
                violations = int(np.count_nonzero(excess > tolerance))
                false_high = int(np.count_nonzero(high & ~true_top))
                false_low = int(np.count_nonzero(low & true_top))
                checks.require(violations == 0 and false_high == 0 and false_low == 0, "bound_validity", f"field{sample}:{budget}:{kind}")
                checks.require(not np.any(high & low), "rank_separation", f"field{sample}:{budget}:{kind}: mutually exclusive flags")
                expected = {"iterations_executed": executed[name], "cg_info": statuses[name], "a_min": a.min(),
                            "actual_contrast": a.max() / a.min(), "resolved_fraction": np.mean(high | low),
                            "all_cells_resolved": bool(np.all(high | low)), "certified_top10_recall": high.sum() / top_count,
                            "certified_high_count": high.sum(), "certified_low_count": low.sum(),
                            "mean_bound_over_mean_abs_error": bound.mean() / magnitude.mean(),
                            "flux_to_baseline_bound_ratio": np.sqrt(a.min() * equilibrated_energy / poisson_energy) if poisson_energy else 1.,
                            "corrected_prediction_relative_l2": np.linalg.norm(pred - z - truth) / truth_norm,
                            "relative_l2_correction_error": np.linalg.norm(error - z) / error_norm,
                            "same_A_identity_relative_error": np.linalg.norm(A @ error - residual) / max(np.linalg.norm(residual), 1e-15),
                            "numeric_tolerance": tolerance, "false_high": false_high, "false_low": false_low,
                            "bound_violations": violations, "raw_positive_bound_exceedances": np.count_nonzero(excess > 0),
                            "max_positive_bound_excess_over_error_scale": max(0., excess.max()) / error_scale,
                            "positive_flux_excess_over_baseline_bound": max(0., np.max(computed_bounds["equilibrated_flux"] - computed_bounds["baseline"]))}
                for key, value in expected.items():
                    checks.close(bounds_table.loc[(sample, budget, kind), key], value, "bound_csv", f"field{sample}:{budget}:{kind}:{key}")
            lost = np.count_nonzero((flags["baseline"][0] & ~flags["equilibrated_flux"][0]) |
                                    (flags["baseline"][1] & ~flags["equilibrated_flux"][1]))
            checks.require(lost == 0, "bound_ordering", f"field{sample}:{budget}: no baseline flags lost")
            checks.require(np.max(computed_bounds["equilibrated_flux"] - computed_bounds["baseline"]) <= tolerance,
                           "bound_ordering", f"field{sample}:{budget}: flux majorant no larger")
            for kind in computed_bounds:
                checks.require(bounds_table.loc[(sample, budget, kind), "baseline_flags_lost_by_flux"] == lost,
                               "bound_csv", f"field{sample}:{budget}:{kind}: lost flags")
        if (sample + 1) % 25 == 0 or sample + 1 == len(prediction):
            print(f"independent numerical checks {sample+1}/{len(prediction)}", flush=True)
    return numerical_extrema


def verify(directory, producer_log=None):
    directory = Path(directory).resolve()
    checks = Checks()
    started = time.perf_counter()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    config = summary["configuration"]
    source_paths = [Path(__file__).resolve()] + [REPO / name for name in summary["source_sha256"]]
    inputs = ([summary_path] + [directory / name for name in summary["output_sha256"]]
              + [directory / "source_snapshot" / name for name in summary["source_sha256"]])
    producer_warnings = {"producer_log_supplied": producer_log is not None}
    if producer_log is not None:
        producer_log = Path(producer_log).resolve()
        inputs.append(producer_log)
        log_lines = producer_log.read_text().splitlines()
        warnings = [line for line in log_lines if "RuntimeWarning:" in line]
        matmul_warnings = [line for line in warnings if "encountered in matmul" in line]
        producer_warnings.update({"log": path_label(producer_log), "log_sha256": digest(producer_log),
                                  "runtime_warning_lines": len(warnings), "matmul_floating_status_warning_lines": len(matmul_warnings),
                                  "warning_text": sorted({line.split("RuntimeWarning:", 1)[1].strip() for line in warnings}),
                                  "note": "The original producer log is retained, including its NumPy matmul floating-status warnings on macOS. The verifier uses explicit elementwise scalar sums and checks finiteness and agreement with independently allocated face energies; it does not suppress the producer warnings."})
    input_hashes = {path: digest(path) for path in inputs}
    source_hashes = {path: digest(path) for path in source_paths}
    portable_config = dict(config)
    portable_config["out"] = path_label(directory)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "directory": path_label(directory),
               "configuration": portable_config, "input_sha256": {path_label(p): h for p, h in input_hashes.items()},
               "verification_source_sha256": {path_label(p): h for p, h in source_hashes.items()},
               "original_producer_warnings": producer_warnings,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
               "scope": {"subsampling": "none", "training_fields": config["n_train"], "test_fields": config["n_test"],
                         "checkpoint_seeds": config["seeds"], "prediction_comparison": "bitwise exact using original batch size and independent input construction",
                         "bootstrap_replicates": config["bootstrap"], "bootstrap_seed": config["bootstrap_seed"],
                         "numeric_tolerances": "identity 1e-8; independent reference residual 1e-10; energy 1e-10; bound validity 1e-10*max(max(abs(error)),1e-12)+1e-13; CSV floats rtol=2e-9 atol=2e-12",
                         "independent_components": ["face-incidence TPFA assembly", "sparse unit-Poisson solve and inverse diagonal", "face cell-energy allocation", "model input/physical-output construction", "localization metrics", "bootstrap summaries", "flux majorants", "strict comparison-count rank decisions"],
                         "shared_components": ["coefficient generator and target solve for exact regeneration", "FNO architecture and saved weights", "SciPy CG and project Poisson preconditioner", "project AMG cycle with retained random seed"],
                         "not_verified": ["training optimization replay", "external timestamp attestation", "wall-clock measurement reproducibility (only retained timing arithmetic checked)", "continuum or outward-rounded certificates"],
                         "counting_unit": "one executed assertion or array comparison, not individual array elements"}}
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        checks.require(summary["experiment"] == "matched_smooth_fno_robustness", "configuration", "expected experiment")
        checks.require(not config["pilot_epochs"], "configuration", "completed experiment, not timing pilot")
        check_sources(directory, summary, checks)
        arrays, norm = check_datasets(directory, summary, checks)
        check_predictions(directory, summary, arrays, norm, checks)
        tables = check_summaries(directory, summary, checks)
        receipt["numerical_extrema"] = check_numerics(arrays, summary, tables, checks)
        receipt["energy_verification"] = {"all_test_fields_finite_and_consistent": True,
            "independent_relative_discrepancy_definition": "abs(sum(independently allocated face energies)-sum(error*(A@error)))/sum(error*(A@error)), with elementwise scalar summation",
            "maximum_independent_relative_discrepancy": receipt["numerical_extrema"]["energy_sum_relative_error_max"],
            "original_producer_reported_maximum_relative_discrepancy": receipt["numerical_extrema"]["producer_reported_energy_sum_relative_error_max"]}
        for path, expected in {**input_hashes, **source_hashes}.items():
            checks.require(digest(path) == expected, "stability", path_label(path))
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt.update({"executed_assertions": sum(checks.counts.values()), "checks_by_category": dict(sorted(checks.counts.items())),
                        "failed_assertions": checks.failures, "wall_time_s": time.perf_counter() - started})
        destination = directory / "verification.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(destination)
    print(f"PASS: {sum(checks.counts.values())} executed assertions; no fields subsampled; receipt {destination}", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=REPO / "results/smooth_robustness")
    parser.add_argument("--producer-log", type=Path, default=None,
                        help="Retain and hash the original producer log, including any floating-status warnings.")
    arguments = parser.parse_args()
    verify(arguments.directory, arguments.producer_log)
