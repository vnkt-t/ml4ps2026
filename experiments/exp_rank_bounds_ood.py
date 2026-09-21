"""Deterministic Poisson-PCG error/rank bounds on frozen OOD FNO predictions.

No fitting, tuning, exchangeability, or target calibration is used. The bound is
an existing exact-arithmetic discrete inequality, evaluated in float64 and
independently checked against saved reference solutions. This implementation
is not an interval-arithmetic proof of floating-point containment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.linalg import cg

from analysis.metrics import localization_metrics_top10
from solver.assemble_fd import assemble
from solver.poisson import poisson_preconditioner
from uncertainty.rank_bounds import magnitude_envelope, poisson_error_bound, separate_top_set

DATASETS = ["test_id", "test_cs", "test_r", "test_b"]
NUMERIC_RELATIVE_TOLERANCE = 1e-10
NUMERIC_ABSOLUTE_TOLERANCE = 1e-13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _validate_field(a, forcing, prediction, truth):
    values = [np.asarray(value, dtype=np.float64) for value in [a, forcing, prediction, truth]]
    shape = values[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
        raise ValueError("coefficient must be a nonempty square field")
    if any(value.shape != shape or not np.isfinite(value).all() for value in values):
        raise ValueError("all fields must have the same shape and finite entries")
    if np.min(values[0]) <= 0.0:
        raise ValueError("coefficients must be strictly positive")
    return values


def evaluate_field(a, forcing, prediction, truth, *, budgets=(20, 40), top_fraction=0.1):
    """Evaluate scores and bounds; truth is used exclusively for diagnostics.

    Residual correction, coefficient lower bound, magnitude enclosures, and rank
    flags are computed before any truth-based acceptance comparison. No floating
    tolerance is added to the analytic bound or used to construct rank flags.
    """
    a, forcing, prediction, truth = _validate_field(a, forcing, prediction, truth)
    if top_fraction != 0.1:
        raise ValueError("this experiment reports top-10% diagnostics; top_fraction must be 0.1")
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(isinstance(k, bool) or int(k) != k or k < 1 for k in budgets)):
        raise ValueError("budgets must be nonempty, unique, positive integers")
    N = a.shape[0]
    A = assemble(a)
    residual = A @ prediction.reshape(-1) - forcing.reshape(-1)
    preconditioner = poisson_preconditioner(N)
    a_min, a_max = float(a.min()), float(a.max())
    corrections = []
    for budget in budgets:
        iterations = [0]
        def count_iteration(_):
            iterations[0] += 1
        z, info = cg(A, residual, M=preconditioner, x0=np.zeros_like(residual),
                     rtol=8 * np.finfo(np.float64).eps, atol=0.0,
                     maxiter=int(budget), callback=count_iteration)
        if info < 0 or not np.isfinite(z).all():
            raise RuntimeError(f"Poisson-PCG{budget} failed: info={info}")
        defect = (residual - A @ z).reshape(N, N)
        correction = z.reshape(N, N)
        bound = poisson_error_bound(defect, a_min)
        lower, upper = magnitude_envelope(correction, bound)
        certain_high, certain_low = separate_top_set(lower, upper, top_fraction)
        corrections.append((budget, iterations[0], info, correction, defect, bound, lower, upper, certain_high, certain_low))

    # Evaluation-only reference data enters after every certificate is formed.
    error = prediction - truth
    error_norm, truth_norm = float(np.linalg.norm(error)), float(np.linalg.norm(truth))
    if error_norm == 0.0 or truth_norm == 0.0:
        raise ValueError("relative error/ranking diagnostics require nonzero truth and prediction error")
    absolute_error = np.abs(error)
    reference_scale = max(float(absolute_error.max()), 1e-12)
    numeric_tolerance = NUMERIC_RELATIVE_TOLERANCE * reference_scale + NUMERIC_ABSOLUTE_TOLERANCE
    residual_norm = float(np.linalg.norm(residual))
    identity = float(np.linalg.norm(A @ error.reshape(-1) - residual) / max(residual_norm, 1e-15))
    if identity > 1e-8:
        raise AssertionError(f"saved reference solution fails same-A residual identity: {identity}")
    n_top = int(np.ceil(top_fraction * error.size))
    truth_top = np.zeros(error.size, dtype=bool)
    truth_top[np.argpartition(absolute_error.reshape(-1), -n_top)[-n_top:]] = True
    common = {
        "a_min": a_min, "a_max": a_max, "actual_coefficient_contrast": a_max / a_min,
        "original_prediction_relative_l2": error_norm / truth_norm,
        "same_A_identity_relative_error": identity,
    }
    rows = []
    for method, score in [("raw_residual", np.abs(residual).reshape(N, N)),
                          ("prediction_magnitude", np.abs(prediction))]:
        rows.append({**common, "method": method, **localization_metrics_top10(score, absolute_error)})
    for budget, iterations, info, correction, defect, bound, lower, upper, certain_high, certain_low in corrections:
        difference = error - correction
        signed_excess = np.abs(difference) - bound
        false_high = int(np.sum(certain_high.reshape(-1) & ~truth_top))
        false_low = int(np.sum(certain_low.reshape(-1) & truth_top))
        violations = int(np.sum(signed_excess > numeric_tolerance))
        magnitude_violations = int(np.sum((absolute_error < lower - numeric_tolerance) |
                                          (absolute_error > upper + numeric_tolerance)))
        if np.any(certain_high & certain_low):
            raise AssertionError("rank flags cannot be both high and low")
        if false_high or false_low or violations or magnitude_violations:
            raise AssertionError(f"Poisson-PCG{budget}: false_high={false_high}, false_low={false_low}, "
                                 f"signed_bound_violations={violations}, magnitude_violations={magnitude_violations}")
        rows.append({
            **common, "method": f"poisson_pcg{budget}", "iteration_budget": int(budget),
            "iterations_executed": int(iterations), "cg_info": int(info),
            **localization_metrics_top10(np.abs(correction), absolute_error),
            "corrected_prediction_relative_l2": float(np.linalg.norm(prediction - correction - truth) / truth_norm),
            "relative_l2_correction_error": float(np.linalg.norm(difference) / error_norm),
            "relative_defect_l2": float(np.linalg.norm(defect) / max(residual_norm, 1e-15)),
            "resolved_fraction": float(np.mean(certain_high | certain_low)),
            "all_cells_resolved": bool(np.all(certain_high | certain_low)),
            "certified_top10_recall": float(certain_high.sum() / n_top),
            "certified_high_count": int(certain_high.sum()), "certified_low_count": int(certain_low.sum()),
            "mean_bound_over_mean_abs_error": float(bound.mean() / absolute_error.mean()),
            "numeric_tolerance": numeric_tolerance,
            "max_positive_bound_excess": float(max(0.0, signed_excess.max())),
            "max_positive_bound_excess_over_error_scale": float(max(0.0, signed_excess.max()) / reference_scale),
            "raw_float64_bound_exceedance_count": int(np.sum(signed_excess > 0.0)),
            "bound_violations_beyond_numeric_tolerance": violations,
            "magnitude_violations_beyond_numeric_tolerance": magnitude_violations,
            "false_high_count": false_high, "false_low_count": false_low,
        })
    return rows


def mean_ci(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("field bootstrap requires a nonempty finite vector")
    quantiles = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_low": float(quantiles[0]), "ci95_high": float(quantiles[1])}


def summarize(frame, *, n_fields, bootstrap, seed):
    draws = np.random.default_rng(seed).integers(0, n_fields, size=(bootstrap, n_fields))
    metrics = ["spearman", "auroc", "error_capture_top10", "top10_overlap",
               "original_prediction_relative_l2", "corrected_prediction_relative_l2",
               "relative_l2_correction_error", "relative_defect_l2", "resolved_fraction",
               "all_cells_resolved", "certified_top10_recall", "mean_bound_over_mean_abs_error"]
    methods = {}
    for name, block in frame.groupby("method", sort=False):
        if block["sample"].tolist() != list(range(n_fields)):
            raise ValueError("each method must contain the same ordered sample indices")
        methods[name] = {metric: mean_ci(block[metric].to_numpy(), draws)
                         for metric in metrics if metric in block and block[metric].notna().all()}
        if name.startswith("poisson_pcg"):
            for key in ["false_high_count", "false_low_count", "bound_violations_beyond_numeric_tolerance",
                        "magnitude_violations_beyond_numeric_tolerance", "raw_float64_bound_exceedance_count"]:
                methods[name][f"{key}_total"] = int(block[key].sum())
            methods[name]["max_positive_bound_excess_over_error_scale"] = float(block.max_positive_bound_excess_over_error_scale.max())
            methods[name]["executed_iterations_range"] = [int(block.iterations_executed.min()), int(block.iterations_executed.max())]
    wide = frame.pivot(index="sample", columns="method", values="spearman")
    paired = {}
    for method in methods:
        if method.startswith("poisson_pcg"):
            for baseline in ["raw_residual", "prediction_magnitude"]:
                paired[f"{method}_minus_{baseline}"] = mean_ci((wide[method] - wide[baseline]).to_numpy(), draws)
    return {"methods": methods, "paired_spearman_differences": paired,
            "bootstrap_replicates": bootstrap, "bootstrap_seed": seed}


def run(args):
    start = time.perf_counter()
    bundle = args.input.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    initial_hash = sha256(bundle)
    source_files = ["experiments/exp_rank_bounds_ood.py", "solver/assemble_fd.py", "solver/poisson.py",
                    "uncertainty/rank_bounds.py", "analysis/metrics.py"]
    report = {
        "experiment": "ood_discrete_error_and_rank_bounds", "input_bundle": str(bundle),
        "input_bundle_sha256": initial_hash,
        "source_sha256": {name: sha256(REPO / name) for name in source_files},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "torch_metadata_only": version("torch")},
        "protocol": {
            "bound": "sqrt(diag(B^-1) * d.T @ B^-1 @ d) / a_min; B=A(1), d=r-Az",
            "source_of_bound": "classical energy Cauchy-Schwarz and inverse Loewner ordering; no novel theorem claimed",
            "a_min": "actual minimum of each observed coefficient field, not nominal contrast metadata",
            "correction": "zero-start Poisson-preconditioned CG at fixed iteration budgets",
            "corrected_prediction": "u_hat - z; its relative L2 error uses ||u_true|| in the denominator",
            "no_fitting_or_calibration": True,
            "truth_usage": "reference errors enter only evaluation, after enclosures and rank flags are formed",
            "top_fraction": args.top_fraction,
            "rank_ties": "strict separation leaves ambiguous ties unresolved; truth top set selects ceil(q*n) cells",
            "numeric_scope": "exact-arithmetic discrete inequality evaluated in float64; no formal floating-point enclosure",
            "numeric_tolerance": "1e-10 * max(max(abs(u_hat-u_true)),1e-12) + 1e-13",
            "numeric_tolerance_use": "evaluation only; never added to analytic bounds or used to create flags",
            "resampling_unit": "whole sampled coefficient fields; repeated checkerboards are repeated draws from a finite catalogue",
            "ci_scope": "conditional on frozen trained networks and the declared sampled-field distribution",
            "no_runtime_or_continuum_claim": True,
        },
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "per_dataset": {},
    }
    with np.load(bundle, allow_pickle=False) as data:
        report["bundle_metadata"] = {key: np.asarray(data[key]).tolist() for key in ["grid_N", "kappa", "alpha"] if key in data}
        for offset, dataset in enumerate(DATASETS):
            fields = {key: np.asarray(data[f"{dataset}_{key}"], dtype=np.float64)
                      for key in ["a", "f", "u_hat_mean", "u_true"]}
            available = fields["a"].shape[0]
            n_fields = available if args.samples is None else args.samples
            if not 2 <= n_fields <= available:
                raise ValueError(f"requested {n_fields} fields but {dataset} has {available}")
            fields = {key: value[:n_fields] for key, value in fields.items()}
            fingerprints = [array_sha256(field) for field in fields["a"]]
            catalogue = Counter(fingerprints)
            rows = []
            for index in range(n_fields):
                try:
                    records = evaluate_field(fields["a"][index], fields["f"][index], fields["u_hat_mean"][index],
                                             fields["u_true"][index], budgets=args.budgets, top_fraction=args.top_fraction)
                except Exception as exc:
                    raise RuntimeError(f"{dataset} field {index} failed validation") from exc
                rows.extend({"dataset": dataset, "sample": index, "coefficient_sha256": fingerprints[index], **record}
                            for record in records)
                if (index + 1) % 50 == 0 or index + 1 == n_fields:
                    print(f"OOD rank bounds {dataset}: {index + 1}/{n_fields}", flush=True)
            frame = pd.DataFrame(rows)
            field_csv = out / f"fields_{dataset}.csv"
            frame.to_csv(field_csv, index=False)
            summary = summarize(frame, n_fields=n_fields, bootstrap=args.bootstrap, seed=args.bootstrap_seed + offset)
            minima = fields["a"].min(axis=(1, 2))
            maxima = fields["a"].max(axis=(1, 2))
            report["per_dataset"][dataset] = {
                "n_sampled_fields": n_fields, "n_unique_coefficient_fields": len(catalogue),
                "coefficient_catalogue_counts": dict(sorted(catalogue.items())),
                "a_min_range": [float(minima.min()), float(minima.max())],
                "a_max_range": [float(maxima.min()), float(maxima.max())],
                "actual_contrast_range": [float((maxima/minima).min()), float((maxima/minima).max())],
                "grid_N": int(fields["a"].shape[1]),
                "forcing_sha256": array_sha256(fields["f"]),
                "prediction_sha256": array_sha256(fields["u_hat_mean"]),
                "truth_sha256": array_sha256(fields["u_true"]),
                "per_field_csv": str(field_csv), "per_field_csv_sha256": sha256(field_csv),
                "same_A_identity_relative_error_max": float(frame.same_A_identity_relative_error.max()),
                **summary,
            }
            write_json(out / "summary.json", report)
            print(dataset + ": " + ", ".join(f"{method} rho={stats['spearman']['mean']:.4f}" for method, stats in summary["methods"].items()), flush=True)
    if sha256(bundle) != initial_hash:
        raise RuntimeError("frozen OOD bundle changed during evaluation")
    for name, digest in report["source_sha256"].items():
        if sha256(REPO / name) != digest:
            raise RuntimeError(f"source file changed during evaluation: {name}; rerun for consistent provenance")
    report["input_and_source_hashes_unchanged"] = True
    report["wall_time_s"] = time.perf_counter() - start
    write_json(out / "summary.json", report)
    print("PASS: no false rank flags or bound violations beyond the declared float64 evaluation tolerance", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "results/ood_features_bundle.npz")
    parser.add_argument("--out", type=Path, default=REPO / "results/rank_bounds_ood")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.bootstrap < 1 or any(budget < 1 for budget in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        parser.error("bootstrap and unique iteration budgets must be positive")
    if args.top_fraction != 0.1:
        parser.error("this experiment reports top-10% diagnostics; top-fraction must be 0.1")
    return args


if __name__ == "__main__":
    run(parse_args())
