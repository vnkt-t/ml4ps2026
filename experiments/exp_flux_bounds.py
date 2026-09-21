"""Compare classical global-coefficient and equilibrated-flux error majorants.

Both bounds use the same one Poisson solve after a fixed Poisson-PCG correction.
No labels enter correction or bound construction. Reference errors are used only
for held-out validity/informativeness checks. Original artifacts are untouched.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.linalg import cg

from solver.assemble_fd import assemble
from solver.poisson import poisson_preconditioner
from uncertainty.flux_bounds import poisson_flux_bounds
from uncertainty.rank_bounds import magnitude_envelope, separate_top_set


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_hash(field):
    return hashlib.sha256(np.ascontiguousarray(field, dtype=np.float64).tobytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def mean_ci(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("mean CI requires nonempty finite values")
    quantiles = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return {"mean": float(values.mean()), "ci95_low": float(quantiles[0]), "ci95_high": float(quantiles[1])}


def evaluate_field(a, forcing, prediction, truth, budgets):
    arrays = [np.asarray(x, dtype=np.float64) for x in (a, forcing, prediction, truth)]
    a, forcing, prediction, truth = arrays
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0 or any(x.shape != a.shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError("fields must be matching finite nonempty square arrays")
    A = assemble(a)
    N = a.shape[0]
    rhs = A @ prediction.reshape(-1) - forcing.reshape(-1)
    M = poisson_preconditioner(N)
    computed = []
    for budget in budgets:
        count = [0]
        def callback(_):
            count[0] += 1
        z, info = cg(A, rhs, M=M, x0=np.zeros_like(rhs), rtol=8*np.finfo(float).eps,
                     atol=0.0, maxiter=budget, callback=callback)
        if info < 0 or not np.isfinite(z).all():
            raise RuntimeError(f"Poisson-PCG{budget} failed with info={info}")
        correction = z.reshape(N, N)
        defect = (rhs - A @ z).reshape(N, N)
        result = poisson_flux_bounds(defect, a)
        variants = {}
        for name, bound in [("baseline", result.baseline_bound), ("equilibrated_flux", result.bound)]:
            lower, upper = magnitude_envelope(correction, bound)
            high, low = separate_top_set(lower, upper, .1)
            variants[name] = (bound, high, low)
        computed.append((budget, count[0], info, correction, result, variants))
    # Truth enters only after all corrections, bounds, and decisions are fixed.
    error = prediction - truth
    error_norm, truth_norm = np.linalg.norm(error), np.linalg.norm(truth)
    if error_norm <= 0.0 or truth_norm <= 0.0:
        raise ValueError("relative diagnostics require nonzero error and truth")
    magnitude = np.abs(error)
    error_scale = max(float(magnitude.max()), 1e-12)
    tolerance = 1e-10 * error_scale + 1e-13
    identity = np.linalg.norm(A @ error.reshape(-1) - rhs) / max(np.linalg.norm(rhs), 1e-15)
    if identity > 1e-8:
        raise AssertionError(f"same-A identity failed: {identity}")
    top_count = int(np.ceil(.1 * error.size)); true_top = np.zeros(error.size, dtype=bool)
    true_top[np.argpartition(magnitude.reshape(-1), -top_count)[-top_count:]] = True
    rows = []
    for budget, iterations, info, correction, result, variants in computed:
        excess_new_over_old = float(np.max(result.bound - result.baseline_bound))
        if excess_new_over_old > tolerance:
            raise AssertionError("equilibrated-flux majorant exceeded the baseline beyond tolerance")
        base_high, base_low = variants["baseline"][1:]
        flux_high, flux_low = variants["equilibrated_flux"][1:]
        lost_flags = int(np.sum((base_high & ~flux_high) | (base_low & ~flux_low)))
        ratio = float(np.sqrt(result.a_min * result.equilibrated_energy / result.poisson_energy)) if result.poisson_energy else 1.0
        for name, (bound, high, low) in variants.items():
            excess = np.abs(error - correction) - bound
            false_high = int(np.sum(high.reshape(-1) & ~true_top))
            false_low = int(np.sum(low.reshape(-1) & true_top))
            violations = int(np.sum(excess > tolerance))
            if false_high or false_low or violations:
                raise AssertionError(f"{name} budget{budget}: high={false_high}, low={false_low}, violations={violations}")
            rows.append({
                "iterations": int(budget), "iterations_executed": int(iterations), "cg_info": int(info), "bound": name,
                "a_min": result.a_min, "actual_contrast": float(a.max()/a.min()),
                "resolved_fraction": float(np.mean(high | low)), "all_cells_resolved": bool(np.all(high | low)),
                "certified_top10_recall": float(high.sum()/top_count),
                "certified_high_count": int(high.sum()), "certified_low_count": int(low.sum()),
                "mean_bound_over_mean_abs_error": float(bound.mean()/magnitude.mean()),
                "flux_to_baseline_bound_ratio": ratio,
                "corrected_prediction_relative_l2": float(np.linalg.norm(prediction-correction-truth)/truth_norm),
                "relative_l2_correction_error": float(np.linalg.norm(error-correction)/error_norm),
                "same_A_identity_relative_error": float(identity), "numeric_tolerance": tolerance,
                "false_high": false_high, "false_low": false_low, "bound_violations": violations,
                "raw_positive_bound_exceedances": int(np.sum(excess > 0)),
                "max_positive_bound_excess_over_error_scale": float(max(0.0, excess.max())/error_scale),
                "positive_flux_excess_over_baseline_bound": max(0.0, excess_new_over_old),
                "baseline_flags_lost_by_flux": lost_flags,
            })
    return rows


def sources(args):
    for directory in args.cache_dirs or []:
        for kappa in args.kappas:
            path = directory.resolve() / f"predictions_k{kappa:g}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {"a": np.asarray(data["a"]), "f": np.asarray(data["f"]), "truth": np.asarray(data["u"]),
                          "prediction": np.asarray(data["members"], dtype=np.float64).mean(axis=0)}
                signature = str(data["signature"].item()) if "signature" in data else None
            yield f"{directory.name}_k{kappa:g}", path, arrays, {"cache_signature": signature, "nominal_kappa": float(kappa)}
    if args.ood_bundle:
        path = args.ood_bundle.resolve()
        with np.load(path, allow_pickle=False) as data:
            for dataset in ["test_id", "test_cs", "test_r", "test_b"]:
                arrays = {key: np.asarray(data[f"{dataset}_{source}"]) for key, source in
                          [("a", "a"), ("f", "f"), ("truth", "u_true"), ("prediction", "u_hat_mean")]}
                yield f"ood_{dataset}", path, arrays, {"nominal_kappa": float(data["kappa"]), "ood_dataset": dataset}


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    source_names = ["experiments/exp_flux_bounds.py", "uncertainty/flux_bounds.py", "uncertainty/rank_bounds.py",
                    "solver/poisson.py", "solver/assemble_fd.py"]
    report = {
        "experiment": "equilibrated_flux_majorant_comparison",
        "configuration": {k: [str(x) for x in v] if k == "cache_dirs" and v else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {name: sha256(REPO/name) for name in source_names},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "protocol": {
            "interpretation": "classical equilibrated-flux/Thomson majorant, not a novel theorem",
            "flux_bound": "sqrt(diag(B^-1) * E_eq / a_min), E_eq=sum(q_face^2/W_A_face), q=W_B D.T B^-1 d",
            "baseline_bound": "sqrt(diag(B^-1) * d.T B^-1 d)/a_min",
            "cost_relationship": "both bounds share one Poisson solve; flux variant adds coefficient-weighted face sums",
            "coefficient_minimum": "actual per-field minimum; no fitted or calibrated parameter",
            "rank_policy": "strict top-ceil(10%*N^2) separation; no true errors enter decisions",
            "numeric_tolerance": "1e-10*max(max(abs(error)),1e-12)+1e-13, evaluation only, not applied to bounds or flags",
            "numeric_scope": "exact-arithmetic discrete inequality evaluated in float64, not an outward-rounded certificate",
            "confidence_intervals": "5000 by default paired whole-field bootstraps, conditional on frozen predictor and sampled distribution",
        },
        "per_source": {},
    }
    started = time.perf_counter()
    input_hashes = {}
    for source_index, (name, path, fields, metadata) in enumerate(sources(args)):
        input_hashes.setdefault(str(path), sha256(path))
        available = len(fields["a"]); n = available if args.samples is None else args.samples
        if not 2 <= n <= available:
            raise ValueError(f"requested{n} fields from{available} in{path}")
        fields = {key: value[:n] for key, value in fields.items()}
        records = []
        for i in range(n):
            try:
                rows = evaluate_field(fields["a"][i], fields["f"][i], fields["prediction"][i], fields["truth"][i], args.budgets)
            except Exception as exc:
                raise RuntimeError(f"flux audit failed for{name} field{i}") from exc
            records.extend({"sample": i, **row} for row in rows)
        frame = pd.DataFrame(records)
        path_csv = args.out / f"fields_{name}.csv"; frame.to_csv(path_csv, index=False)
        draws = np.random.default_rng(args.bootstrap_seed+source_index).integers(0, n, size=(args.bootstrap, n))
        summary = {}
        metrics = ["resolved_fraction", "all_cells_resolved", "certified_top10_recall", "mean_bound_over_mean_abs_error",
                   "flux_to_baseline_bound_ratio", "corrected_prediction_relative_l2", "relative_l2_correction_error"]
        for budget in args.budgets:
            summary[str(budget)] = {}
            for kind in ["baseline", "equilibrated_flux"]:
                block = frame[(frame.iterations == budget) & (frame.bound == kind)]
                summary[str(budget)][kind] = {metric: mean_ci(block[metric].to_numpy(), draws) for metric in metrics}
                summary[str(budget)][kind].update({f"{metric}_total": int(block[metric].sum()) for metric in
                    ["false_high", "false_low", "bound_violations", "raw_positive_bound_exceedances", "baseline_flags_lost_by_flux"]})
                summary[str(budget)][kind]["max_positive_bound_excess_over_error_scale"] = float(block.max_positive_bound_excess_over_error_scale.max())
            summary[str(budget)]["paired_flux_minus_baseline"] = {}
            for metric in ["resolved_fraction", "certified_top10_recall", "all_cells_resolved"]:
                wide = frame[frame.iterations == budget].pivot(index="sample", columns="bound", values=metric).astype(float)
                summary[str(budget)]["paired_flux_minus_baseline"][metric] = mean_ci((wide.equilibrated_flux-wide.baseline).to_numpy(), draws)
        report["per_source"][name] = {
            "input": str(path), "input_sha256": input_hashes[str(path)], "n_fields": n, "grid_N": int(fields["a"].shape[1]),
            "unique_coefficient_fields": len({field_hash(field) for field in fields["a"]}),
            "predictions_sha256": field_hash(fields["prediction"]), "coefficients_sha256": field_hash(fields["a"]),
            "truth_sha256": field_hash(fields["truth"]), "forcing_sha256": field_hash(fields["f"]),
            "same_A_identity_relative_error_max": float(frame.same_A_identity_relative_error.max()),
            "fields_csv": str(path_csv), "fields_csv_sha256": sha256(path_csv),
            "metadata": metadata, "budgets": summary,
        }
        write_json(args.out / "summary.json", report)
        print(name + ": " + ", ".join(f"k={budget} certified={values['baseline']['certified_top10_recall']['mean']:.4f}→{values['equilibrated_flux']['certified_top10_recall']['mean']:.4f}, "
                                        f"resolved={values['baseline']['resolved_fraction']['mean']:.4f}→{values['equilibrated_flux']['resolved_fraction']['mean']:.4f}"
                                        for budget, values in summary.items()), flush=True)
    for name, digest in report["source_sha256"].items():
        if sha256(REPO/name) != digest: raise RuntimeError(f"source changed during run:{name}")
    for path, digest in input_hashes.items():
        if sha256(path) != digest: raise RuntimeError(f"input changed during run:{path}")
    report["input_and_source_hashes_unchanged"] = True
    report["wall_time_s"] = time.perf_counter()-started
    write_json(args.out / "summary.json", report)
    print("PASS: pointwise/rank checks and stronger-majorant ordering verified within declared numerical tolerance", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--ood-bundle", type=Path, default=None)
    parser.add_argument("--kappas", type=float, nargs="+", default=[10, 100, 500])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.cache_dirs and not args.ood_bundle: parser.error("provide a cache directory or OOD bundle")
    if args.bootstrap < 1 or any(k < 1 for k in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        parser.error("positive bootstrap and unique positive budgets required")
    return args


if __name__ == "__main__":
    run(parse_args())
