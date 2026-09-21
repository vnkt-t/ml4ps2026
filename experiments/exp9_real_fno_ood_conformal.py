"""Experiment 9: real-FNO OOD conformal coverage and localization suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ["MPLCONFIGDIR"] = ".cache/mpl"
os.environ["XDG_CACHE_HOME"] = ".cache"
os.environ["PYTHONPYCACHEPREFIX"] = ".pytest_cache/pycache"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for cache_dir in [
    REPO_ROOT / ".cache" / "mpl",
    REPO_ROOT / ".cache",
    REPO_ROOT / ".pytest_cache" / "pycache",
    REPO_ROOT / "results",
    REPO_ROOT / "figures",
]:
    cache_dir.mkdir(parents=True, exist_ok=True)

from data.generate_contrast import default_rhs, family_B_sample, family_C_sample  # noqa: E402
from data.generate_grf import grf_coeff  # noqa: E402
from experiments.exp0_pilot_fno import rel_l2  # noqa: E402
from experiments.exp7_weighted_cp import FEATURE_NAMES, coefficient_features  # noqa: E402
from experiments.exp_real_fno_multi_kappa import (  # noqa: E402
    assert_finite_payload,
    bootstrap_mean_ci,
    load_ensemble,
    write_json,
)
from solver.assemble_fd import assemble  # noqa: E402
from solver.residual import residual  # noqa: E402
from solver.solve import solve  # noqa: E402
from uncertainty.compute_ladder import compute_ladder  # noqa: E402
from uncertainty.conformal import (  # noqa: E402
    ConformalCalibration,
    calibrate_cellwise,
    calibrate_fieldwise,
    conformal_quantile,
    coverage_metrics,
    interval_radius,
    safe_scale,
    scale_floor_from_calibration,
)
from uncertainty.weighted_conformal import (  # noqa: E402
    calibrate_weighted,
    coverage_from_scores,
    fit_density_ratio_classifier,
)


DATASET_ORDER = ["test_id", "test_r", "test_b", "test_cs"]
DATASET_LABELS = {
    "calib": "CALIB",
    "test_id": "TEST_ID",
    "test_r": "TEST_R",
    "test_b": "TEST_B",
    "test_cs": "TEST_CS",
}
METHOD_LABELS = OrderedDict(
    [
        ("global_cell", "global cell"),
        ("global_field", "global field"),
        ("ensemble_norm", "ensemble norm"),
        ("raw_residual_norm", "raw |r| norm"),
        ("jacobi20_norm", "Jacobi-20 norm"),
        ("rung4_norm", "rung-4 norm"),
        ("hybrid_ens_rung4", "ens+rung4 hybrid"),
        ("weighted_field_cp", "weighted field"),
    ]
)
CELL_WIDTH_METHODS = [
    "global_cell",
    "global_field",
    "ensemble_norm",
    "raw_residual_norm",
    "jacobi20_norm",
    "rung4_norm",
    "hybrid_ens_rung4",
]
WEIGHTED_CP_DATASET_ORDER = ["test_cs", "test_r", "test_b"]


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _field_fingerprints(a: np.ndarray) -> set[str]:
    out: set[str] = set()
    for field in np.asarray(a):
        arr = np.ascontiguousarray(field, dtype=np.float64)
        out.add(hashlib.sha1(arr.view(np.uint8)).hexdigest())
    return out


def _finite_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _finite_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_dict(v) for v in value]
    if isinstance(value, tuple):
        return [_finite_dict(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_dict(value.tolist())
    if isinstance(value, np.generic):
        return _finite_dict(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def coefficient_feature_matrix(a: np.ndarray) -> np.ndarray:
    return np.stack([coefficient_features(field) for field in np.asarray(a, dtype=np.float64)], axis=0)


def gradient_interface_mask(a: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """Cells in the high-gradient coefficient band, with nonempty interior."""

    arr = np.asarray(a, dtype=np.float64)
    loga = np.log(np.maximum(arr, 1.0e-12))
    jump = np.zeros_like(loga, dtype=np.float64)
    dx = np.abs(np.diff(loga, axis=0))
    dy = np.abs(np.diff(loga, axis=1))
    jump[:-1, :] = np.maximum(jump[:-1, :], dx)
    jump[1:, :] = np.maximum(jump[1:, :], dx)
    jump[:, :-1] = np.maximum(jump[:, :-1], dy)
    jump[:, 1:] = np.maximum(jump[:, 1:], dy)
    positive = jump[jump > 1.0e-12]
    if positive.size == 0:
        mask = np.zeros_like(jump, dtype=bool)
        mask[0, 0] = True
        return mask
    threshold = float(np.quantile(positive, quantile))
    mask = jump >= max(threshold, 1.0e-12)
    if not np.any(mask):
        mask.flat[int(np.argmax(jump))] = True
    if np.all(mask):
        order = np.argsort(-jump.reshape(-1), kind="mergesort")
        keep = max(1, int(round(0.25 * jump.size)))
        mask = np.zeros_like(jump, dtype=bool)
        mask.reshape(-1)[order[:keep]] = True
    return mask


def _family_r_sample(
    N: int,
    rng: np.random.Generator,
    lengthscale_range: tuple[float, float],
    sigma_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    lengthscale = float(rng.uniform(*lengthscale_range))
    sigma = float(rng.uniform(*sigma_range))
    a = grf_coeff(N, rng, lengthscale=lengthscale, sigma=sigma)
    f = default_rhs(N)
    u_true = solve(assemble(a), f).reshape(N, N)
    meta: dict[str, float | str] = {
        "family": "R",
        "generator": "data.generate_grf.grf_coeff",
        "lengthscale": lengthscale,
        "sigma": sigma,
        "source": "data.generate_contrast.default_rhs",
    }
    return a, f, u_true, meta


def _onsupport_family_c_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_incl_range": tuple(int(x) for x in args.onsupport_n_incl),
        "radius_range": tuple(float(x) for x in args.onsupport_radius),
        "channel_prob": float(args.onsupport_channel_prob),
    }


def _generate_raw_dataset(
    name: str,
    family: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(args.n)
    N = int(args.N)
    a_all = np.empty((n, N, N), dtype=np.float64)
    f_all = np.empty_like(a_all)
    u_all = np.empty_like(a_all)
    interface_all = np.empty((n, N, N), dtype=bool)
    meta: list[dict[str, Any]] = []

    for i in range(n):
        if family == "C":
            field_kwargs = _onsupport_family_c_kwargs(args) if name == "test_cs" else {}
            sample = family_C_sample(N=N, kappa=float(args.kappa), rng=rng, **field_kwargs)
            a, f, u_true, sample_meta = sample.a, sample.f, sample.u_true, dict(sample.meta)
        elif family == "B":
            sample = family_B_sample(
                N=N,
                kappa=float(args.kappa),
                rng=rng,
                block_size=int(args.b_block_size) if args.b_block_size else None,
            )
            a, f, u_true, sample_meta = sample.a, sample.f, sample.u_true, dict(sample.meta)
        elif family == "R":
            a, f, u_true, sample_meta = _family_r_sample(
                N=N,
                rng=rng,
                lengthscale_range=tuple(float(x) for x in args.rough_lengthscale),
                sigma_range=tuple(float(x) for x in args.rough_sigma),
            )
        else:
            raise ValueError(f"unknown family {family!r}")
        a_all[i] = a
        f_all[i] = f
        u_all[i] = u_true
        interface_all[i] = gradient_interface_mask(a)
        meta.append(sample_meta)
        if (i + 1) % max(1, int(args.progress_every)) == 0 or i + 1 == n:
            print(f"exp9 generated {DATASET_LABELS[name]}: {i + 1}/{n}", flush=True)

    return {
        "name": name,
        "label": DATASET_LABELS[name],
        "family": family,
        "seed": int(seed),
        "a": a_all,
        "f": f_all,
        "u_true": u_all,
        "interface_mask": interface_all,
        "meta": meta,
    }


def _build_feature_block(
    raw: dict[str, Any],
    ensemble: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    a_all = np.asarray(raw["a"], dtype=np.float64)
    f_all = np.asarray(raw["f"], dtype=np.float64)
    u_all = np.asarray(raw["u_true"], dtype=np.float64)
    n, N, _ = a_all.shape

    pred_members = np.asarray(ensemble.predict_members(a_all), dtype=np.float64)
    u_hat = pred_members.mean(axis=0)
    sigma_ens = pred_members.std(axis=0) if pred_members.shape[0] > 1 else np.zeros_like(u_hat)
    abs_e = np.abs(u_hat - u_all)
    rel = rel_l2(u_hat, u_all)

    abs_r = np.empty_like(abs_e)
    smooth_abs_r = np.empty_like(abs_e)
    jacobi20 = np.empty_like(abs_e)
    rung4 = np.empty_like(abs_e)
    rung4_backend: list[str] = []
    identity_errors: list[float] = []
    oracle_identity_errors: list[float] = []

    for i in range(n):
        A = assemble(a_all[i])
        u_i = np.asarray(u_hat[i], dtype=np.float64)
        f_i = np.asarray(f_all[i], dtype=np.float64)
        err_i = u_i - np.asarray(u_all[i], dtype=np.float64)
        r_vec = residual(A, u_i, f_i)
        if i == 0:
            exact_from_residual = solve(A, r_vec).reshape(N, N)
            identity_errors.append(float(np.max(np.abs(exact_from_residual - err_i))))

        ladder = compute_ladder(
            A=A,
            u_hat=u_i,
            f=f_i,
            u_test=np.asarray(u_all[i], dtype=np.float64),
            u_pred_ensemble=np.asarray(pred_members[:, i], dtype=np.float64),
            sigma_ens=np.asarray(sigma_ens[i], dtype=np.float64),
        )
        abs_r[i] = np.abs(r_vec.reshape(N, N))
        ladder_abs_r = np.asarray(ladder["1_abs_r"]["score"], dtype=np.float64)
        abs_r_delta = float(np.max(np.abs(abs_r[i] - ladder_abs_r)))
        if abs_r_delta > 1.0e-10:
            raise RuntimeError(f"ladder residual mismatch in {raw['label']} sample {i}: {abs_r_delta:.3e}")
        smooth_abs_r[i] = np.asarray(ladder["2_smooth_abs_r"]["score"], dtype=np.float64)
        jacobi20[i] = np.asarray(ladder["3e_jacobi_20"]["score"], dtype=np.float64)
        rung4[i] = np.asarray(ladder["4_vcycle"]["score"], dtype=np.float64)
        rung4_backend.append(str(ladder["4_vcycle"].get("backend", "")))
        oracle_identity_errors.append(
            float(np.max(np.abs(np.asarray(ladder["5_oracle"]["score"], dtype=np.float64) - abs_e[i])))
        )
        if (i + 1) % max(1, int(args.progress_every)) == 0 or i + 1 == n:
            print(f"exp9 ladder {raw['label']}: {i + 1}/{n}", flush=True)

    max_sign_identity = float(np.max(identity_errors)) if identity_errors else 0.0
    max_oracle_identity = float(np.max(oracle_identity_errors)) if oracle_identity_errors else 0.0
    if max_sign_identity > 1.0e-8:
        raise RuntimeError(f"same-A signed residual identity failed in {raw['label']}: {max_sign_identity:.3e}")
    if max_oracle_identity > 1.0e-8:
        raise RuntimeError(f"ladder oracle residual identity failed in {raw['label']}: {max_oracle_identity:.3e}")

    out = dict(raw)
    out.update(
        {
            "pred_members": pred_members,
            "u_hat_mean": u_hat,
            "sigma_ens": sigma_ens,
            "abs_e": abs_e,
            "abs_r": abs_r,
            "smooth_abs_r": smooth_abs_r,
            "jacobi20": jacobi20,
            "rung4": rung4,
            "rung4_backend": np.asarray(rung4_backend, dtype="U64"),
            "rel_l2": rel,
            "same_A_residual_identity_max_abs": max_sign_identity,
            "oracle_identity_max_abs": max_oracle_identity,
        }
    )
    return out


def _safe_median(scale: np.ndarray) -> float:
    return max(float(np.median(safe_scale(scale))), 1.0e-12)


def _hybrid_scale(block: dict[str, Any], medians: dict[str, float]) -> np.ndarray:
    sigma_unit = safe_scale(
        np.asarray(block["sigma_ens"], dtype=np.float64), floor=medians["sigma_ens_floor"]
    ) / medians["sigma_ens"]
    rung_unit = safe_scale(
        np.asarray(block["rung4"], dtype=np.float64), floor=medians["rung4_floor"]
    ) / medians["rung4"]
    return 0.5 * (sigma_unit + rung_unit)


def _calibrate_methods(calib: dict[str, Any], alpha: float) -> tuple[OrderedDict[str, dict[str, Any]], dict[str, float]]:
    medians = {
        "sigma_ens": _safe_median(calib["sigma_ens"]),
        "rung4": _safe_median(calib["rung4"]),
        "sigma_ens_floor": scale_floor_from_calibration(calib["sigma_ens"]),
        "rung4_floor": scale_floor_from_calibration(calib["rung4"]),
    }
    hybrid = _hybrid_scale(calib, medians)
    methods: OrderedDict[str, dict[str, Any]] = OrderedDict(
        [
            (
                "global_cell",
                {
                    "calibration": calibrate_cellwise(calib["abs_e"], alpha=alpha, scale=None, method="global_cell"),
                    "scale": None,
                },
            ),
            (
                "global_field",
                {
                    "calibration": calibrate_fieldwise(calib["abs_e"], alpha=alpha, scale=None, method="global_field"),
                    "scale": None,
                },
            ),
            (
                "ensemble_norm",
                {
                    "calibration": calibrate_cellwise(
                        calib["abs_e"], alpha=alpha, scale=calib["sigma_ens"], method="ensemble_norm"
                    ),
                    "scale": "sigma_ens",
                },
            ),
            (
                "raw_residual_norm",
                {
                    "calibration": calibrate_cellwise(
                        calib["abs_e"], alpha=alpha, scale=calib["abs_r"], method="raw_residual_norm"
                    ),
                    "scale": "abs_r",
                },
            ),
            (
                "jacobi20_norm",
                {
                    "calibration": calibrate_cellwise(
                        calib["abs_e"], alpha=alpha, scale=calib["jacobi20"], method="jacobi20_norm"
                    ),
                    "scale": "jacobi20",
                },
            ),
            (
                "rung4_norm",
                {
                    "calibration": calibrate_cellwise(
                        calib["abs_e"], alpha=alpha, scale=calib["rung4"], method="rung4_norm"
                    ),
                    "scale": "rung4",
                },
            ),
            (
                "hybrid_ens_rung4",
                {
                    "calibration": calibrate_cellwise(
                        calib["abs_e"], alpha=alpha, scale=hybrid, method="hybrid_ens_rung4"
                    ),
                    "scale": "hybrid",
                },
            ),
        ]
    )
    return methods, medians


def _scale_for(block: dict[str, Any], scale_name: str | None, medians: dict[str, float]) -> np.ndarray | None:
    if scale_name is None:
        return None
    if scale_name == "hybrid":
        return _hybrid_scale(block, medians)
    return np.asarray(block[scale_name], dtype=np.float64)


def _radius_array(
    calibration: ConformalCalibration,
    scale: np.ndarray | None,
    shape: tuple[int, int, int],
) -> np.ndarray:
    radius = np.asarray(interval_radius(calibration, scale=scale), dtype=np.float64)
    if radius.ndim == 0:
        return np.full(shape, float(radius), dtype=np.float64)
    return radius.reshape(shape)


def _top_indices(x: np.ndarray, q: float) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    k = max(1, int(np.ceil(q * flat.size)))
    return np.argpartition(flat, -k)[-k:]


def width_localization_metrics(width: np.ndarray, abs_error: np.ndarray) -> dict[str, float]:
    s = np.asarray(width, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    if s.size != e.size:
        raise ValueError(f"width and abs_error sizes differ: {s.size} vs {e.size}")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(e)):
        raise ValueError("width and abs_error must be finite")
    if s.size <= 1 or np.all(e == e[0]):
        return {"spearman": 0.0, "auroc_top10": 0.5}
    if np.all(s == s[0]):
        # Every positive/negative pair is tied: AUROC is chance, not zero.
        return {"spearman": 0.0, "auroc_top10": 0.5}
    rho = spearmanr(s, e).statistic
    if not np.isfinite(rho):
        rho = 0.0
    y = np.zeros(e.size, dtype=np.int8)
    y[_top_indices(e, 0.10)] = 1
    try:
        auc = float(roc_auc_score(y, s))
    except ValueError:
        auc = 0.5
    if not np.isfinite(auc):
        auc = 0.5
    return {"spearman": float(rho), "auroc_top10": auc}


def _bootstrap_masked_coverage(
    covered: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    """Cluster bootstrap a pooled-cell ratio by resampling whole fields.

    Interface masks contain different cell counts across fields, so bootstrapping
    the unweighted mean of per-field coverages estimates a different quantity.
    """
    if covered.shape != mask.shape or covered.ndim < 2:
        raise ValueError("covered and mask must have matching sample/field axes")
    num = (covered & mask).reshape(covered.shape[0], -1).sum(axis=1)
    den = mask.reshape(mask.shape[0], -1).sum(axis=1)
    if np.any(den == 0):
        raise ValueError("each field must contain at least one masked cell")
    estimate = float(num.sum() / den.sum())
    if n_boot <= 0 or num.size == 1:
        return {"mean": estimate, "ci95_low": estimate, "ci95_high": estimate}
    indices = rng.integers(0, num.size, size=(int(n_boot), num.size))
    boot = num[indices].sum(axis=1) / den[indices].sum(axis=1)
    return {
        "mean": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _per_field_eval(
    block: dict[str, Any],
    calibration: ConformalCalibration,
    scale: np.ndarray | None,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    abs_e = np.asarray(block["abs_e"], dtype=np.float64)
    radius = _radius_array(calibration, scale, abs_e.shape)
    covered = abs_e <= radius
    flat_axes = tuple(range(1, abs_e.ndim))
    interface = np.asarray(block["interface_mask"], dtype=bool)
    interior = ~interface

    per_field_marginal = covered.reshape(covered.shape[0], -1).mean(axis=1)
    per_field_fieldwise = covered.reshape(covered.shape[0], -1).all(axis=1).astype(np.float64)
    per_field_width = (2.0 * radius).reshape(radius.shape[0], -1).mean(axis=1)
    per_field_interface = np.array(
        [float(covered[i][interface[i]].mean()) if np.any(interface[i]) else 0.0 for i in range(abs_e.shape[0])],
        dtype=np.float64,
    )
    per_field_interior = np.array(
        [float(covered[i][interior[i]].mean()) if np.any(interior[i]) else 0.0 for i in range(abs_e.shape[0])],
        dtype=np.float64,
    )
    loc = [width_localization_metrics(radius[i], abs_e[i]) for i in range(abs_e.shape[0])]
    per_field_rho = np.array([x["spearman"] for x in loc], dtype=np.float64)
    per_field_auroc = np.array([x["auroc_top10"] for x in loc], dtype=np.float64)

    ci = {
        "marginal_coverage": bootstrap_mean_ci(per_field_marginal, rng, n_boot),
        "fieldwise_coverage": bootstrap_mean_ci(per_field_fieldwise, rng, n_boot),
        "mean_width": bootstrap_mean_ci(per_field_width, rng, n_boot),
        "interface_coverage": _bootstrap_masked_coverage(covered, interface, rng, n_boot),
        "interior_coverage": _bootstrap_masked_coverage(covered, interior, rng, n_boot),
        "width_error_spearman": bootstrap_mean_ci(per_field_rho, rng, n_boot),
        "width_error_auroc_top10": bootstrap_mean_ci(per_field_auroc, rng, n_boot),
    }
    return {
        "radius": radius,
        "covered": covered,
        "per_field": {
            "marginal_coverage": per_field_marginal,
            "fieldwise_coverage": per_field_fieldwise,
            "mean_width": per_field_width,
            "interface_coverage": per_field_interface,
            "interior_coverage": per_field_interior,
            "width_error_spearman": per_field_rho,
            "width_error_auroc_top10": per_field_auroc,
        },
        "ci": ci,
        "covered_axes": flat_axes,
    }


def _coverage_row(
    block: dict[str, Any],
    method: str,
    calibration: ConformalCalibration,
    scale: np.ndarray | None,
    medians: dict[str, float],
    rng: np.random.Generator,
    n_boot: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    masks = {
        "interface": np.asarray(block["interface_mask"], dtype=bool),
        "interior": ~np.asarray(block["interface_mask"], dtype=bool),
    }
    metrics = coverage_metrics(block["abs_e"], calibration, scale=scale, masks=masks)
    per_field = _per_field_eval(block, calibration, scale, rng, n_boot)
    row: dict[str, Any] = {
        "test_set": str(block["name"]),
        "test_label": str(block["label"]),
        "family": str(block["family"]),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "alpha": float(calibration.alpha),
        "qhat": float(calibration.qhat),
        "mode": str(calibration.mode),
        "scale_floor": calibration.scale_floor,
        "calibration_count": int(calibration.n_calibration),
        "coverage_scope": (
            "exchangeable_field_split_conformal" if method == "global_field"
            else "empirical_weighted_field_diagnostic" if method == "weighted_field_cp"
            else "empirical_pooled_cell_diagnostic"
        ),
        "scale": "none" if scale is None else str(extra.get("scale_name") if extra else "array"),
        "hybrid_sigma_calib_median": float(medians.get("sigma_ens", 0.0)),
        "hybrid_rung4_calib_median": float(medians.get("rung4", 0.0)),
        "marginal_coverage": float(metrics["marginal_coverage"]),
        "fieldwise_coverage": float(metrics["fieldwise_coverage"]),
        "mean_width": float(metrics["mean_width"]),
        "interface_coverage": float(metrics["interface_coverage"]),
        "interior_coverage": float(metrics["interior_coverage"]),
        "width_error_spearman": float(per_field["ci"]["width_error_spearman"]["mean"]),
        "width_error_auroc_top10": float(per_field["ci"]["width_error_auroc_top10"]["mean"]),
    }
    for metric_name, stats in per_field["ci"].items():
        row[f"{metric_name}_ci95_low"] = float(stats["ci95_low"])
        row[f"{metric_name}_ci95_high"] = float(stats["ci95_high"])
    if extra:
        row.update({k: v for k, v in extra.items() if k != "scale_name"})
    return row


def _base_coverage_rows(
    blocks: dict[str, dict[str, Any]],
    methods: OrderedDict[str, dict[str, Any]],
    medians: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))
    for dataset_name in DATASET_ORDER:
        block = blocks[dataset_name]
        for method, spec in methods.items():
            scale_name = spec["scale"]
            scale = _scale_for(block, scale_name, medians)
            rows.append(
                _coverage_row(
                    block,
                    method,
                    spec["calibration"],
                    scale,
                    medians,
                    rng,
                    int(args.bootstrap),
                    extra={"scale_name": scale_name or "none"},
                )
            )
    return rows


def _weighted_rows(
    calib: dict[str, Any],
    blocks: dict[str, dict[str, Any]],
    medians: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_X = coefficient_feature_matrix(calib["a"])
    source_scores = np.asarray(calib["abs_e"], dtype=np.float64).reshape(calib["abs_e"].shape[0], -1).max(axis=1)
    unweighted_q = conformal_quantile(source_scores, float(args.alpha))
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "score": "fieldwise max_abs_error",
        "feature_names": FEATURE_NAMES,
        "source_feature_source": "coefficient_features(a) only; no errors, predictions, or scores",
        "source_score_summary": coverage_from_scores(source_scores, float("inf")),
        "guarantee": "none; estimated clipped ratios and fixed test mass are an empirical diagnostic",
        "ratio_fit_data": "source calibration and unlabeled target covariates (in-sample classifier fit)",
        "test_mass": "fixed normalized weight; not an individual target density ratio",
        "support_overlap": "not established by classifier AUC or matching coefficient extrema",
    }
    rng = np.random.default_rng(int(args.bootstrap_seed) + 909)

    for dataset_name, offset, claim_scope in [
        ("test_r", 17, "roughness-shift weighted-CP diagnostic"),
        ("test_b", 29, "documented support-breaking failure diagnostic"),
        ("test_cs", 41, "in-family geometry-shift empirical weighted-CP diagnostic"),
    ]:
        block = blocks[dataset_name]
        target_X = coefficient_feature_matrix(block["a"])
        target_scores = np.asarray(block["abs_e"], dtype=np.float64).reshape(block["abs_e"].shape[0], -1).max(axis=1)
        ratio = fit_density_ratio_classifier(
            source_X,
            target_X,
            random_state=int(args.seed) + offset,
            model=str(args.ratio_model),
        )
        weighted_cal = calibrate_weighted(
            source_scores,
            ratio.source_weights,
            alpha=float(args.alpha),
            method=f"weighted_{dataset_name}",
            clip_quantile=float(args.weight_clip_quantile),
            test_weight=float(args.test_weight),
        )
        calibration = ConformalCalibration(
            method="weighted_field_cp",
            alpha=float(args.alpha),
            qhat=float(weighted_cal.qhat),
            mode="fieldwise",
        )
        unweighted_metrics = coverage_from_scores(target_scores, unweighted_q)
        weighted_metrics = coverage_from_scores(target_scores, weighted_cal.qhat)
        rows.append(
            _coverage_row(
                block,
                "weighted_field_cp",
                calibration,
                None,
                medians,
                rng,
                int(args.bootstrap),
                extra={
                    "scale_name": "none",
                    "weighted_claim_scope": claim_scope,
                    "weighted_ratio_model": str(args.ratio_model),
                    "weighted_classifier_train_auc": float(ratio.train_auc),
                    "weighted_effective_sample_size": float(weighted_cal.effective_sample_size),
                    "weighted_weight_min": float(weighted_cal.weight_min),
                    "weighted_weight_max": float(weighted_cal.weight_max),
                    "unweighted_scalar_qhat": float(unweighted_q),
                    "unweighted_scalar_fieldwise_coverage": float(unweighted_metrics["coverage"]),
                    "weighted_scalar_fieldwise_coverage": float(weighted_metrics["coverage"]),
                },
            )
        )
        diagnostics[block["label"]] = {
            "claim_scope": claim_scope,
            "unweighted": {
                "qhat": float(unweighted_q),
                "coverage": unweighted_metrics,
            },
            "weighted": {
                "qhat": float(weighted_cal.qhat),
                "coverage": weighted_metrics,
                "calibration": asdict(weighted_cal),
            },
            "density_ratio": {
                "model": str(args.ratio_model),
                "classifier_train_auc": float(ratio.train_auc),
                "source_weight_summary": ratio.source_weight_summary,
                "target_weight_summary": ratio.target_weight_summary,
                "common_weight_normalization": float(ratio.weight_normalization),
            },
            "target_score_summary": coverage_from_scores(target_scores, float("inf")),
            "target_feature_summary": {
                name: _summary_stats(target_X[:, j]) for j, name in enumerate(FEATURE_NAMES)
            },
        }
    return rows, diagnostics


def _save_bundle(blocks: dict[str, dict[str, Any]], args: argparse.Namespace, self_checks: dict[str, Any]) -> None:
    bundle_datasets = ["calib", *DATASET_ORDER]
    bundle: dict[str, Any] = {
        "dataset_names": np.asarray(bundle_datasets, dtype="U16"),
        "grid_N": np.asarray(int(args.N), dtype=np.int64),
        "kappa": np.asarray(float(args.kappa), dtype=np.float64),
        "alpha": np.asarray(float(args.alpha), dtype=np.float64),
        "feature_names": np.asarray(FEATURE_NAMES, dtype="U32"),
        "self_check_keys": np.asarray(sorted(self_checks.keys()), dtype="U96"),
    }
    for name in bundle_datasets:
        block = blocks[name]
        prefix = name
        for key in [
            "a",
            "f",
            "u_true",
            "u_hat_mean",
            "sigma_ens",
            "abs_e",
            "abs_r",
            "smooth_abs_r",
            "jacobi20",
            "rung4",
            "interface_mask",
            "rel_l2",
        ]:
            bundle[f"{prefix}_{key}"] = np.asarray(block[key])
        bundle[f"{prefix}_rung4_backend"] = np.asarray(block["rung4_backend"], dtype="U64")
        bundle[f"{prefix}_seed"] = np.asarray(int(block["seed"]), dtype=np.int64)
        bundle[f"{prefix}_family"] = np.asarray(str(block["family"]), dtype="U8")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.bundle, **bundle)
    with np.load(args.bundle, allow_pickle=False) as npz:
        required = ["calib_a", *[f"{name}_abs_e" for name in DATASET_ORDER], "test_cs_rel_l2"]
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise RuntimeError(f"saved bundle missing keys: {missing}")


def _make_coverage_figure(coverage: pd.DataFrame, out_path: Path, alpha: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    methods = CELL_WIDTH_METHODS
    labels = [METHOD_LABELS[m] for m in methods]
    datasets = DATASET_ORDER
    colors = {"test_id": "tab:blue", "test_r": "tab:orange", "test_b": "tab:green", "test_cs": "tab:red"}
    x = np.arange(len(methods))
    width = min(0.20, 0.80 / max(1, len(datasets)))
    for ax, metric, ylabel in [
        (axes[0], "marginal_coverage", "cell coverage"),
        (axes[1], "mean_width", "mean interval width"),
    ]:
        for j, dataset in enumerate(datasets):
            vals = []
            for method in methods:
                sub = coverage[(coverage["test_set"] == dataset) & (coverage["method"] == method)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            offset = (j - 0.5 * (len(datasets) - 1)) * width
            ax.bar(x + offset, vals, width=width, color=colors[dataset], label=DATASET_LABELS[dataset])
        if metric == "marginal_coverage":
            ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1.0, label="nominal")
            ax.set_ylim(0.0, 1.03)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_weighted_cp_figure(coverage: pd.DataFrame, out_path: Path, alpha: float) -> None:
    rows = []
    for dataset in WEIGHTED_CP_DATASET_ORDER:
        sub = coverage[(coverage["test_set"] == dataset) & (coverage["method"] == "weighted_field_cp")]
        if len(sub) != 1:
            raise ValueError(f"expected one weighted_field_cp row for {dataset}, found {len(sub)}")
        rows.append(sub.iloc[0])

    labels = [DATASET_LABELS[str(row["test_set"])] for row in rows]
    unweighted = np.asarray([float(row["unweighted_scalar_fieldwise_coverage"]) for row in rows], dtype=np.float64)
    weighted = np.asarray([float(row["weighted_scalar_fieldwise_coverage"]) for row in rows], dtype=np.float64)
    aucs = [float(row["weighted_classifier_train_auc"]) for row in rows]
    ess = [float(row["weighted_effective_sample_size"]) for row in rows]

    x = np.arange(len(rows), dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    bars_unweighted = ax.bar(x - width / 2, unweighted, width=width, color="tab:gray", label="unweighted")
    bars_weighted = ax.bar(x + width / 2, weighted, width=width, color="tab:purple", label="weighted")
    ax.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1.0, label="nominal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fieldwise coverage")
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Empirical weighted field coverage across shifts")
    ax.legend(frameon=False, fontsize=9, loc="center right")

    for bars in [bars_unweighted, bars_weighted]:
        for bar in bars:
            height = float(bar.get_height())
            y = height + 0.025 if height > 0.0 else 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for xi, auc, eff_n in zip(x, aucs, ess):
        ax.text(
            xi,
            1.015,
            f"AUC={auc:.3f}\nESS={eff_n:.1f}",
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_localization_figure(coverage: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    methods = CELL_WIDTH_METHODS
    labels = [METHOD_LABELS[m] for m in methods]
    datasets = DATASET_ORDER
    colors = {"test_id": "tab:blue", "test_r": "tab:orange", "test_b": "tab:green", "test_cs": "tab:red"}
    x = np.arange(len(methods))
    width = min(0.20, 0.80 / max(1, len(datasets)))
    for ax, metric, ylabel in [
        (axes[0], "width_error_spearman", "Spearman rho"),
        (axes[1], "width_error_auroc_top10", "AUROC top-10% error"),
    ]:
        for j, dataset in enumerate(datasets):
            vals = []
            for method in methods:
                sub = coverage[(coverage["test_set"] == dataset) & (coverage["method"] == method)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            offset = (j - 0.5 * (len(datasets) - 1)) * width
            ax.bar(x + offset, vals, width=width, color=colors[dataset], label=DATASET_LABELS[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _apply_smoke_paths(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.n = 12
    args.bootstrap = min(int(args.bootstrap), 100)
    args.progress_every = min(int(args.progress_every), 4)
    defaults = {
        "out_json": Path("results/exp9_ood_conformal.json"),
        "out_csv": Path("results/exp9_ood_conformal.csv"),
        "bundle": Path("results/ood_features_bundle.npz"),
        "coverage_fig": Path("figures/fig9_ood_conformal_coverage.png"),
        "localization_fig": Path("figures/fig9_ood_width_localization.png"),
        "weighted_cp_fig": Path("figures/fig9_ood_weighted_cp.png"),
    }
    smoke_paths = {
        "out_json": Path("results/_smoke_exp9_ood_conformal.json"),
        "out_csv": Path("results/_smoke_exp9_ood_conformal.csv"),
        "bundle": Path("results/_smoke_ood_features_bundle.npz"),
        "coverage_fig": Path("figures/_smoke_fig9_ood_conformal_coverage.png"),
        "localization_fig": Path("figures/_smoke_fig9_ood_width_localization.png"),
        "weighted_cp_fig": Path("figures/_smoke_fig9_ood_weighted_cp.png"),
    }
    for attr, default in defaults.items():
        if getattr(args, attr) == default:
            setattr(args, attr, smoke_paths[attr])
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa", type=float, default=100.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=202606069)
    parser.add_argument("--seed-base", type=int, default=9000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026060691)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--rough-lengthscale", type=float, nargs=2, default=(0.035, 0.065))
    parser.add_argument("--rough-sigma", type=float, nargs=2, default=(0.90, 1.25))
    parser.add_argument("--b-block-size", type=int, default=4)
    parser.add_argument("--onsupport-n-incl", type=int, nargs=2, default=(4, 7))
    parser.add_argument("--onsupport-radius", type=float, nargs=2, default=(0.07, 0.15))
    parser.add_argument("--onsupport-channel-prob", type=float, default=0.75)
    parser.add_argument("--ratio-model", choices=["logistic", "gradient_boosting"], default="logistic")
    parser.add_argument("--weight-clip-quantile", type=float, default=0.995)
    parser.add_argument("--test-weight", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, default=Path("results/exp9_ood_conformal.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/exp9_ood_conformal.csv"))
    parser.add_argument("--bundle", type=Path, default=Path("results/ood_features_bundle.npz"))
    parser.add_argument("--coverage-fig", type=Path, default=Path("figures/fig9_ood_conformal_coverage.png"))
    parser.add_argument("--localization-fig", type=Path, default=Path("figures/fig9_ood_width_localization.png"))
    parser.add_argument("--weighted-cp-fig", type=Path, default=Path("figures/fig9_ood_weighted_cp.png"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    return _apply_smoke_paths(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    set_global_seed(int(args.seed))
    print(f"exp9 loading real FNO ensemble kappa={float(args.kappa):g} seeds={args.seeds}", flush=True)
    ensemble = load_ensemble(float(args.kappa), seeds=[int(s) for s in args.seeds], device=str(args.device), smoke=False)
    if len(ensemble) != len(args.seeds):
        raise RuntimeError(f"expected {len(args.seeds)} ensemble members, got {len(ensemble)}")
    grid_N = int(ensemble[0].metadata["grid_N"])
    if grid_N != int(args.N):
        raise ValueError(f"checkpoint grid_N={grid_N}; requested N={args.N}")
    normalizations = [member.metadata["normalization"] for member in ensemble]
    frozen_normalization = dict(normalizations[0])
    for idx, norm in enumerate(normalizations[1:], start=1):
        for key, value in frozen_normalization.items():
            if not np.isclose(float(value), float(norm[key]), rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"normalization mismatch in ensemble member {idx} key {key}")
    print(
        "exp9 frozen Family-C normalization "
        f"logmean={frozen_normalization['logmean']:.6g} logstd={frozen_normalization['logstd']:.6g}",
        flush=True,
    )

    seeds = {
        "calib": int(args.seed_base),
        "test_id": int(args.seed_base) + 10000,
        "test_r": int(args.seed_base) + 20000,
        "test_b": int(args.seed_base) + 30000,
        "test_cs": int(args.seed_base) + 40000,
    }
    raw_blocks = {
        "calib": _generate_raw_dataset("calib", "C", args, seeds["calib"]),
        "test_id": _generate_raw_dataset("test_id", "C", args, seeds["test_id"]),
        "test_r": _generate_raw_dataset("test_r", "R", args, seeds["test_r"]),
        "test_b": _generate_raw_dataset("test_b", "B", args, seeds["test_b"]),
        "test_cs": _generate_raw_dataset("test_cs", "C", args, seeds["test_cs"]),
    }
    blocks = {name: _build_feature_block(block, ensemble, args) for name, block in raw_blocks.items()}

    fingerprints = {name: _field_fingerprints(block["a"]) for name, block in blocks.items()}
    contamination_pairs: dict[str, int] = {}
    for left in fingerprints:
        for right in fingerprints:
            if left >= right:
                continue
            contamination_pairs[f"{left}__{right}"] = len(fingerprints[left] & fingerprints[right])
    if any(v != 0 for v in contamination_pairs.values()):
        raise RuntimeError(f"calib/test contamination detected: {contamination_pairs}")

    calib_X = coefficient_feature_matrix(blocks["calib"]["a"])
    test_cs_X = coefficient_feature_matrix(blocks["test_cs"]["a"])
    feature_idx = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    fixed_support_features = {}
    fixed_support_match = True
    for feature_name in ["log_a_max", "log_a_contrast"]:
        cvals = calib_X[:, feature_idx[feature_name]]
        tvals = test_cs_X[:, feature_idx[feature_name]]
        fixed_support_features[feature_name] = {
            "calib_min": float(np.min(cvals)),
            "calib_max": float(np.max(cvals)),
            "test_cs_min": float(np.min(tvals)),
            "test_cs_max": float(np.max(tvals)),
            "mean_abs_delta": float(abs(np.mean(cvals) - np.mean(tvals))),
        }
        fixed_support_match = fixed_support_match and bool(
            np.ptp(cvals) <= 1.0e-12 and np.ptp(tvals) <= 1.0e-12 and abs(np.mean(cvals) - np.mean(tvals)) <= 1.0e-12
        )

    methods, medians = _calibrate_methods(blocks["calib"], float(args.alpha))
    rows = _base_coverage_rows(blocks, methods, medians, args)
    weighted_rows, weighted_diagnostics = _weighted_rows(blocks["calib"], blocks, medians, args)
    rows.extend(weighted_rows)
    coverage_df = pd.DataFrame(rows)
    for col in coverage_df.columns:
        if pd.api.types.is_numeric_dtype(coverage_df[col]):
            coverage_df[col] = coverage_df[col].fillna(0.0)
        else:
            coverage_df[col] = coverage_df[col].fillna("")
    numeric_df = coverage_df.select_dtypes(include=[np.number])
    finite_mask = np.isfinite(numeric_df.to_numpy(dtype=np.float64))
    if not np.all(finite_mask):
        bad_rows, bad_cols = np.where(~finite_mask)
        details = []
        for r, c in zip(bad_rows[:10], bad_cols[:10]):
            row = coverage_df.iloc[int(r)]
            col = numeric_df.columns[int(c)]
            details.append(f"{row['test_set']}:{row['method']}:{col}={row[col]}")
        raise RuntimeError("coverage table contains non-finite numeric values: " + "; ".join(details))

    _make_coverage_figure(coverage_df, args.coverage_fig, float(args.alpha))
    _make_localization_figure(coverage_df, args.localization_fig)
    _make_weighted_cp_figure(coverage_df, args.weighted_cp_fig, float(args.alpha))

    self_checks = {
        "calib_test_id_same_distribution_disjoint_seeds": bool(seeds["calib"] != seeds["test_id"]),
        "test_cs_seed_disjoint_from_all_other_sets": bool(
            all(seeds["test_cs"] != seed for name, seed in seeds.items() if name != "test_cs")
        ),
        "test_cs_log_a_max_log_a_contrast_match_calib": fixed_support_match,
        "test_cs_log_a_max_log_a_contrast_feature_ranges": fixed_support_features,
        "frozen_family_C_normalization_used_for_all_predictions": True,
        "same_A_signed_residual_identity_max_abs": float(
            max(block["same_A_residual_identity_max_abs"] for block in blocks.values())
        ),
        "ladder_oracle_identity_max_abs": float(max(block["oracle_identity_max_abs"] for block in blocks.values())),
        "weighted_cp_features_are_coefficient_only": True,
        "calib_test_contamination_pair_counts": contamination_pairs,
        "bundle_reloadable": False,
    }
    _save_bundle(blocks, args, self_checks)
    self_checks["bundle_reloadable"] = True

    out_json = REPO_ROOT / args.out_json
    out_csv = REPO_ROOT / args.out_csv
    payload = {
        "experiment": "exp9_real_fno_ood_conformal",
        "smoke": bool(args.smoke),
        "alpha": float(args.alpha),
        "target_coverage": float(1.0 - float(args.alpha)),
        "grid_N": int(args.N),
        "kappa": float(args.kappa),
        "n_per_dataset": int(args.n),
        "seeds": seeds,
        "ensemble_seeds": [int(s) for s in args.seeds],
        "checkpoint_paths": [str(member.checkpoint.relative_to(REPO_ROOT)) for member in ensemble],
        "frozen_normalization": frozen_normalization,
        "source_f": "data.generate_contrast.default_rhs",
        "onsupport_test_cs_geometry": _onsupport_family_c_kwargs(args),
        "hybrid_definition": (
            "0.5 * (safe_scale(sigma_ens, floor=calib_sigma_floor) / median_calib_safe_sigma + "
            "safe_scale(rung4, floor=calib_rung4_floor) / median_calib_safe_rung4)"
        ),
        "calibration_protocol": {
            "independent_observation": "one coefficient/solution field",
            "cellwise_guarantee": "none: dependent spatial cells are pooled descriptively",
            "global_field_guarantee": "split conformal under exchangeable calibration/test fields",
            "ood_guarantee": "none",
            "scale_floors": "fitted on calibration covariates and frozen for all target batches",
            "masked_coverage_ci": "whole-field cluster bootstrap of pooled covered/masked cell ratio",
        },
        "datasets": {
            name: {
                "label": block["label"],
                "family": block["family"],
                "seed": int(block["seed"]),
                "rel_l2": bootstrap_mean_ci(
                    np.asarray(block["rel_l2"], dtype=np.float64),
                    np.random.default_rng(int(args.bootstrap_seed) + idx),
                    int(args.bootstrap),
                ),
                "interface_fraction": float(np.mean(block["interface_mask"])),
                "rung4_backend": str(block["rung4_backend"][0]) if len(block["rung4_backend"]) else "",
            }
            for idx, (name, block) in enumerate(blocks.items())
        },
        "calibrations": {
            method: {
                "qhat": float(spec["calibration"].qhat),
                "mode": str(spec["calibration"].mode),
                "scale": str(spec["scale"] or "none"),
                "scale_floor": spec["calibration"].scale_floor,
                "n_calibration": int(spec["calibration"].n_calibration),
            }
            for method, spec in methods.items()
        },
        "weighted_cp": weighted_diagnostics,
        "coverage": coverage_df.to_dict(orient="records"),
        "self_checks": self_checks,
        "outputs": {
            "json": str(out_json.relative_to(REPO_ROOT)),
            "csv": str(out_csv.relative_to(REPO_ROOT)),
            "bundle": str((REPO_ROOT / args.bundle).relative_to(REPO_ROOT)),
            "coverage_figure": str((REPO_ROOT / args.coverage_fig).relative_to(REPO_ROOT)),
            "localization_figure": str((REPO_ROOT / args.localization_fig).relative_to(REPO_ROOT)),
            "weighted_cp_figure": str((REPO_ROOT / args.weighted_cp_fig).relative_to(REPO_ROOT)),
        },
        "wall_time_s": float(time.time() - t0),
    }
    payload = _finite_dict(payload)
    assert_finite_payload(payload)
    write_json(payload, out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(out_csv, index=False)

    id_global = coverage_df[(coverage_df["test_set"] == "test_id") & (coverage_df["method"] == "global_cell")].iloc[0]
    weighted_r = weighted_diagnostics["TEST_R"]
    weighted_b = weighted_diagnostics["TEST_B"]
    weighted_cs = weighted_diagnostics["TEST_CS"]
    print("EXP9_SUMMARY", flush=True)
    print(
        pd.DataFrame(
            [
                {
                    "check": "ID global-cell marginal coverage",
                    "value": float(id_global["marginal_coverage"]),
                },
                {
                    "check": "Family-R unweighted field score coverage",
                    "value": float(weighted_r["unweighted"]["coverage"]["coverage"]),
                },
                {
                    "check": "Family-R weighted field score coverage",
                    "value": float(weighted_r["weighted"]["coverage"]["coverage"]),
                },
                {
                    "check": "Family-B weighted field score coverage",
                    "value": float(weighted_b["weighted"]["coverage"]["coverage"]),
                },
                {
                    "check": "TEST_CS unweighted field score coverage",
                    "value": float(weighted_cs["unweighted"]["coverage"]["coverage"]),
                },
                {
                    "check": "TEST_CS weighted field score coverage",
                    "value": float(weighted_cs["weighted"]["coverage"]["coverage"]),
                },
                {
                    "check": "TEST_CS density-ratio classifier AUC",
                    "value": float(weighted_cs["density_ratio"]["classifier_train_auc"]),
                },
                {
                    "check": "TEST_CS weighted effective sample size",
                    "value": float(weighted_cs["weighted"]["calibration"]["effective_sample_size"]),
                },
            ]
        ).to_string(index=False),
        flush=True,
    )
    print(f"saved {out_json.relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {out_csv.relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {(REPO_ROOT / args.bundle).relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {(REPO_ROOT / args.coverage_fig).relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {(REPO_ROOT / args.localization_fig).relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {(REPO_ROOT / args.weighted_cp_fig).relative_to(REPO_ROOT)}", flush=True)
    print("OOD_CONFORMAL_DONE", flush=True)
    return payload


if __name__ == "__main__":
    run(parse_args())
