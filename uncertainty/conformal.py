"""Conformal thresholds and empirical cell-coverage diagnostics for fields.

Independent coefficient/solution *fields* are the exchangeable observations here.
Pooling their dependent cells is useful descriptively but does not create an
exchangeable cell sample or a distribution-free cell-coverage guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    """An interval rule, including the scale floor fixed before evaluation."""

    method: str
    alpha: float
    qhat: float
    mode: str
    scale_floor: float | None = None
    n_calibration: int = 0


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ceil((n + 1) * (1 - alpha)) order statistic.

    The conservative value is infinity when the requested rank exceeds n.
    Positive-infinite scores retain their mass; silently removing invalid or
    infinite observations can make a conformal threshold anticonservative.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("scores must be nonempty")
    if np.any(np.isnan(arr)) or np.any(np.isneginf(arr)):
        raise ValueError("scores must not contain NaN or negative infinity")
    rank = int(np.ceil((arr.size + 1) * (1.0 - alpha)))
    if rank > arr.size:
        return float("inf")
    return float(np.partition(arr, rank - 1)[rank - 1])


def scale_floor_from_calibration(scale: np.ndarray, floor_fraction: float = 0.05) -> float:
    """Fit an empirical scale floor using calibration covariates only.

    This is used for pooled-cell diagnostics. For guaranteed fieldwise split CP,
    fix the scale rule independently of calibration (e.g. on training data).
    """
    arr = _valid_scale(scale)
    if not np.isfinite(floor_fraction) or floor_fraction < 0.0:
        raise ValueError("floor_fraction must be finite and nonnegative")
    positive = arr[arr > 0.0]
    floor = floor_fraction * float(np.median(positive)) if positive.size else 1.0
    return max(floor, 1e-12)


def _valid_scale(scale: np.ndarray) -> np.ndarray:
    arr = np.asarray(scale, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("scale must be nonempty, finite, and nonnegative")
    return arr


def safe_scale(
    scale: np.ndarray,
    floor_fraction: float = 0.05,
    *,
    floor: float | None = None,
) -> np.ndarray:
    """Floor a scale; pass a fixed ``floor`` when evaluating new observations.

    Without ``floor`` this fits a data-adaptive floor for backward compatibility.
    Such fitting must not be repeated on each target batch.
    """
    arr = _valid_scale(scale)
    if floor is None:
        floor = scale_floor_from_calibration(arr, floor_fraction)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and strictly positive")
    return np.maximum(arr, float(floor))


def _valid_error(abs_error: np.ndarray) -> np.ndarray:
    err = np.asarray(abs_error, dtype=np.float64)
    if err.size == 0 or not np.all(np.isfinite(err)) or np.any(err < 0.0):
        raise ValueError("abs_error must be nonempty, finite, and nonnegative")
    return err


def _scale_for_error(scale: np.ndarray, err: np.ndarray, floor: float) -> np.ndarray:
    value = safe_scale(scale, floor=floor)
    try:
        return np.broadcast_to(value, err.shape)
    except ValueError as exc:
        raise ValueError(f"scale shape {value.shape} cannot broadcast to errors {err.shape}") from exc


def calibrate_cellwise(
    abs_error: np.ndarray,
    alpha: float = 0.1,
    scale: np.ndarray | None = None,
    method: str = "global_cell",
    *,
    scale_floor: float | None = None,
) -> ConformalCalibration:
    """Fit a pooled-cell empirical quantile (dependent cells are not IID).

    The conformal order-statistic correction does not by itself establish a
    coverage guarantee for pooled spatial cells. Scale floors are fitted here
    and then reused unchanged on target observations.
    """
    err = _valid_error(abs_error)
    floor = None
    if scale is None:
        scores = err
    else:
        floor = scale_floor_from_calibration(scale) if scale_floor is None else float(scale_floor)
        scores = err / _scale_for_error(scale, err, floor)
    return ConformalCalibration(
        method=method, alpha=alpha, qhat=conformal_quantile(scores, alpha),
        mode="cellwise", scale_floor=floor, n_calibration=int(err.size),
    )


def calibrate_fieldwise(
    abs_error: np.ndarray,
    alpha: float = 0.1,
    scale: np.ndarray | None = None,
    method: str = "global_field",
    *,
    scale_floor: float = 1e-12,
) -> ConformalCalibration:
    """Calibrate simultaneous whole-field coverage using one maximum per field.

    Under exchangeable fields, the usual split-CP guarantee applies when the
    predictor and scale function were fixed independently of these calibration
    fields. The scale floor is fixed (not estimated from a target batch).
    """
    err = _valid_error(abs_error)
    if err.ndim < 2:
        raise ValueError(f"abs_error must include sample and field axes; got {err.shape}")
    floor = None if scale is None else float(scale_floor)
    normed = err if scale is None else err / _scale_for_error(scale, err, floor)
    scores = normed.reshape(normed.shape[0], -1).max(axis=1)
    return ConformalCalibration(
        method=method, alpha=alpha, qhat=conformal_quantile(scores, alpha),
        mode="fieldwise", scale_floor=floor, n_calibration=int(err.shape[0]),
    )


def interval_radius(calibration: ConformalCalibration, scale: np.ndarray | None = None) -> np.ndarray:
    """Return interval half-widths using the calibration rule's fixed floor."""
    if np.isnan(calibration.qhat) or calibration.qhat < 0.0:
        raise ValueError("interval qhat must be nonnegative and not NaN")
    if scale is None:
        if calibration.scale_floor is not None:
            raise ValueError("this calibrated rule requires a scale at evaluation")
        return np.asarray(calibration.qhat, dtype=np.float64)
    # Manually constructed legacy rules have no saved floor. Keep their old
    # behavior, while fitted rules above always carry a fixed positive floor.
    return calibration.qhat * safe_scale(scale, floor=calibration.scale_floor)


def coverage_metrics(
    abs_error: np.ndarray,
    calibration: ConformalCalibration,
    scale: np.ndarray | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate pooled-cell, whole-field, width, and optional masked coverage."""
    err = _valid_error(abs_error)
    raw_radius = interval_radius(calibration, scale)
    try:
        radius = np.broadcast_to(raw_radius, err.shape)
    except ValueError as exc:
        raise ValueError(f"radius shape {raw_radius.shape} cannot broadcast to errors {err.shape}") from exc
    covered = err <= radius
    out: dict[str, float] = {
        "marginal_coverage": float(covered.mean()),
        "fieldwise_coverage": float(covered.reshape(err.shape[0], -1).all(axis=1).mean()) if err.ndim > 1 else float(covered.mean()),
        "mean_width": float(np.mean(2.0 * radius)),
    }
    if masks:
        for name, mask in masks.items():
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != err.shape:
                raise ValueError(f"mask {name} shape {mask_arr.shape} differs from errors {err.shape}")
            out[f"{name}_coverage"] = float(covered[mask_arr].mean()) if np.any(mask_arr) else float("nan")
    return out
