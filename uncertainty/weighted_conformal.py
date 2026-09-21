"""Weighted split-conformal helpers for covariate shift diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np


@dataclass(frozen=True)
class WeightedConformalCalibration:
    """A weighted split-conformal threshold for scalar nonconformity scores."""

    method: str
    alpha: float
    qhat: float
    n_calib: int
    effective_sample_size: float
    weight_min: float
    weight_max: float
    weight_mean: float
    test_weight: float
    clip_quantile: float | None


@dataclass(frozen=True)
class DensityRatioResult:
    """Classifier-based density-ratio estimates."""

    estimator: Any
    source_weights: np.ndarray
    target_weights: np.ndarray
    train_auc: float
    source_weight_summary: dict[str, float]
    target_weight_summary: dict[str, float]
    weight_normalization: float = 1.0


def _valid_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be nonempty, finite, and nonnegative")
    return w


def effective_sample_size(weights: np.ndarray) -> float:
    """Return Kish's effective sample size."""

    w = _valid_weights(weights)
    maximum = float(np.max(w))
    if maximum == 0.0:
        return 0.0
    w = w / maximum
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def normalize_density_ratio_weights(
    weights: np.ndarray,
    *,
    clip_quantile: float | None = 0.995,
    floor: float = 1e-12,
) -> np.ndarray:
    """Return positive weights normalized to mean one.

    Classifier odds can be heavy-tailed. Clipping and flooring alter density
    ratios and therefore remove the exact weighted-CP coverage guarantee. This
    helper is for empirical diagnostics, not a support-overlap certificate.
    """

    w = _valid_weights(weights)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and strictly positive")
    w = np.maximum(w, floor)
    if clip_quantile is not None:
        if not 0.0 < clip_quantile <= 1.0:
            raise ValueError(f"clip_quantile must be in (0, 1]; got {clip_quantile}")
        cap = float(np.quantile(w, clip_quantile))
        w = np.minimum(w, max(cap, floor))
    w = w / float(np.max(w))
    return w / float(np.mean(w))


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    test_weight: float = 1.0,
) -> float:
    """Weighted split-conformal quantile for scalar scores.

    Calibration weights and ``test_weight`` must use the SAME ratio scale.
    The latter is the individual target point's density ratio, whose mass is
    placed at infinity. Jointly rescaling all weights leaves the result fixed;
    independently normalizing calibration and target weights does not.

    A fixed test weight of 1 is an empirical approximation unless the target
    ratio really equals 1. Exact covariate-shift coverage further requires the
    true ratios, support overlap, and an unchanged conditional response law.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    w = _valid_weights(weights)
    if s.size != w.size:
        raise ValueError(f"scores and weights sizes differ: {s.size} vs {w.size}")
    if np.any(np.isnan(s)) or np.any(np.isneginf(s)):
        raise ValueError("scores must not contain NaN or negative infinity")
    if not np.isfinite(test_weight) or test_weight < 0.0:
        raise ValueError("test_weight must be finite and nonnegative")
    mask = w > 0.0
    s = s[mask]
    w = w[mask]
    if s.size == 0:
        if test_weight > 0.0:
            return float("inf")
        raise ValueError("calibration and test weights cannot all be zero")

    # Avoid overflow in the total mass without changing the weighted measure.
    common_scale = max(float(np.max(w)), float(test_weight))
    w = w / common_scale
    test_weight = float(test_weight) / common_scale

    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    target_mass = (1.0 - alpha) * (float(np.sum(w_sorted)) + float(test_weight))
    cumulative = np.cumsum(w_sorted)
    if target_mass > float(cumulative[-1]):
        return float("inf")
    idx = int(np.searchsorted(cumulative, target_mass, side="left"))
    return float(s_sorted[min(idx, s_sorted.size - 1)])


def calibrate_weighted(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.1,
    *,
    method: str = "weighted",
    clip_quantile: float | None = 0.995,
    test_weight: float = 1.0,
) -> WeightedConformalCalibration:
    """Fit the historical clipped-weight scalar diagnostic.

    ``test_weight`` is on the normalized (mean-one calibration) scale. For
    exact target-specific quantiles from true ratios, use
    :func:`weighted_conformal_quantile` directly, without separate clipping or
    normalization. Estimated/clipped ratios confer no exact guarantee.
    """

    w = normalize_density_ratio_weights(weights, clip_quantile=clip_quantile)
    qhat = weighted_conformal_quantile(scores, w, alpha, test_weight=test_weight)
    return WeightedConformalCalibration(
        method=method,
        alpha=float(alpha),
        qhat=qhat,
        n_calib=int(w.size),
        effective_sample_size=effective_sample_size(w),
        weight_min=float(np.min(w)),
        weight_max=float(np.max(w)),
        weight_mean=float(np.mean(w)),
        test_weight=float(test_weight),
        clip_quantile=clip_quantile,
    )


def coverage_from_scores(scores: np.ndarray, qhat: float) -> dict[str, float]:
    """Evaluate scalar-score coverage for a conformal threshold."""

    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0 or np.any(np.isnan(s)) or np.any(np.isneginf(s)):
        raise ValueError("scores must be nonempty and contain no NaN or negative infinity")
    if np.isnan(qhat):
        raise ValueError("qhat must not be NaN")
    covered = s <= float(qhat)
    return {
        "coverage": float(np.mean(covered)),
        "n": int(covered.size),
        "score_mean": float(np.mean(s)),
        "score_p50": float(np.quantile(s, 0.50)),
        "score_p90": float(np.quantile(s, 0.90)),
        "score_p95": float(np.quantile(s, 0.95)),
        "score_max": float(np.max(s)),
    }


def _weight_summary(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(w)),
        "p50": float(np.quantile(w, 0.50)),
        "p90": float(np.quantile(w, 0.90)),
        "p99": float(np.quantile(w, 0.99)),
        "max": float(np.max(w)),
        "mean": float(np.mean(w)),
        "ess": effective_sample_size(w),
    }


def density_ratio_from_classifier(
    estimator: Any,
    X: np.ndarray,
    *,
    n_source: int,
    n_target: int,
    probability_floor: float = 1e-4,
) -> np.ndarray:
    """Convert classifier target probabilities to p_target(x) / p_source(x)."""

    if n_source <= 0 or n_target <= 0:
        raise ValueError("n_source and n_target must be strictly positive")
    if not 0.0 < probability_floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5)")

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or not np.all(np.isfinite(X_arr)):
        raise ValueError("X must be a finite 2D feature array")
    X_arr = np.clip(X_arr, -1e6, 1e6)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        proba = np.asarray(estimator.predict_proba(X_arr), dtype=np.float64)[:, 1]
    p = np.clip(proba, probability_floor, 1.0 - probability_floor)
    prior_correction = float(n_source) / float(n_target)
    return (p / (1.0 - p)) * prior_correction


def fit_density_ratio_classifier(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    random_state: int = 0,
    model: str = "logistic",
    probability_floor: float = 1e-4,
) -> DensityRatioResult:
    """Fit a source-vs-target classifier and return density-ratio weights."""

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xs = np.asarray(X_source, dtype=np.float64)
    Xt = np.asarray(X_target, dtype=np.float64)
    if not np.all(np.isfinite(Xs)) or not np.all(np.isfinite(Xt)):
        raise ValueError("source and target features must be finite")
    Xs = np.clip(Xs, -1e6, 1e6)
    Xt = np.clip(Xt, -1e6, 1e6)
    if Xs.ndim != 2 or Xt.ndim != 2:
        raise ValueError("X_source and X_target must be 2D arrays")
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError(f"feature dimension mismatch: {Xs.shape[1]} vs {Xt.shape[1]}")
    if Xs.shape[0] == 0 or Xt.shape[0] == 0 or Xs.shape[1] == 0:
        raise ValueError("source and target must contain observations and features")

    X = np.vstack([Xs, Xt])
    y = np.concatenate([np.zeros(Xs.shape[0], dtype=np.int8), np.ones(Xt.shape[0], dtype=np.int8)])

    if model == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="liblinear"),
        )
    elif model == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=10,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown density-ratio classifier model {model!r}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        estimator.fit(X, y)
        prob = np.asarray(estimator.predict_proba(X), dtype=np.float64)[:, 1]
    train_auc = float(roc_auc_score(y, prob))
    source_w = density_ratio_from_classifier(
        estimator,
        Xs,
        n_source=Xs.shape[0],
        n_target=Xt.shape[0],
        probability_floor=probability_floor,
    )
    target_w = density_ratio_from_classifier(
        estimator,
        Xt,
        n_source=Xs.shape[0],
        n_target=Xt.shape[0],
        probability_floor=probability_floor,
    )
    # A single common normalization preserves target/source relative mass.
    # Separately forcing target weights to mean one breaks target-specific CP.
    normalization = float(np.mean(source_w))
    source_w = source_w / normalization
    target_w = target_w / normalization
    return DensityRatioResult(
        estimator=estimator,
        source_weights=source_w,
        target_weights=target_w,
        train_auc=train_auc,
        source_weight_summary=_weight_summary(source_w),
        target_weight_summary=_weight_summary(target_w),
        weight_normalization=normalization,
    )
