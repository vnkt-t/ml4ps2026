"""Localization metrics for score fields versus true pointwise error."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def _flatten_pair(score: np.ndarray, abs_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    if s.size != e.size:
        raise ValueError(f"score and abs_error sizes differ: {s.size} vs {e.size}")
    if s.size == 0:
        raise ValueError("score and abs_error must be nonempty")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(e)):
        raise ValueError("score and abs_error must be finite")
    if np.any(s < 0.0) or np.any(e < 0.0):
        raise ValueError("score and abs_error must be nonnegative magnitude fields")
    return s, e


def _top_membership(x: np.ndarray, q: float) -> np.ndarray:
    """Expected membership of the largest ceil(q*n) cells under uniform ties.

    Cells above the cutoff receive weight one, cells below zero, and tied cells
    split the remaining places uniformly. Weights sum to the integer top-set
    size. This prevents array-order artifacts for constant or quantized scores.
    """
    if not 0.0 < q <= 1.0 or x.size == 0:
        raise ValueError("q must be in (0,1] and x must be nonempty")
    k = max(1, int(np.ceil(q * x.size)))
    cutoff = np.partition(x, x.size - k)[x.size - k]
    weights = (x > cutoff).astype(np.float64)
    tied = x == cutoff
    weights[tied] = (k - float(weights.sum())) / int(tied.sum())
    return weights


def _top_overlap(score: np.ndarray, abs_error: np.ndarray, q: float) -> float:
    top_s = _top_membership(score, q)
    top_e = _top_membership(abs_error, q)
    return float(np.sum(top_s * top_e) / top_e.sum())


def _spearman(score: np.ndarray, abs_error: np.ndarray) -> float:
    if np.all(score == score[0]) or np.all(abs_error == abs_error[0]):
        return 0.0
    return float(spearmanr(score, abs_error).statistic)


def _high_error_auroc(score: np.ndarray, abs_error: np.ndarray, q: float) -> float:
    # The rank-sum formula is linear in binary target membership. Replacing it
    # by expected membership gives the exact expected AUROC over all equally
    # likely top-error sets, retaining the fixed class count k.
    membership = _top_membership(abs_error, q)
    k = int(np.ceil(q * abs_error.size))
    return (
        float((np.sum(membership * rankdata(score)) - k * (k + 1) / 2.0) / (k * (abs_error.size - k)))
        if k < abs_error.size else 0.5
    )


def localization_metrics_top10(score: np.ndarray, abs_error: np.ndarray) -> dict[str, float]:
    """Top-10% localization metrics, accepting fields or flattened vectors.

    Select exactly ceil(0.1*n) cells and average uniformly over cutoff ties in
    both scores and target errors. This matters for geometry baselines such as
    distance to the domain boundary. Keys match the submission experiments.
    """
    s, e = _flatten_pair(score, abs_error)
    top = _top_membership(s, 0.1)
    total = float(e.sum())
    return {
        "spearman": _spearman(s, e),
        "auroc": _high_error_auroc(s, e, 0.1),
        "error_capture_top10": float(np.sum(top * e) / total) if total > 0.0 else 0.0,
        "top10_overlap": _top_overlap(s, e, 0.1),
    }


def _center_of_mass(field: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    total = float(w.sum())
    if total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    H, W = w.shape
    x = (np.arange(H, dtype=np.float64) + 0.5) / H
    y = (np.arange(W, dtype=np.float64) + 0.5) / W
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.array([float((w * X).sum() / total), float((w * Y).sum() / total)])


def compute_metrics(score: np.ndarray, abs_error: np.ndarray) -> dict[str, float]:
    """Compute localization metrics for two equally shaped 2D magnitude fields.

    Top-q means exactly ceil(q*n) cells, with uniform random tie breaking
    integrated analytically. AUROC uses top-20% error cells and averages over
    cutoff ties. Constant fields have Spearman 0 by convention, constant scores
    AUROC 0.5 and chance-level top-q overlap/capture. With no error, capture is
    0 by convention; it has no localization interpretation.
    """
    score_field = np.asarray(score, dtype=np.float64)
    error_field = np.asarray(abs_error, dtype=np.float64)
    if score_field.ndim != 2 or score_field.shape != error_field.shape:
        raise ValueError("score and abs_error must have the same 2D field shape")
    s, e = _flatten_pair(score, abs_error)

    rho = _spearman(s, e)

    top20 = _top_membership(s, 0.2)
    total_error = float(e.sum())
    capture = float(np.sum(top20 * e) / total_error) if total_error > 0.0 else 0.0

    auc = _high_error_auroc(s, e, 0.2)

    com_dist = float(np.linalg.norm(_center_of_mass(score_field) - _center_of_mass(error_field)))

    return {
        "spearman": float(rho),
        "top10_overlap": float(_top_overlap(s, e, 0.1)),
        "top20_overlap": float(_top_overlap(s, e, 0.2)),
        "top30_overlap": float(_top_overlap(s, e, 0.3)),
        "error_capture": capture,
        "auroc": auc,
        "com_dist": com_dist,
    }
