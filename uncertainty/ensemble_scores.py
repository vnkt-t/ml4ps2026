"""Ensemble uncertainty score fields."""
from __future__ import annotations

import numpy as np


def ensemble_std(u_pred: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the ensemble standard deviation score over ``axis``.

    A single ensemble member is handled gracefully and returns zeros with the
    member axis removed.
    """
    arr = np.asarray(u_pred, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("u_pred must contain at least one ensemble axis and field axes")
    if arr.shape[axis] <= 1:
        return np.zeros(np.delete(np.array(arr.shape), axis).astype(int), dtype=np.float64)
    return np.std(arr, axis=axis)
