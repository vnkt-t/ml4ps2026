"""Dataset split helper for train/val/calibration/test/OOD partitions."""
from __future__ import annotations

from collections.abc import Sized

import numpy as np

DEFAULT_FRACTIONS = {
    "train": 0.6,
    "val": 0.1,
    "calibration": 0.1,
    "test": 0.1,
    "OOD": 0.1,
}


def _length(data_or_n: int | Sized) -> int:
    if isinstance(data_or_n, (int, np.integer)):
        return int(data_or_n)
    return len(data_or_n)


def make_splits(
    data_or_n: int | Sized,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    """Return disjoint split indices.

    Fractions default to 0.6/0.1/0.1/0.1/0.1 for
    train/val/calibration/test/OOD and must sum to 1.
    """
    n = _length(data_or_n)
    if n < 0:
        raise ValueError("dataset length must be nonnegative")

    fracs = dict(DEFAULT_FRACTIONS if fractions is None else fractions)
    missing = set(DEFAULT_FRACTIONS) - set(fracs)
    if missing:
        raise ValueError(f"missing split fractions: {sorted(missing)}")
    if any(v < 0 for v in fracs.values()):
        raise ValueError("split fractions must be nonnegative")
    total = float(sum(fracs.values()))
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"split fractions must sum to 1; got {total}")

    names = list(DEFAULT_FRACTIONS)
    raw = np.array([fracs[name] * n for name in names], dtype=np.float64)
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    indices = np.arange(n, dtype=int)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    out: dict[str, np.ndarray] = {}
    start = 0
    for name, count in zip(names, counts):
        out[name] = indices[start : start + int(count)]
        start += int(count)
    return out
