"""High-contrast coefficient families with solved Darcy targets.

Family C is the primary Tier-2 contrast family: random high-conductivity
inclusions and channels with a prescribed contrast kappa = a_max / a_min.
Family B is a checkerboard family for distribution-shift diagnostics. Support
relations depend on the source distribution.

Both public generators return exactly ``(a_field, f_rhs, u_true)`` where
``u_true`` solves ``A(a) u = f`` using the project finite-volume assembly.
The fixed source is independent of ``a`` to avoid the manufactured
``f=A@u_true`` interface-source confound documented in CLAUDE.md.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.assemble_fd import assemble, grid_centers  # noqa: E402
from solver.solve import solve  # noqa: E402


KAPPA_SWEEP = (2, 5, 10, 50, 100, 500)


@dataclass(frozen=True)
class ContrastSample:
    """Solved high-contrast sample plus optional diagnostic masks."""

    a: np.ndarray
    f: np.ndarray
    u_true: np.ndarray
    interface_mask: np.ndarray
    meta: dict


def default_rhs(N: int) -> np.ndarray:
    """Fixed smooth-ish positive source used for solved-data experiments."""

    X, Y = grid_centers(N)
    return 1.0 + 0.25 * np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)


def interface_mask(a: np.ndarray) -> np.ndarray:
    """Mark cells adjacent to a coefficient jump."""

    arr = np.asarray(a)
    mask = np.zeros(arr.shape, dtype=bool)
    mask[:-1, :] |= arr[:-1, :] != arr[1:, :]
    mask[1:, :] |= arr[:-1, :] != arr[1:, :]
    mask[:, :-1] |= arr[:, :-1] != arr[:, 1:]
    mask[:, 1:] |= arr[:, :-1] != arr[:, 1:]
    return mask


def _solve_sample(a: np.ndarray, f: np.ndarray) -> np.ndarray:
    N = a.shape[0]
    return solve(assemble(a), f).reshape(N, N)


def _draw_circle(a: np.ndarray, center: tuple[float, float], radius: float, value: float) -> None:
    N = a.shape[0]
    X, Y = grid_centers(N)
    cx, cy = center
    a[(X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2] = value


def _draw_rectangle(
    a: np.ndarray,
    center: tuple[float, float],
    half_width: float,
    half_height: float,
    value: float,
) -> None:
    X, Y = grid_centers(a.shape[0])
    cx, cy = center
    a[(np.abs(X - cx) <= half_width) & (np.abs(Y - cy) <= half_height)] = value


def family_C_field(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    a_min: float = 1.0,
    *,
    n_incl_range: tuple[int, int] = (2, 6),
    radius_range: tuple[float, float] = (0.055, 0.14),
    rect_hw_range: tuple[float, float] = (0.045, 0.12),
    rect_hh_range: tuple[float, float] = (0.045, 0.16),
    circle_prob: float = 0.55,
    channel_prob: float = 0.65,
) -> tuple[np.ndarray, dict]:
    """Random high-contrast inclusions/channels coefficient field.

    The field is binary with values ``a_min`` and ``a_min*kappa``. Inclusions
    are always present; a high-conductivity channel is added with probability
    ``channel_prob`` to cover both requested morphologies.

    The geometry knobs (``n_incl_range``/``radius_range``/...) default to the
    original Family-C distribution. Changing inclusion geometry at fixed
    ``kappa``/``a_min`` defines an in-family geometry shift with the same two
    coefficient values. Matching extrema does not establish target support
    containment or conditional invariance: wider count/radius ranges can extend
    the latent geometry support. No weighted-conformal guarantee follows from
    these generator settings alone.
    """

    if rng is None:
        rng = np.random.default_rng()
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1, got {kappa}")

    a_high = float(a_min) * float(kappa)
    a = np.full((N, N), float(a_min), dtype=np.float64)

    n_inclusions = int(rng.integers(n_incl_range[0], n_incl_range[1]))
    inclusions: list[dict[str, float | str]] = []
    for _ in range(n_inclusions):
        cx = float(rng.uniform(0.15, 0.85))
        cy = float(rng.uniform(0.15, 0.85))
        if rng.random() < circle_prob:
            radius = float(rng.uniform(radius_range[0], radius_range[1]))
            _draw_circle(a, (cx, cy), radius, a_high)
            inclusions.append({"shape": "circle", "cx": cx, "cy": cy, "radius": radius})
        else:
            hw = float(rng.uniform(rect_hw_range[0], rect_hw_range[1]))
            hh = float(rng.uniform(rect_hh_range[0], rect_hh_range[1]))
            _draw_rectangle(a, (cx, cy), hw, hh, a_high)
            inclusions.append({"shape": "rectangle", "cx": cx, "cy": cy, "hw": hw, "hh": hh})

    channel_meta: dict[str, float | str] | None = None
    if rng.random() < channel_prob:
        X, Y = grid_centers(N)
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        center = float(rng.uniform(0.22, 0.78))
        width = float(rng.uniform(0.045, 0.095))
        if orientation == "vertical":
            wiggle = 0.04 * np.sin(2.0 * np.pi * Y + float(rng.uniform(0.0, 2.0 * np.pi)))
            a[np.abs(X - center - wiggle) <= width] = a_high
        else:
            wiggle = 0.04 * np.sin(2.0 * np.pi * X + float(rng.uniform(0.0, 2.0 * np.pi)))
            a[np.abs(Y - center - wiggle) <= width] = a_high
        channel_meta = {"orientation": orientation, "center": center, "width": width}

    meta = {
        "family": "C",
        "N": N,
        "kappa": float(kappa),
        "a_min": float(a_min),
        "a_max": a_high,
        "n_inclusions": n_inclusions,
        "inclusions": inclusions,
        "channel": channel_meta,
    }
    return a, meta


def family_C_sample(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    a_min: float = 1.0,
    **field_kwargs,
) -> ContrastSample:
    """Return a solved Family-C sample with diagnostics.

    Extra keyword args (``n_incl_range``, ``radius_range``, ``channel_prob``, ...)
    are forwarded to :func:`family_C_field` to specify an in-family geometry shift.
    """

    a, meta = family_C_field(N=N, kappa=kappa, rng=rng, a_min=a_min, **field_kwargs)
    f = default_rhs(N)
    u_true = _solve_sample(a, f)
    return ContrastSample(a=a, f=f, u_true=u_true, interface_mask=interface_mask(a), meta=meta)


def family_C(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    a_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a_field, f_rhs, u_true)`` for Family C."""

    sample = family_C_sample(N=N, kappa=kappa, rng=rng, a_min=a_min)
    return sample.a, sample.f, sample.u_true


def family_B_field(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    block_size: int | None = None,
    a_min: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Binary checkerboard coefficient field, with two phases per block size."""

    if rng is None:
        rng = np.random.default_rng()
    if block_size is None:
        block_size = int(rng.choice(np.array([2, 4, 8], dtype=int)))
    if block_size <= 0 or N % block_size != 0:
        raise ValueError(f"block_size must divide N={N}; got {block_size}")

    ii = np.arange(N)[:, None] // block_size
    jj = np.arange(N)[None, :] // block_size
    phase = int(rng.integers(0, 2))
    high = ((ii + jj + phase) % 2) == 0

    a_high = float(a_min) * float(kappa)
    a = np.full((N, N), float(a_min), dtype=np.float64)
    a[high] = a_high
    meta = {
        "family": "B",
        "N": N,
        "kappa": float(kappa),
        "a_min": float(a_min),
        "a_max": a_high,
        "block_size": int(block_size),
        "phase": phase,
    }
    return a, meta


def family_B_sample(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    block_size: int | None = None,
    a_min: float = 1.0,
) -> ContrastSample:
    """Return a solved checkerboard Family-B sample with diagnostics."""

    a, meta = family_B_field(N=N, kappa=kappa, rng=rng, block_size=block_size, a_min=a_min)
    f = default_rhs(N)
    u_true = _solve_sample(a, f)
    return ContrastSample(a=a, f=f, u_true=u_true, interface_mask=interface_mask(a), meta=meta)


def family_B(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    block_size: int | None = None,
    a_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a_field, f_rhs, u_true)`` for checkerboard Family B."""

    sample = family_B_sample(N=N, kappa=kappa, rng=rng, block_size=block_size, a_min=a_min)
    return sample.a, sample.f, sample.u_true


def _plot_examples() -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".pytest_cache" / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".pytest_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260606)
    examples = [("Family C inclusions/channels", family_C_sample(N=32, kappa=100, rng=rng))]
    examples.append(("Family B checkerboard", family_B_sample(N=32, kappa=100, rng=rng, block_size=4)))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for row, (title, sample) in enumerate(examples):
        images = [
            (sample.a, "a(x)", "viridis"),
            (sample.f, "f(x)", "magma"),
            (sample.u_true, "u_true", "coolwarm"),
        ]
        for col, (field, label, cmap) in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(field, origin="lower", cmap=cmap)
            ax.set_title(f"{title}: {label}" if col == 0 else label)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = REPO_ROOT / "figures" / "generate_contrast_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = _plot_examples()
    print(f"saved {path}")
