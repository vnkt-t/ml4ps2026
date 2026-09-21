"""Experiment 3: contrast sweep and norm-mismatch diagnostics.

For Family-C high-contrast solved PDE instances, evaluate whether raw residual
localization of pointwise |e| changes with contrast and whether |r| tracks
gradient/energy-style errors better. The predictor is deliberately controlled
and synthetic, not trained: ``synthetic_smooth`` = Gaussian-blurred truth plus
small smoothed noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".pytest_cache" / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".pytest_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generate_contrast import KAPPA_SWEEP, family_C_sample  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402
from solver.residual import residual  # noqa: E402
from uncertainty.compute_ladder import compute_ladder  # noqa: E402


PREDICTOR_LABEL = "synthetic_smooth"
LADDER_RUNG_MAP = {
    "V0 raw |r|": "1_abs_r",
    "V1 Jacobi-1": "3a_jacobi_1",
    "V2 Jacobi-5": "3c_jacobi_5",
    "V3 GS-5": "3g_gs_5",
    "V4 AMG/GS-40": "4_vcycle",
    "V5 oracle": "5_oracle",
}


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    stat = spearmanr(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)).statistic
    return float(stat) if np.isfinite(stat) else 0.0


def synthetic_smooth_predictor(
    u_true: np.ndarray,
    rng: np.random.Generator,
    blur_sigma: float = 1.0,
    noise_scale: float = 0.035,
) -> np.ndarray:
    """Controlled non-FNO predictor: blurred truth plus smooth scaled noise."""

    u = np.asarray(u_true, dtype=np.float64)
    smooth = gaussian_filter(u, sigma=blur_sigma, mode="nearest")
    noise = gaussian_filter(rng.standard_normal(u.shape), sigma=1.0, mode="nearest")
    noise = noise / (float(noise.std()) + 1e-12)
    scale = noise_scale * (float(np.std(u)) + 0.05 * float(np.ptp(u)) + 1e-12)
    return smooth + scale * noise


def grad_magnitude(field: np.ndarray) -> np.ndarray:
    N = field.shape[0]
    h = 1.0 / N
    gx, gy = np.gradient(np.asarray(field, dtype=np.float64), h, h, edge_order=1)
    return np.sqrt(gx * gx + gy * gy)


def energy_proxy(error: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    """Pointwise proxy whose sum relates to e^T A e up to signed local terms."""

    e = np.asarray(error, dtype=np.float64).reshape(-1)
    local = np.abs(e * np.asarray(r_vec, dtype=np.float64).reshape(-1))
    return np.sqrt(local).reshape(error.shape)


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _gate2_status(per_kappa: dict[str, dict]) -> dict[str, str | float | bool]:
    kappas = sorted((float(k) for k in per_kappa.keys()))
    if len(kappas) < 2:
        return {"status": "INCOMPLETE", "reason": "fewer than two kappa values completed"}

    abs_e = np.array([per_kappa[str(int(k))]["raw_residual_rho"]["abs_e"]["mean"] for k in kappas])
    grad = np.array([per_kappa[str(int(k))]["raw_residual_rho"]["grad_e"]["mean"] for k in kappas])
    energy = np.array([per_kappa[str(int(k))]["raw_residual_rho"]["energy_proxy"]["mean"] for k in kappas])
    logk = np.log10(kappas)

    abs_slope = float(np.polyfit(logk, abs_e, deg=1)[0])
    grad_margin = float(np.mean(grad - abs_e))
    energy_margin = float(np.mean(energy - abs_e))
    contrast_drop = bool(abs_e[-1] < abs_e[0] and abs_slope < 0.0)
    norm_higher = bool(max(grad_margin, energy_margin) > 0.02)

    if contrast_drop and norm_higher:
        status = "PROVISIONAL PASS"
    elif norm_higher:
        status = "MIXED: norm mismatch present, contrast degradation absent"
    elif contrast_drop:
        status = "MIXED: contrast degradation present, norm mismatch weak"
    else:
        status = "NO PASS"
    return {
        "status": status,
        "abs_e_slope_vs_log10_kappa": abs_slope,
        "abs_e_low_kappa_mean": float(abs_e[0]),
        "abs_e_high_kappa_mean": float(abs_e[-1]),
        "grad_minus_abs_e_mean": grad_margin,
        "energy_minus_abs_e_mean": energy_margin,
        "criterion": "PROVISIONAL PASS iff rho(|r|,|e|) drops from kappa min to max and grad/energy rho is higher on average by >0.02.",
    }


def _write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_path)


def _make_figure(per_kappa: dict[str, dict], out_path: Path) -> None:
    if not per_kappa:
        return
    kappa_labels = sorted(per_kappa.keys(), key=lambda x: float(x))
    kappas = np.array([float(k) for k in kappa_labels], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fig.suptitle("Figure 3: Contrast and Norm-Mismatch Diagnostics")

    norm_series = [
        ("abs_e", "|e|", "tab:blue"),
        ("grad_e", "|grad e|", "tab:orange"),
        ("energy_proxy", "energy proxy", "tab:green"),
    ]
    for key, label, color in norm_series:
        y = [per_kappa[k]["raw_residual_rho"][key]["mean"] for k in kappa_labels]
        axes[0].plot(kappas, y, marker="o", label=label, color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("contrast kappa")
    axes[0].set_ylabel("mean Spearman rho with raw |r|")
    axes[0].set_title("(a) Raw residual vs error norms")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(LADDER_RUNG_MAP)))
    for color, label in zip(colors, LADDER_RUNG_MAP):
        y = [per_kappa[k]["compute_ladder_rho"][label]["mean"] for k in kappa_labels]
        axes[1].plot(kappas, y, marker="o", label=label, color=color)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("contrast kappa")
    axes[1].set_ylabel("mean Spearman rho vs |e|")
    axes[1].set_title("(b) Compute-ladder rungs")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    out_json = REPO_ROOT / args.out_json
    out_fig = REPO_ROOT / args.out_fig

    payload: dict = {
        "experiment": "exp3_contrast_norm",
        "predictor": PREDICTOR_LABEL,
        "grid": args.N,
        "samples_per_kappa": args.samples,
        "seed": args.seed,
        "kappas_requested": [float(k) for k in args.kappas],
        "raw_residual_targets": ["abs_e", "grad_e", "energy_proxy"],
        "ladder_rungs": LADDER_RUNG_MAP,
        "per_kappa": {},
        "gate2": {"status": "INCOMPLETE"},
    }
    _write_json(payload, out_json)

    for kappa in args.kappas:
        raw_metrics = {"abs_e": [], "grad_e": [], "energy_proxy": []}
        ladder_metrics = {label: [] for label in LADDER_RUNG_MAP}
        identity_rel = []

        for i in range(args.samples):
            sample = family_C_sample(N=args.N, kappa=float(kappa), rng=rng)
            A = assemble(sample.a)
            u_pred = synthetic_smooth_predictor(sample.u_true, rng)
            e = u_pred - sample.u_true
            r_vec = residual(A, u_pred, sample.f)
            abs_r = np.abs(r_vec.reshape(args.N, args.N))

            raw_metrics["abs_e"].append(_rho(abs_r, np.abs(e)))
            raw_metrics["grad_e"].append(_rho(abs_r, grad_magnitude(e)))
            raw_metrics["energy_proxy"].append(_rho(abs_r, energy_proxy(e, r_vec)))

            Ae = A.dot(e.reshape(-1))
            denom = np.linalg.norm(r_vec) + 1e-14
            identity_rel.append(float(np.linalg.norm(Ae - r_vec) / denom))

            ladder = compute_ladder(
                A=A,
                u_hat=u_pred,
                f=sample.f,
                u_test=sample.u_true,
                u_pred_ensemble=None,
                sigma_ens=None,
            )
            abs_e = np.abs(e)
            for label, rung_key in LADDER_RUNG_MAP.items():
                ladder_metrics[label].append(_rho(np.asarray(ladder[rung_key]["score"]), abs_e))

            if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == args.samples:
                print(f"exp3 kappa={kappa:g}: {i + 1}/{args.samples}", flush=True)

        kappa_key = str(int(kappa)) if float(kappa).is_integer() else str(float(kappa))
        payload["per_kappa"][kappa_key] = {
            "n": args.samples,
            "raw_residual_rho": {key: _mean_std(vals) for key, vals in raw_metrics.items()},
            "compute_ladder_rho": {key: _mean_std(vals) for key, vals in ladder_metrics.items()},
            "residual_identity_rel_max": float(np.max(identity_rel)),
            "residual_identity_rel_mean": float(np.mean(identity_rel)),
        }
        payload["gate2"] = _gate2_status(payload["per_kappa"])
        _write_json(payload, out_json)
        _make_figure(payload["per_kappa"], out_fig)
        print(f"exp3 wrote partial results through kappa={kappa:g}", flush=True)

    payload["gate2"] = _gate2_status(payload["per_kappa"])
    _write_json(payload, out_json)
    _make_figure(payload["per_kappa"], out_fig)

    summary_rows = []
    for kappa_key, block in payload["per_kappa"].items():
        row = {"kappa": kappa_key}
        row.update({f"rho_raw_{k}": v["mean"] for k, v in block["raw_residual_rho"].items()})
        row.update({f"rho_{k}": v["mean"] for k, v in block["compute_ladder_rho"].items()})
        summary_rows.append(row)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"Gate 2: {payload['gate2']['status']}", flush=True)
    print(f"saved {out_json}", flush=True)
    print(f"saved {out_fig}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--kappas", type=float, nargs="+", default=list(KAPPA_SWEEP))
    parser.add_argument("--out-json", default="results/exp3_contrast_norm.json")
    parser.add_argument("--out-fig", default="figures/fig3_contrast_norm.png")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
