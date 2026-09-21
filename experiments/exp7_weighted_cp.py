"""Experiment 7: weighted conformal prediction under covariate shift.

This is a synthetic-predictor diagnostic. The source calibration distribution
is a broad smooth log-GRF family. Family R is a rough/short-lengthscale subset
of that same support, where density-ratio weighting is meaningful. Family B is
the checkerboard support-breaking family, where weighting cannot repair missing
calibration support and coverage is expected to fail.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".pytest_cache" / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".pytest_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generate_contrast import default_rhs, family_B_field  # noqa: E402
from data.generate_grf import grf_coeff  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402
from solver.solve import solve  # noqa: E402
from uncertainty.conformal import conformal_quantile  # noqa: E402
from uncertainty.weighted_conformal import (  # noqa: E402
    calibrate_weighted,
    coverage_from_scores,
    fit_density_ratio_classifier,
)


PREDICTOR_LABEL = "synthetic_roughness_sensitive_smooth_surrogate"
FEATURE_NAMES = [
    "log_a_mean",
    "log_a_std",
    "log_a_min",
    "log_a_max",
    "log_a_p10",
    "log_a_p90",
    "rough_l1",
    "rough_l2",
    "rough_p95",
    "laplace_l1",
    "hf_energy",
    "log_a_contrast",
]


def _write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_path)


def _roughness_components(loga: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.diff(loga, axis=0)
    dy = np.diff(loga, axis=1)
    return dx, dy


def coefficient_features(a: np.ndarray) -> np.ndarray:
    """Low-dimensional covariate summary used by the ratio classifier."""

    arr = np.asarray(a, dtype=np.float64)
    loga = np.log(np.maximum(arr, 1e-12))
    dx, dy = _roughness_components(loga)
    rough_abs = np.concatenate([np.abs(dx).reshape(-1), np.abs(dy).reshape(-1)])
    rough_sq = np.concatenate([(dx * dx).reshape(-1), (dy * dy).reshape(-1)])
    lap = (
        -4.0 * loga
        + np.roll(loga, 1, axis=0)
        + np.roll(loga, -1, axis=0)
        + np.roll(loga, 1, axis=1)
        + np.roll(loga, -1, axis=1)
    )
    low = gaussian_filter(loga, sigma=2.0, mode="nearest")
    high = loga - low
    return np.array(
        [
            float(np.mean(loga)),
            float(np.std(loga)),
            float(np.min(loga)),
            float(np.max(loga)),
            float(np.quantile(loga, 0.10)),
            float(np.quantile(loga, 0.90)),
            float(np.mean(rough_abs)),
            float(np.sqrt(np.mean(rough_sq))),
            float(np.quantile(rough_abs, 0.95)),
            float(np.mean(np.abs(lap))),
            float(np.mean(high * high)),
            float(np.log(np.max(arr) / max(float(np.min(arr)), 1e-12))),
        ],
        dtype=np.float64,
    )


def _feature_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(X, dtype=np.float64), columns=FEATURE_NAMES)


def _grf_field(
    N: int,
    rng: np.random.Generator,
    lengthscale_range: tuple[float, float],
    sigma_range: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | str]]:
    lengthscale = float(rng.uniform(*lengthscale_range))
    sigma = float(rng.uniform(*sigma_range))
    a = grf_coeff(N, rng, lengthscale=lengthscale, sigma=sigma)
    return a, {"family": "smooth_grf", "lengthscale": lengthscale, "sigma": sigma}


def _solve_field(a: np.ndarray, f: np.ndarray) -> np.ndarray:
    N = a.shape[0]
    return solve(assemble(a), f).reshape(N, N)


def _roughness_index(a: np.ndarray) -> float:
    loga = np.log(np.maximum(np.asarray(a, dtype=np.float64), 1e-12))
    dx, dy = _roughness_components(loga)
    return float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))


def synthetic_surrogate_prediction(
    u_true: np.ndarray,
    a: np.ndarray,
    rng: np.random.Generator,
    *,
    support: str,
) -> np.ndarray:
    """Controlled predictor used only for this diagnostic experiment."""

    u = np.asarray(u_true, dtype=np.float64)
    rough = _roughness_index(a)
    field_scale = float(np.std(u) + 0.05 * np.ptp(u) + 1e-12)
    noise = gaussian_filter(rng.standard_normal(u.shape), sigma=1.0, mode="nearest")
    noise = noise / (float(np.std(noise)) + 1e-12)

    if support == "checkerboard":
        blurred = gaussian_filter(u, sigma=2.6, mode="nearest")
        bias = 0.70 * (blurred - u)
        noise_scale = (0.030 + 0.060 * min(rough, 2.0)) * field_scale
        return u + bias + noise_scale * noise

    blur_sigma = 0.35 + 0.45 * min(rough, 0.75)
    blurred = gaussian_filter(u, sigma=blur_sigma, mode="nearest")
    bias = 0.16 * min(rough, 0.75) * (blurred - u)
    noise_scale = (0.008 + 0.045 * min(rough, 0.75)) * field_scale
    return u + bias + noise_scale * noise


def _score_from_prediction(u_true: np.ndarray, u_hat: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(np.asarray(u_hat, dtype=np.float64) - np.asarray(u_true, dtype=np.float64))
    denom = float(np.linalg.norm(u_true.reshape(-1)) + 1e-12)
    return {
        "max_abs": float(np.max(abs_error)),
        "mean_abs": float(np.mean(abs_error)),
        "rel_l2": float(np.linalg.norm(abs_error.reshape(-1)) / denom),
    }


def _build_source_calibration(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, np.ndarray]:
    f = default_rhs(args.N)
    X = np.empty((args.n_calib, len(FEATURE_NAMES)), dtype=np.float64)
    scores = np.empty(args.n_calib, dtype=np.float64)
    mean_abs = np.empty(args.n_calib, dtype=np.float64)
    rel_l2 = np.empty(args.n_calib, dtype=np.float64)
    roughness = np.empty(args.n_calib, dtype=np.float64)
    lengthscale = np.empty(args.n_calib, dtype=np.float64)

    for i in range(args.n_calib):
        a, meta = _grf_field(args.N, rng, args.source_lengthscale, args.source_sigma)
        u = _solve_field(a, f)
        u_hat = synthetic_surrogate_prediction(u, a, rng, support="smooth_grf")
        score = _score_from_prediction(u, u_hat)
        X[i] = coefficient_features(a)
        scores[i] = score["max_abs"]
        mean_abs[i] = score["mean_abs"]
        rel_l2[i] = score["rel_l2"]
        roughness[i] = _roughness_index(a)
        lengthscale[i] = float(meta["lengthscale"])
        if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == args.n_calib:
            print(f"exp7 source calibration: {i + 1}/{args.n_calib}", flush=True)

    return {
        "X": X,
        "scores": scores,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "roughness": roughness,
        "lengthscale": lengthscale,
    }


def _target_feature_block(
    args: argparse.Namespace,
    rng: np.random.Generator,
    *,
    family: str,
    n: int,
) -> np.ndarray:
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i in range(n):
        if family == "R":
            a, _ = _grf_field(args.N, rng, args.rough_lengthscale, args.rough_sigma)
        elif family == "B":
            a, _ = family_B_field(args.N, kappa=args.b_kappa, rng=rng)
        else:
            raise ValueError(f"unknown target family {family}")
        X[i] = coefficient_features(a)
    return X


def _build_target_scores(
    args: argparse.Namespace,
    rng: np.random.Generator,
    *,
    family: str,
    n: int,
) -> dict[str, np.ndarray]:
    f = default_rhs(args.N)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    scores = np.empty(n, dtype=np.float64)
    mean_abs = np.empty(n, dtype=np.float64)
    rel_l2 = np.empty(n, dtype=np.float64)
    roughness = np.empty(n, dtype=np.float64)
    lengthscale = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if family == "R":
            a, meta = _grf_field(args.N, rng, args.rough_lengthscale, args.rough_sigma)
            support = "smooth_grf"
            lengthscale[i] = float(meta["lengthscale"])
        elif family == "B":
            a, _ = family_B_field(args.N, kappa=args.b_kappa, rng=rng)
            support = "checkerboard"
        else:
            raise ValueError(f"unknown target family {family}")
        u = _solve_field(a, f)
        u_hat = synthetic_surrogate_prediction(u, a, rng, support=support)
        score = _score_from_prediction(u, u_hat)
        X[i] = coefficient_features(a)
        scores[i] = score["max_abs"]
        mean_abs[i] = score["mean_abs"]
        rel_l2[i] = score["rel_l2"]
        roughness[i] = _roughness_index(a)
        if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == n:
            print(f"exp7 target {family}: {i + 1}/{n}", flush=True)

    return {
        "X": X,
        "scores": scores,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "roughness": roughness,
        "lengthscale": lengthscale,
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "p10": float(np.quantile(finite, 0.10)) if finite.size else float("nan"),
        "p50": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
        "p90": float(np.quantile(finite, 0.90)) if finite.size else float("nan"),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
    }


def _evaluate_family(
    name: str,
    source: dict[str, np.ndarray],
    target_ratio_X: np.ndarray,
    target_test: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict:
    ratio = fit_density_ratio_classifier(
        source["X"],
        target_ratio_X,
        random_state=args.seed + (11 if name == "Family_R" else 29),
        model=args.ratio_model,
    )
    unweighted_q = conformal_quantile(source["scores"], args.alpha)
    weighted_cal = calibrate_weighted(
        source["scores"],
        ratio.source_weights,
        args.alpha,
        method=f"weighted_{name}",
        clip_quantile=args.weight_clip_quantile,
        test_weight=args.test_weight,
    )
    unweighted_metrics = coverage_from_scores(target_test["scores"], unweighted_q)
    weighted_metrics = coverage_from_scores(target_test["scores"], weighted_cal.qhat)

    return {
        "family": name,
        "unweighted": {
            "qhat": float(unweighted_q),
            "coverage": unweighted_metrics,
        },
        "weighted": {
            "qhat": float(weighted_cal.qhat),
            "coverage": weighted_metrics,
            "calibration": weighted_cal.__dict__,
        },
        "density_ratio": {
            "model": args.ratio_model,
            "classifier_train_auc": float(ratio.train_auc),
            "source_weight_summary": ratio.source_weight_summary,
            "target_weight_summary": ratio.target_weight_summary,
        },
        "target_score_summary": coverage_from_scores(target_test["scores"], float("inf")),
        "target_roughness": _summary_stats(target_test["roughness"]),
        "target_lengthscale": _summary_stats(target_test["lengthscale"]),
        "source_feature_weighted_means": {
            name_: float(np.average(source["X"][:, j], weights=ratio.source_weights))
            for j, name_ in enumerate(FEATURE_NAMES)
        },
    }


def _make_figure(
    payload: dict,
    source_scores: np.ndarray,
    family_R_scores: np.ndarray,
    family_B_scores: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    fig.suptitle("Experiment 7: Weighted CP Works On-Support, Fails Off-Support")

    rows = []
    for family_key, label in [("Family_R", "Family R"), ("Family_B", "Family B")]:
        block = payload["families"][family_key]
        rows.append(
            {
                "family": label,
                "method": "unweighted",
                "coverage": block["unweighted"]["coverage"]["coverage"],
            }
        )
        rows.append(
            {
                "family": label,
                "method": "weighted",
                "coverage": block["weighted"]["coverage"]["coverage"],
            }
        )
    cov = pd.DataFrame(rows)
    x = np.arange(2)
    width = 0.35
    methods = [("unweighted", "tab:gray"), ("weighted", "tab:blue")]
    for offset, (method, color) in zip([-width / 2, width / 2], methods):
        y = [
            float(cov[(cov["family"] == family) & (cov["method"] == method)]["coverage"].iloc[0])
            for family in ["Family R", "Family B"]
        ]
        axes[0].bar(x + offset, y, width=width, label=method, color=color)
    axes[0].axhline(payload["target_coverage"], color="black", linestyle="--", linewidth=1, label="nominal")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Family R\nroughness shift", "Family B\ncheckerboard"])
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("empirical fieldwise coverage")
    axes[0].set_title("(a) Coverage at alpha=0.1")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    box_data = [source_scores, family_R_scores, family_B_scores]
    labels = ["source\ncalib", "Family R\ntest", "Family B\ntest"]
    axes[1].boxplot(box_data, tick_labels=labels, showfliers=False, patch_artist=True)
    axes[1].axhline(payload["families"]["Family_R"]["unweighted"]["qhat"], color="tab:gray", linestyle="-", linewidth=1.5, label="unweighted q")
    axes[1].axhline(payload["families"]["Family_R"]["weighted"]["qhat"], color="tab:blue", linestyle="-", linewidth=1.5, label="weighted q for R")
    axes[1].axhline(payload["families"]["Family_B"]["weighted"]["qhat"], color="tab:red", linestyle="--", linewidth=1.5, label="weighted q for B")
    axes[1].set_ylabel("max absolute error score")
    axes[1].set_title("(b) Score support mismatch")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    out_json = REPO_ROOT / args.out_json
    out_fig = REPO_ROOT / args.out_fig

    source = _build_source_calibration(args, rng)
    target_R_ratio_X = _target_feature_block(args, rng, family="R", n=args.n_ratio)
    target_B_ratio_X = _target_feature_block(args, rng, family="B", n=args.n_ratio)
    target_R = _build_target_scores(args, rng, family="R", n=args.n_test)
    target_B = _build_target_scores(args, rng, family="B", n=args.n_test)

    family_R = _evaluate_family("Family_R", source, target_R_ratio_X, target_R, args)
    family_B = _evaluate_family("Family_B", source, target_B_ratio_X, target_B, args)

    payload = {
        "experiment": "exp7_weighted_cp",
        "predictor": PREDICTOR_LABEL,
        "note": (
            "Synthetic-predictor diagnostic. Family R is a roughness/lengthscale shift inside "
            "the broad smooth-GRF source support. Family B is checkerboard support-breaking; "
            "weighted CP has no support guarantee there and the coverage failure is expected."
        ),
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "grid": args.N,
        "seed": args.seed,
        "n_calib": args.n_calib,
        "n_ratio": args.n_ratio,
        "n_test": args.n_test,
        "score": "fieldwise max_abs_error",
        "feature_names": FEATURE_NAMES,
        "source_distribution": {
            "family": "smooth_log_grf",
            "lengthscale_uniform": list(args.source_lengthscale),
            "sigma_uniform": list(args.source_sigma),
            "score_summary": coverage_from_scores(source["scores"], float("inf")),
            "roughness": _summary_stats(source["roughness"]),
            "lengthscale": _summary_stats(source["lengthscale"]),
        },
        "target_distributions": {
            "Family_R": {
                "family": "smooth_log_grf",
                "lengthscale_uniform": list(args.rough_lengthscale),
                "sigma_uniform": list(args.rough_sigma),
            },
            "Family_B": {
                "family": "checkerboard",
                "kappa": args.b_kappa,
            },
        },
        "families": {
            "Family_R": family_R,
            "Family_B": family_B,
        },
        "interpretation": {
            "Family_R": "Weighted split conformal should be meaningful because target rough fields lie inside source GRF support.",
            "Family_B": "Failure is expected: checkerboard coefficients are off support, so classifier ratios cannot create calibration scores for unseen error modes.",
        },
    }
    _write_json(payload, out_json)
    _make_figure(payload, source["scores"], target_R["scores"], target_B["scores"], out_fig)

    rows = []
    for family_key in ["Family_R", "Family_B"]:
        block = payload["families"][family_key]
        rows.append(
            {
                "family": family_key,
                "unweighted_cov": block["unweighted"]["coverage"]["coverage"],
                "weighted_cov": block["weighted"]["coverage"]["coverage"],
                "unweighted_q": block["unweighted"]["qhat"],
                "weighted_q": block["weighted"]["qhat"],
                "ratio_auc": block["density_ratio"]["classifier_train_auc"],
                "ess": block["weighted"]["calibration"]["effective_sample_size"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"saved {out_json}", flush=True)
    print(f"saved {out_fig}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--n-calib", type=int, default=350)
    parser.add_argument("--n-ratio", type=int, default=250)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=202606067)
    parser.add_argument("--source-lengthscale", type=float, nargs=2, default=(0.035, 0.18))
    parser.add_argument("--rough-lengthscale", type=float, nargs=2, default=(0.035, 0.065))
    parser.add_argument("--source-sigma", type=float, nargs=2, default=(0.75, 1.25))
    parser.add_argument("--rough-sigma", type=float, nargs=2, default=(0.90, 1.25))
    parser.add_argument("--b-kappa", type=float, default=100.0)
    parser.add_argument("--ratio-model", choices=["logistic", "gradient_boosting"], default="logistic")
    parser.add_argument("--weight-clip-quantile", type=float, default=0.995)
    parser.add_argument("--test-weight", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--out-json", default="results/exp7_weighted_cp.json")
    parser.add_argument("--out-fig", default="results/figures/exp7_weighted_cp.png")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
