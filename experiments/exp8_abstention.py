"""Experiment 8: solver-handoff Pareto abstention curves."""
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

from data.generate_contrast import family_C_sample  # noqa: E402
from experiments.exp3_contrast_norm import synthetic_smooth_predictor  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402
from solver.multigrid import vcycle, vcycle_backend  # noqa: E402
from solver.relaxation import jacobi_sweep  # noqa: E402
from uncertainty.residual_scores import residual_vector  # noqa: E402


PREDICTOR_LABEL_REAL = "real_tiny_fno_ensemble_kappa100"
PREDICTOR_LABEL_FALLBACK = "synthetic_smooth_ensemble_fallback"


def _write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_path)


def _sample_first_predictions(u_pred: np.ndarray, n_test: int, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(u_pred, dtype=np.float64)
    if arr.ndim == 4:
        if arr.shape[0] == n_test and arr.shape[2:] == shape:
            return arr[:, None, :, :]
        if arr.shape[1] == n_test and arr.shape[2:] == shape:
            return np.moveaxis(arr, 0, 1)
    if arr.ndim == 3 and arr.shape[0] == n_test and arr.shape[1:] == shape:
        return arr[:, None, :, :]
    raise ValueError(f"unrecognized u_pred shape {arr.shape} for n_test={n_test}, field={shape}")


def _synthetic_ensemble(u_true: np.ndarray, rng: np.random.Generator, members: int) -> np.ndarray:
    preds = []
    for _ in range(members):
        blur = float(np.clip(1.0 + 0.15 * rng.standard_normal(), 0.6, 1.5))
        noise = float(np.clip(0.035 * (1.0 + 0.2 * rng.standard_normal()), 0.015, 0.07))
        preds.append(synthetic_smooth_predictor(u_true, rng, blur_sigma=blur, noise_scale=noise))
    return np.stack(preds, axis=0)


def _load_real_or_fallback(args: argparse.Namespace) -> dict[str, np.ndarray | str]:
    in_path = REPO_ROOT / args.input
    if in_path.exists():
        npz = np.load(in_path, allow_pickle=False)
        a_test = np.asarray(npz["a_test"], dtype=np.float64)
        u_test = np.asarray(npz["u_test"], dtype=np.float64)
        n_test, H, W = u_test.shape
        pred = _sample_first_predictions(np.asarray(npz["u_pred"]), n_test, (H, W))
        sigma = np.asarray(npz["sigma_ens"], dtype=np.float64) if "sigma_ens" in npz.files else pred.std(axis=1)
        if "f_test" in npz.files:
            f_test = np.asarray(npz["f_test"], dtype=np.float64)
        else:
            f_source = npz["f_source"].item() if "f_source" in npz.files else 1.0
            f_test = np.full((n_test, H, W), float(f_source), dtype=np.float64)
        return {
            "a": a_test,
            "u_true": u_test,
            "pred": pred,
            "sigma": sigma,
            "f": f_test,
            "predictor": PREDICTOR_LABEL_REAL,
            "input": str(in_path),
            "note": "Loaded saved real-FNO ensemble artifact.",
        }

    rng = np.random.default_rng(args.seed)
    a_all = np.empty((args.synthetic_samples, args.N, args.N), dtype=np.float64)
    u_all = np.empty_like(a_all)
    f_all = np.empty_like(a_all)
    pred_all = np.empty((args.synthetic_samples, args.members, args.N, args.N), dtype=np.float64)
    sigma_all = np.empty_like(a_all)
    for i in range(args.synthetic_samples):
        sample = family_C_sample(N=args.N, kappa=args.kappa, rng=rng)
        ensemble = _synthetic_ensemble(sample.u_true, rng, args.members)
        a_all[i] = sample.a
        u_all[i] = sample.u_true
        f_all[i] = sample.f
        pred_all[i] = ensemble
        sigma_all[i] = ensemble.std(axis=0)
    return {
        "a": a_all,
        "u_true": u_all,
        "pred": pred_all,
        "sigma": sigma_all,
        "f": f_all,
        "predictor": PREDICTOR_LABEL_FALLBACK,
        "input": str(in_path),
        "note": "Input NPZ was missing; used a clearly labeled synthetic smooth ensemble fallback.",
    }


def _field_stats(prefix: str, field: np.ndarray) -> dict[str, float]:
    arr = np.asarray(field, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_l2": float(np.linalg.norm(arr) / np.sqrt(arr.size)),
    }


def _build_feature_table(block: dict[str, np.ndarray | str], args: argparse.Namespace) -> pd.DataFrame:
    a_all = np.asarray(block["a"], dtype=np.float64)
    u_all = np.asarray(block["u_true"], dtype=np.float64)
    pred_all = np.asarray(block["pred"], dtype=np.float64)
    sigma_all = np.asarray(block["sigma"], dtype=np.float64)
    f_all = np.asarray(block["f"], dtype=np.float64)

    n_samples = int(a_all.shape[0])
    max_samples = n_samples if args.max_samples is None else min(args.max_samples, n_samples)
    rows: list[dict[str, float | int | str]] = []
    backend = vcycle_backend()

    for i in range(max_samples):
        A = assemble(a_all[i])
        ensemble = pred_all[i]
        u_hat = ensemble.mean(axis=0)
        err = u_hat - u_all[i]
        abs_error = np.abs(err)
        denom = float(np.linalg.norm(u_all[i].reshape(-1)) + 1e-12)
        r_vec = residual_vector(A, u_hat, f_all[i])
        abs_r = np.abs(r_vec.reshape(u_hat.shape))
        smooth_r = gaussian_filter(abs_r, sigma=1.0, mode="nearest")
        e0 = np.zeros_like(u_hat, dtype=np.float64)
        jac20 = np.abs(jacobi_sweep(A, r_vec, e0, 20).reshape(u_hat.shape))
        rung4 = np.abs(vcycle(A, r_vec).reshape(u_hat.shape))

        row: dict[str, float | int | str] = {
            "sample": i,
            "rel_l2_error": float(np.linalg.norm(err.reshape(-1)) / denom),
            "mean_abs_error": float(np.mean(abs_error)),
            "max_abs_error": float(np.max(abs_error)),
            "rung4_backend": backend,
        }
        row.update(_field_stats("ens_sigma", sigma_all[i]))
        row.update(_field_stats("res_abs", abs_r))
        row.update(_field_stats("res_smooth", smooth_r))
        row.update(_field_stats("ladder_jac20", jac20))
        row.update(_field_stats("ladder_rung4", rung4))
        row["ladder_rung4_to_res_p95"] = float(
            np.clip(float(row["ladder_rung4_p95"]) / max(float(row["res_abs_p95"]), 1e-12), 0.0, 1e6)
        )
        row["ladder_jac20_to_res_p95"] = float(
            np.clip(float(row["ladder_jac20_p95"]) / max(float(row["res_abs_p95"]), 1e-12), 0.0, 1e6)
        )
        rows.append(row)

        if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == max_samples:
            print(f"exp8 features: {i + 1}/{max_samples}", flush=True)

    return pd.DataFrame(rows)


def _split_indices(n: int, seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    ensemble_cols = [c for c in df.columns if c.startswith("ens_sigma_")]
    residual_cols = [c for c in df.columns if c.startswith("res_abs_") or c.startswith("res_smooth_")]
    ladder_cols = [c for c in df.columns if c.startswith("ladder_")]
    combined_cols = ensemble_cols + residual_cols + ladder_cols
    return ensemble_cols, combined_cols


def _fit_risk_scores(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cols: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = df.iloc[train_idx][cols].to_numpy(dtype=np.float64)
    y_train = df.iloc[train_idx]["rel_l2_error"].to_numpy(dtype=np.float64)
    X_test = df.iloc[test_idx][cols].to_numpy(dtype=np.float64)
    y_test = df.iloc[test_idx]["rel_l2_error"].to_numpy(dtype=np.float64)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    X_train = np.log1p(np.clip(X_train, 0.0, 1e6))
    X_test = np.log1p(np.clip(X_test, 0.0, 1e6))

    if args.risk_model == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
    elif args.risk_model == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=6,
            random_state=args.seed,
        )
    else:
        raise ValueError(f"unknown risk model {args.risk_model!r}")
    model.fit(X_train, y_train)
    score = np.asarray(model.predict(X_test), dtype=np.float64)
    diagnostics = {
        "mae": float(mean_absolute_error(y_test, score)),
        "r2": float(r2_score(y_test, score)),
    }
    return score, diagnostics


def _pareto_curve(risk_score: np.ndarray, high_error: np.ndarray) -> pd.DataFrame:
    scores = np.asarray(risk_score, dtype=np.float64).reshape(-1)
    high = np.asarray(high_error, dtype=bool).reshape(-1)
    if scores.size != high.size:
        raise ValueError(f"risk_score and high_error sizes differ: {scores.size} vs {high.size}")
    order = np.argsort(-scores)
    n = scores.size
    total_high = int(high.sum())
    rows = []
    handed = np.zeros(n, dtype=bool)
    for k in range(n + 1):
        if k > 0:
            handed[order[k - 1]] = True
        accepted = ~handed
        uncaught_high = int(np.sum(high & accepted))
        accepted_n = int(np.sum(accepted))
        threshold = float(scores[order[k - 1]]) if k > 0 else float("inf")
        rows.append(
            {
                "handoff_count": int(k),
                "handoff_rate": float(k / n),
                "threshold": threshold,
                "uncaught_high_error_count": uncaught_high,
                "uncaught_high_error_rate": float(uncaught_high / n),
                "high_error_miss_rate": float(uncaught_high / total_high) if total_high else 0.0,
                "accepted_high_error_rate": float(uncaught_high / accepted_n) if accepted_n else 0.0,
                "accepted_count": accepted_n,
            }
        )
    return pd.DataFrame(rows)


def _operating_points(curve: pd.DataFrame, rates: list[float]) -> list[dict[str, float | int]]:
    points = []
    n = int(curve["handoff_count"].max())
    for rate in rates:
        k = int(np.ceil(rate * n))
        row = curve[curve["handoff_count"] == k].iloc[0]
        points.append({key: (int(value) if key.endswith("_count") else float(value)) for key, value in row.items()})
    return points


def _curve_summary(curve: pd.DataFrame, high_total: int) -> dict[str, float | int]:
    x = curve["handoff_rate"].to_numpy(dtype=np.float64)
    y = curve["uncaught_high_error_rate"].to_numpy(dtype=np.float64)
    half = curve[curve["high_error_miss_rate"] <= 0.5]
    zero = curve[curve["uncaught_high_error_count"] <= 0]
    return {
        "area_under_handoff_uncaught_curve": float(np.trapezoid(y, x)),
        "initial_uncaught_high_error_rate": float(y[0]),
        "high_error_count": int(high_total),
        "handoff_rate_for_half_high_errors_caught": float(half["handoff_rate"].iloc[0]) if not half.empty else 1.0,
        "handoff_rate_for_zero_uncaught_high_errors": float(zero["handoff_rate"].iloc[0]) if not zero.empty else 1.0,
    }


def _make_figure(curves: dict[str, pd.DataFrame], summaries: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), constrained_layout=True)
    colors = {"ensemble_only": "tab:orange", "combined": "tab:blue"}
    labels = {"ensemble_only": "ensemble-only", "combined": "combined"}
    for key in ["ensemble_only", "combined"]:
        curve = curves[key]
        auc = summaries[key]["area_under_handoff_uncaught_curve"]
        ax.step(
            curve["handoff_rate"],
            curve["uncaught_high_error_rate"],
            where="post",
            color=colors[key],
            label=f"{labels[key]} (AUC={auc:.3f})",
        )
    ax.set_xlabel("solver handoff rate")
    ax.set_ylabel("uncaught high-error rate per test case")
    ax.set_title("Experiment 8: Solver-Handoff Pareto (lower-left is better)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(0.02, max(float(c["uncaught_high_error_rate"].max()) for c in curves.values()) * 1.08))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    block = _load_real_or_fallback(args)
    df = _build_feature_table(block, args)
    train_idx, test_idx = _split_indices(len(df), args.seed, args.train_fraction)
    threshold = float(np.quantile(df.iloc[train_idx]["rel_l2_error"].to_numpy(dtype=np.float64), args.high_error_quantile))
    test_error = df.iloc[test_idx]["rel_l2_error"].to_numpy(dtype=np.float64)
    high_test = test_error >= threshold

    ensemble_cols, combined_cols = _feature_columns(df)
    ens_score, ens_diag = _fit_risk_scores(df, train_idx, test_idx, ensemble_cols, args)
    comb_score, comb_diag = _fit_risk_scores(df, train_idx, test_idx, combined_cols, args)
    curves = {
        "ensemble_only": _pareto_curve(ens_score, high_test),
        "combined": _pareto_curve(comb_score, high_test),
    }
    high_total = int(np.sum(high_test))
    summaries = {key: _curve_summary(curve, high_total) for key, curve in curves.items()}

    payload = {
        "experiment": "exp8_abstention",
        "predictor": str(block["predictor"]),
        "input": str(block["input"]),
        "note": str(block["note"]),
        "grid": int(np.asarray(block["u_true"]).shape[1]),
        "n_samples": int(len(df)),
        "train_samples": int(len(train_idx)),
        "heldout_samples": int(len(test_idx)),
        "risk_model": args.risk_model,
        "high_error_metric": "relative_l2_error",
        "high_error_quantile_on_train": args.high_error_quantile,
        "high_error_threshold": threshold,
        "heldout_high_error_count": high_total,
        "heldout_high_error_rate": float(np.mean(high_test)),
        "rung4_backend": str(df["rung4_backend"].iloc[0]) if len(df) else "",
        "feature_sets": {
            "ensemble_only": ensemble_cols,
            "combined": combined_cols,
        },
        "risk_fit_diagnostics": {
            "ensemble_only": ens_diag,
            "combined": comb_diag,
        },
        "summary": summaries,
        "operating_points": {
            key: _operating_points(curve, args.operating_rates) for key, curve in curves.items()
        },
        "curves": {
            key: curve.to_dict(orient="records") for key, curve in curves.items()
        },
        "interpretation": (
            "The combined score aggregates ensemble spread, residual magnitude, and partial-inverse "
            "ladder fields. Lower area under the handoff-vs-uncaught curve indicates that fewer "
            "high-error cases remain at a fixed solver handoff budget."
        ),
    }

    out_json = REPO_ROOT / args.out_json
    out_fig = REPO_ROOT / args.out_fig
    _write_json(payload, out_json)
    _make_figure(curves, summaries, out_fig)

    rows = []
    for key in ["ensemble_only", "combined"]:
        rows.append(
            {
                "score": key,
                **summaries[key],
                "fit_mae": payload["risk_fit_diagnostics"][key]["mae"],
                "fit_r2": payload["risk_fit_diagnostics"][key]["r2"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"high-error threshold rel_l2={threshold:.6g}; heldout high-error count={high_total}", flush=True)
    print(f"saved {out_json}", flush=True)
    print(f"saved {out_fig}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/fno_ensemble_kappa100.npz")
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=100.0)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--synthetic-samples", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--high-error-quantile", type=float, default=0.85)
    parser.add_argument("--risk-model", choices=["ridge", "gbr"], default="gbr")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--operating-rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--seed", type=int, default=202606068)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--out-json", default="results/exp8_abstention.json")
    parser.add_argument("--out-fig", default="results/figures/exp8_abstention.png")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
