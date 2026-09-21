"""Experiment 10: supervised within-shift field-ranking diagnostics.

Each target distribution supplies labeled training fields for its own risk model.
Curves simulate ranking whole cases for handoff; they do not execute a solver or
demonstrate zero-shot shift generalization, cellwise repair, or runtime savings.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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

from experiments.exp8_abstention import (  # noqa: E402
    _curve_summary as exp8_curve_summary,
)
from experiments.exp8_abstention import (
    _feature_columns as exp8_feature_columns,
)
from experiments.exp8_abstention import (
    _field_stats as exp8_field_stats,
)
from experiments.exp8_abstention import (
    _fit_risk_scores as exp8_fit_risk_scores,
)
from experiments.exp8_abstention import (
    _operating_points as exp8_operating_points,
)
from experiments.exp8_abstention import (
    _pareto_curve as exp8_pareto_curve,
)
from experiments.exp8_abstention import (
    _split_indices as exp8_split_indices,
)
from experiments.exp_real_fno_multi_kappa import (  # noqa: E402
    assert_finite_payload,
    bootstrap_abstention_summaries,
    finite_records,
    write_json,
)


DATASET_ORDER = ["test_id", "test_r", "test_b", "test_cs"]
DATASET_LABELS = {
    "test_id": "TEST_ID",
    "test_r": "TEST_R",
    "test_b": "TEST_B",
    "test_cs": "TEST_CS",
}


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _finite_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _finite_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_payload(v) for v in value]
    if isinstance(value, tuple):
        return [_finite_payload(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_payload(value.tolist())
    if isinstance(value, np.generic):
        return _finite_payload(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _require_keys(npz: np.lib.npyio.NpzFile, keys: list[str]) -> None:
    missing = [key for key in keys if key not in npz.files]
    if missing:
        raise KeyError(f"bundle missing required keys: {missing}")


def _load_bundle(path: Path) -> dict[str, dict[str, np.ndarray]]:
    bundle_path = REPO_ROOT / path
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing exp9 bundle: {bundle_path}")
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(bundle_path, allow_pickle=False) as npz:
        for name in DATASET_ORDER:
            required = [
                f"{name}_a",
                f"{name}_u_true",
                f"{name}_u_hat_mean",
                f"{name}_sigma_ens",
                f"{name}_abs_e",
                f"{name}_abs_r",
                f"{name}_smooth_abs_r",
                f"{name}_jacobi20",
                f"{name}_rung4",
                f"{name}_interface_mask",
                f"{name}_rel_l2",
            ]
            _require_keys(npz, required)
            out[name] = {key.removeprefix(f"{name}_"): np.asarray(npz[key]) for key in required}
            if f"{name}_rung4_backend" in npz.files:
                out[name]["rung4_backend"] = np.asarray(npz[f"{name}_rung4_backend"])
            else:
                out[name]["rung4_backend"] = np.asarray(["precomputed_bundle"] * out[name]["rel_l2"].shape[0])
        metadata_keys = [key for key in ["grid_N", "kappa", "alpha", "dataset_names"] if key in npz.files]
        out["_metadata"] = {key: np.asarray(npz[key]) for key in metadata_keys}
    return out


def _assert_bundle_finite(blocks: dict[str, dict[str, np.ndarray]]) -> None:
    for dataset in DATASET_ORDER:
        for key, value in blocks[dataset].items():
            if key == "rung4_backend":
                continue
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "i", "u"} and not np.all(np.isfinite(arr)):
                raise RuntimeError(f"non-finite bundle array {dataset}_{key}")
        block = blocks[dataset]
        shape = np.asarray(block["abs_e"]).shape
        if len(shape) != 3 or shape[0] < 4 or min(shape[1:]) < 1:
            raise ValueError(f"{dataset} must contain at least four nonempty 2D fields; got {shape}")
        for key in ["a", "u_true", "u_hat_mean", "sigma_ens", "abs_r", "smooth_abs_r", "jacobi20", "rung4", "interface_mask"]:
            if np.asarray(block[key]).shape != shape:
                raise ValueError(f"{dataset}_{key} shape differs from abs_e: {np.asarray(block[key]).shape} vs {shape}")
        if np.asarray(block["rel_l2"]).shape != (shape[0],):
            raise ValueError(f"{dataset}_rel_l2 must contain one score per field")
        expected_error = np.abs(np.asarray(block["u_hat_mean"]) - np.asarray(block["u_true"]))
        if not np.allclose(block["abs_e"], expected_error, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{dataset}_abs_e disagrees with saved predictions and truth")


def _build_feature_table(dataset: str, block: dict[str, np.ndarray]) -> pd.DataFrame:
    abs_e = np.asarray(block["abs_e"], dtype=np.float64)
    rel = np.asarray(block["rel_l2"], dtype=np.float64).reshape(-1)
    n = int(abs_e.shape[0])
    rows: list[dict[str, float | int | str]] = []
    backend = np.asarray(block.get("rung4_backend", np.asarray(["precomputed_bundle"] * n)))

    for i in range(n):
        row: dict[str, float | int | str] = {
            "sample": int(i),
            "dataset": dataset,
            "rel_l2_error": float(rel[i]),
            "mean_abs_error": float(np.mean(abs_e[i])),
            "max_abs_error": float(np.max(abs_e[i])),
            "rung4_backend": str(backend[i]) if backend.size else "precomputed_bundle",
        }
        row.update(exp8_field_stats("ens_sigma", np.asarray(block["sigma_ens"][i], dtype=np.float64)))
        row.update(exp8_field_stats("res_abs", np.asarray(block["abs_r"][i], dtype=np.float64)))
        row.update(exp8_field_stats("res_smooth", np.asarray(block["smooth_abs_r"][i], dtype=np.float64)))
        row.update(exp8_field_stats("ladder_jac20", np.asarray(block["jacobi20"][i], dtype=np.float64)))
        row.update(exp8_field_stats("ladder_rung4", np.asarray(block["rung4"][i], dtype=np.float64)))
        row["ladder_rung4_to_res_p95"] = float(
            np.clip(float(row["ladder_rung4_p95"]) / max(float(row["res_abs_p95"]), 1.0e-12), 0.0, 1.0e6)
        )
        row["ladder_jac20_to_res_p95"] = float(
            np.clip(float(row["ladder_jac20_p95"]) / max(float(row["res_abs_p95"]), 1.0e-12), 0.0, 1.0e6)
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError(f"non-finite feature table values for {dataset}")
    return df


def _run_shift(
    dataset: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    seed_offset: int,
) -> dict[str, Any]:
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < float(args.high_error_quantile) < 1.0:
        raise ValueError("high_error_quantile must be in (0, 1)")
    split_seed = int(args.seed) + int(seed_offset)
    train_idx, test_idx = exp8_split_indices(len(df), split_seed, float(args.train_fraction))
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError("risk fitting and held-out evaluation each require at least two fields")
    threshold = float(
        np.quantile(
            df.iloc[train_idx]["rel_l2_error"].to_numpy(dtype=np.float64),
            float(args.high_error_quantile),
        )
    )
    test_error = df.iloc[test_idx]["rel_l2_error"].to_numpy(dtype=np.float64)
    high_test = test_error >= threshold

    ensemble_cols, combined_cols = exp8_feature_columns(df)
    ens_score, ens_diag = exp8_fit_risk_scores(df, train_idx, test_idx, ensemble_cols, args)
    comb_score, comb_diag = exp8_fit_risk_scores(df, train_idx, test_idx, combined_cols, args)
    curves = {
        "ensemble_only": exp8_pareto_curve(ens_score, high_test),
        "combined": exp8_pareto_curve(comb_score, high_test),
    }
    high_total = int(np.sum(high_test))
    summaries = {key: exp8_curve_summary(curve, high_total) for key, curve in curves.items()}
    ci = bootstrap_abstention_summaries(
        {"ensemble_only": ens_score, "combined": comb_score},
        high_test,
        rng,
        int(args.bootstrap),
    )
    for key in ["ensemble_only", "combined"]:
        for metric, stats in ci[key].items():
            summaries[key][f"{metric}_ci95_low"] = float(stats["ci95_low"])
            summaries[key][f"{metric}_ci95_high"] = float(stats["ci95_high"])

    return {
        "dataset": dataset,
        "label": DATASET_LABELS[dataset],
        "n_samples": int(len(df)),
        "train_samples": int(len(train_idx)),
        "heldout_samples": int(len(test_idx)),
        "split_seed": int(split_seed),
        "training_indices": train_idx.tolist(),
        "heldout_indices": test_idx.tolist(),
        "training_distribution": dataset,
        "evaluation_protocol": "supervised_within_shift_disjoint_field_split",
        "handoff_unit": "whole_field",
        "target_labels_used_for_risk_training": True,
        "solver_handoff_executed": False,
        "risk_model": str(args.risk_model),
        "high_error_metric": "relative_l2_error",
        "high_error_quantile_on_train": float(args.high_error_quantile),
        "high_error_threshold": threshold,
        "heldout_high_error_count": high_total,
        "heldout_high_error_rate": float(np.mean(high_test)) if high_test.size else 0.0,
        "feature_sets": {
            "ensemble_only": ensemble_cols,
            "combined": combined_cols,
        },
        "risk_fit_diagnostics": {
            "ensemble_only": ens_diag,
            "combined": comb_diag,
        },
        "heldout_risk_scores": {
            "ensemble_only": ens_score.tolist(),
            "combined": comb_score.tolist(),
        },
        "heldout_high_error_labels": high_test.tolist(),
        "summary": summaries,
        "summary_ci": ci,
        "operating_points": {
            key: exp8_operating_points(curve, list(args.operating_rates)) for key, curve in curves.items()
        },
        "curves": {key: finite_records(curve) for key, curve in curves.items()},
    }


def _make_figure(per_shift: dict[str, Any], out_path: Path) -> None:
    fig, axes_arr = plt.subplots(1, len(DATASET_ORDER), figsize=(5.0 * len(DATASET_ORDER), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes_arr)
    colors = {"ensemble_only": "tab:orange", "combined": "tab:blue"}
    labels = {"ensemble_only": "ensemble-only", "combined": "combined"}
    for ax, dataset in zip(axes, DATASET_ORDER):
        block = per_shift[dataset]
        for key in ["ensemble_only", "combined"]:
            curve = pd.DataFrame(block["curves"][key])
            auc = float(block["summary"][key]["area_under_handoff_uncaught_curve"])
            ax.step(
                curve["handoff_rate"],
                curve["uncaught_high_error_rate"],
                where="post",
                color=colors[key],
                label=f"{labels[key]} AUC={auc:.3f}",
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("fraction of whole fields selected for handoff")
        ax.set_ylabel("uncaught high-error rate")
        ax.set_xlim(0.0, 1.0)
        ymax = max(0.02, max(float(pd.DataFrame(block["curves"][key])["uncaught_high_error_rate"].max()) for key in colors))
        ax.set_ylim(0.0, ymax * 1.08)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _summary_rows(per_shift: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        block = per_shift[dataset]
        for method in ["ensemble_only", "combined"]:
            summary = block["summary"][method]
            rows.append(
                {
                    "dataset": dataset,
                    "label": DATASET_LABELS[dataset],
                    "method": method,
                    "area": float(summary["area_under_handoff_uncaught_curve"]),
                    "area_ci95_low": float(summary["area_under_handoff_uncaught_curve_ci95_low"]),
                    "area_ci95_high": float(summary["area_under_handoff_uncaught_curve_ci95_high"]),
                    "handoff_at_50pct_caught": float(summary["handoff_rate_for_half_high_errors_caught"]),
                    "handoff_at_50pct_caught_ci95_low": float(
                        summary["handoff_rate_for_half_high_errors_caught_ci95_low"]
                    ),
                    "handoff_at_50pct_caught_ci95_high": float(
                        summary["handoff_rate_for_half_high_errors_caught_ci95_high"]
                    ),
                    "heldout_high_error_count": int(block["heldout_high_error_count"]),
                }
            )
    return rows


def _apply_smoke_paths(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.bootstrap = min(int(args.bootstrap), 100)
    defaults = {
        "input": Path("results/ood_features_bundle.npz"),
        "out_json": Path("results/exp10_ood_abstention.json"),
        "out_fig": Path("figures/fig10_ood_abstention.png"),
    }
    smoke_paths = {
        "input": Path("results/_smoke_ood_features_bundle.npz"),
        "out_json": Path("results/_smoke_exp10_ood_abstention.json"),
        "out_fig": Path("figures/_smoke_fig10_ood_abstention.png"),
    }
    for attr, default in defaults.items():
        if getattr(args, attr) == default:
            setattr(args, attr, smoke_paths[attr])
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/ood_features_bundle.npz"))
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--high-error-quantile", type=float, default=0.85)
    parser.add_argument("--risk-model", choices=["ridge", "gbr"], default="gbr")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--operating-rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--seed", type=int, default=202606070)
    parser.add_argument("--bootstrap-seed", type=int, default=2026060701)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out-json", type=Path, default=Path("results/exp10_ood_abstention.json"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/fig10_ood_abstention.png"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    return _apply_smoke_paths(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    set_global_seed(int(args.seed))
    print(f"exp10 loading bundle {(REPO_ROOT / args.input).relative_to(REPO_ROOT)}", flush=True)
    blocks = _load_bundle(args.input)
    _assert_bundle_finite(blocks)
    feature_tables = {dataset: _build_feature_table(dataset, blocks[dataset]) for dataset in DATASET_ORDER}
    rng = np.random.default_rng(int(args.bootstrap_seed))
    per_shift = {
        dataset: _run_shift(dataset, feature_tables[dataset], args, rng, idx * 101)
        for idx, dataset in enumerate(DATASET_ORDER)
    }

    _make_figure(per_shift, REPO_ROOT / args.out_fig)
    rows = _summary_rows(per_shift)
    summary_df = pd.DataFrame(rows)
    numeric = summary_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("exp10 summary contains non-finite numeric values")

    metadata = blocks.get("_metadata", {})
    payload = {
        "experiment": "exp10_ood_abstention",
        "smoke": bool(args.smoke),
        "input": str(args.input),
        "grid_N": int(np.asarray(metadata.get("grid_N", np.asarray(0))).reshape(-1)[0]),
        "kappa": float(np.asarray(metadata.get("kappa", np.asarray(0.0))).reshape(-1)[0]),
        "train_fraction": float(args.train_fraction),
        "risk_model": str(args.risk_model),
        "evaluation_protocol": "separate supervised risk fit within each target distribution",
        "training_labels": "60% split by default; true relative L2 errors from each target distribution",
        "handoff_unit": "whole_field",
        "solver_handoff_executed": False,
        "runtime_savings_measured": False,
        "zero_shot_ood_evaluation": False,
        "high_error_metric": "relative_l2_error",
        "high_error_quantile_on_train": float(args.high_error_quantile),
        "bootstrap": int(args.bootstrap),
        "per_shift": per_shift,
        "summary_table": rows,
        "outputs": {
            "json": str(args.out_json),
            "figure": str(args.out_fig),
        },
        "bundle_reloadable_without_fno": True,
        "wall_time_s": float(time.time() - t0),
    }
    payload = _finite_payload(payload)
    assert_finite_payload(payload)
    write_json(payload, REPO_ROOT / args.out_json)

    print("EXP10_SUMMARY", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"saved {args.out_json}", flush=True)
    print(f"saved {args.out_fig}", flush=True)
    print("OOD_ABSTENTION_DONE", flush=True)
    return payload


if __name__ == "__main__":
    run(parse_args())
