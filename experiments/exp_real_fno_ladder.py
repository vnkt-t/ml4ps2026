"""Real-FNO high-contrast residual localization confirmation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".pytest_cache" / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".pytest_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.metrics import compute_metrics  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402
from uncertainty.compute_ladder import compute_ladder  # noqa: E402


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


def _as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _mean_std(values: pd.Series) -> dict[str, float]:
    arr = values.to_numpy(dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_path)


def _make_figure(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fig.suptitle(title)

    x = np.arange(len(summary))
    labels = summary["rung"].to_list()

    axes[0].bar(x - 0.18, summary["spearman_mean"], width=0.36, label="Spearman")
    axes[0].bar(x + 0.18, summary["auroc_mean"], width=0.36, label="AUROC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("mean localization metric")
    axes[0].set_title("(a) Localization")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].scatter(np.maximum(summary["time_s_mean"], 1e-7), summary["spearman_mean"], s=44)
    for _, row in summary.iterrows():
        axes[1].annotate(
            row["rung"],
            (max(float(row["time_s_mean"]), 1e-7), float(row["spearman_mean"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean time per sample (s, log)")
    axes[1].set_ylabel("mean Spearman")
    axes[1].set_title("(b) Quality vs cost")
    axes[1].grid(True, which="both", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    in_path = REPO_ROOT / args.input
    out_json = REPO_ROOT / args.out_json
    out_fig = REPO_ROOT / args.out_fig
    if not in_path.exists():
        raise FileNotFoundError(f"missing FNO ensemble NPZ: {in_path}")

    npz = np.load(in_path, allow_pickle=False)
    a_test = np.asarray(npz["a_test"], dtype=np.float64)
    u_test = np.asarray(npz["u_test"], dtype=np.float64)
    n_test, H, W = u_test.shape
    pred = _sample_first_predictions(np.asarray(npz["u_pred"]), n_test, (H, W))
    sigma_all = np.asarray(npz["sigma_ens"], dtype=np.float64) if "sigma_ens" in npz.files else pred.std(axis=1)
    if "f_test" in npz.files:
        f_all = np.asarray(npz["f_test"], dtype=np.float64)
    else:
        f_source = npz["f_source"].item() if "f_source" in npz.files else 1.0
        f_all = np.full((n_test, H, W), float(f_source), dtype=np.float64)

    max_samples = n_test if args.max_samples is None else min(args.max_samples, n_test)
    rows: list[dict[str, float | int | str]] = []
    metadata: OrderedDict[str, object] = OrderedDict()
    for key in npz.files:
        if key in {"a_test", "u_test", "u_pred", "sigma_ens", "f_test"}:
            continue
        metadata[key] = _as_jsonable(npz[key])

    for i in range(max_samples):
        A = assemble(a_test[i])
        ensemble_i = pred[i]
        u_hat = ensemble_i.mean(axis=0)
        abs_error = np.abs(u_hat - u_test[i])
        ladder = compute_ladder(
            A=A,
            u_hat=u_hat,
            f=f_all[i],
            u_test=u_test[i],
            u_pred_ensemble=ensemble_i,
            sigma_ens=sigma_all[i],
        )
        for rung, payload in ladder.items():
            metrics = compute_metrics(payload["score"], abs_error)
            rows.append(
                {
                    "sample": i,
                    "rung": rung,
                    "spearman": metrics["spearman"],
                    "auroc": metrics["auroc"],
                    "top20_overlap": metrics["top20_overlap"],
                    "error_capture": metrics["error_capture"],
                    "com_dist": metrics["com_dist"],
                    "time_s": float(payload["time_s"]),
                    "matvecs": int(payload["matvecs"]),
                    "backend": str(payload.get("backend", "")),
                }
            )
        if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == max_samples:
            print(f"real-fno ladder processed {i + 1}/{max_samples}", flush=True)

    per_sample = pd.DataFrame(rows)
    summary_rows = []
    for rung, block in per_sample.groupby("rung", sort=False):
        summary_rows.append(
            {
                "rung": rung,
                "spearman_mean": _mean_std(block["spearman"])["mean"],
                "spearman_std": _mean_std(block["spearman"])["std"],
                "auroc_mean": _mean_std(block["auroc"])["mean"],
                "auroc_std": _mean_std(block["auroc"])["std"],
                "top20_overlap_mean": _mean_std(block["top20_overlap"])["mean"],
                "error_capture_mean": _mean_std(block["error_capture"])["mean"],
                "com_dist_mean": _mean_std(block["com_dist"])["mean"],
                "time_s_mean": _mean_std(block["time_s"])["mean"],
                "matvecs": int(round(float(block["matvecs"].mean()))),
                "backend": next((b for b in block["backend"].unique().tolist() if b), ""),
            }
        )
    summary = pd.DataFrame(summary_rows)

    raw = summary.loc[summary["rung"] == "1_abs_r"].iloc[0]
    rung4 = summary.loc[summary["rung"] == "4_vcycle"].iloc[0]
    best_partial = summary[summary["rung"].str.startswith(("3", "4"))].sort_values(
        "spearman_mean", ascending=False
    ).iloc[0]
    payload = {
        "experiment": "exp_real_fno_ladder",
        "input_npz": str(in_path),
        "n_samples": int(max_samples),
        "grid": int(H),
        "metadata": dict(metadata),
        "summary": summary.to_dict(orient="records"),
        "raw_residual": {
            "spearman_mean": float(raw["spearman_mean"]),
            "auroc_mean": float(raw["auroc_mean"]),
        },
        "rung4": {
            "spearman_mean": float(rung4["spearman_mean"]),
            "auroc_mean": float(rung4["auroc_mean"]),
            "backend": str(rung4["backend"]),
        },
        "best_partial_inverse": {
            "rung": str(best_partial["rung"]),
            "spearman_mean": float(best_partial["spearman_mean"]),
            "auroc_mean": float(best_partial["auroc_mean"]),
        },
        "confirmation": {
            "raw_poor_threshold": 0.35,
            "raw_poor": bool(float(raw["spearman_mean"]) < 0.35),
            "partial_inverse_recovers": bool(float(best_partial["spearman_mean"]) > float(raw["spearman_mean"]) + 0.2),
        },
    }
    _write_json(payload, out_json)
    _make_figure(summary, out_fig, "Real FNO High-Contrast Localization")
    print(summary.to_string(index=False), flush=True)
    print(f"saved {out_json}", flush=True)
    print(f"saved {out_fig}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/fno_ensemble_kappa100.npz")
    parser.add_argument("--out-json", default="results/real_fno_kappa100_ladder.json")
    parser.add_argument("--out-fig", default="figures/fig_real_fno_kappa100_ladder.png")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
