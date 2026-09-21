"""Multi-kappa real-FNO residual-localization and abstention replication.

This script extends the single-kappa pilot into a checkpointed, reusable
multi-contrast run. It intentionally keeps the kappa=100 training defaults
aligned with the existing pilot: Family-C data, 32x32 grid, 700 train / 200
test samples, tiny FNO architecture (modes=12, width=24, layers=3), and 80
epochs with seeds 0, 1, 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["MPLCONFIGDIR"] = ".cache/mpl"
os.environ["XDG_CACHE_HOME"] = ".cache"
os.environ["PYTHONPYCACHEPREFIX"] = ".pytest_cache/pycache"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for cache_dir in [
    REPO_ROOT / ".cache" / "mpl",
    REPO_ROOT / ".pytest_cache" / "pycache",
    REPO_ROOT / "results" / "checkpoints",
    REPO_ROOT / "figures",
]:
    cache_dir.mkdir(parents=True, exist_ok=True)

from experiments.exp0_pilot_fno import build_inputs, make_contrast_dataset, rel_l2  # noqa: E402
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
from models.tiny_fno import FNO2d  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402
from solver.residual import residual  # noqa: E402
from solver.solve import solve  # noqa: E402
from uncertainty.compute_ladder import compute_ladder  # noqa: E402


DEFAULT_OUT_PREFIX = "results/real_fno_multi_kappa"
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints"
SOURCE_SPEC = "data.generate_contrast.default_rhs: 1 + 0.25*sin(2*pi*x)*sin(pi*y)"
SELECTED_RUNGS = {
    "raw_abs_r": "1_abs_r",
    "jacobi_20": "3e_jacobi_20",
    "rung4": "4_vcycle",
    "oracle": "5_oracle",
}


def kappa_label(kappa: float) -> str:
    return f"{float(kappa):g}".replace(".", "p")


def checkpoint_path(kappa: float, seed: int, smoke: bool = False) -> Path:
    prefix = "fno_smoke" if smoke else "fno"
    return CHECKPOINT_DIR / f"{prefix}_kappa{kappa_label(kappa)}_seed{int(seed)}.pt"


def checkpoint_path_in_dir(kappa: float, seed: int, checkpoint_dir: Path, smoke: bool = False) -> Path:
    prefix = "fno_smoke" if smoke else "fno"
    return checkpoint_dir / f"{prefix}_kappa{kappa_label(kappa)}_seed{int(seed)}.pt"


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_path)


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_model_physical(
    model: torch.nn.Module,
    X: np.ndarray,
    normalization: dict[str, float],
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return (pred_norm * normalization["ustd"] + normalization["umean"]).astype(np.float32)


def build_checkpoint_payload(
    model: torch.nn.Module,
    seed: int,
    args: argparse.Namespace,
    kappa: float,
    normalization: dict[str, float],
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    arch = {
        "modes": int(args.modes),
        "width": int(args.width),
        "n_layers": int(args.layers),
        "in_ch": 3,
    }
    metadata = {
        "arch": arch,
        "normalization": {k: float(v) for k, v in normalization.items()},
        "kappa": float(kappa),
        "family": "contrast",
        "grid_N": int(args.N),
        "source_f": SOURCE_SPEC,
        "seed": int(seed),
        "data_seed": int(getattr(args, "data_seed", -1)),
        "training": {
            "n_train": int(n_train),
            "n_test": int(n_test),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "device": "cpu",
        },
    }
    return {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "arch": arch,
        "normalization": metadata["normalization"],
    }


def save_checkpoint(
    model: torch.nn.Module,
    seed: int,
    args: argparse.Namespace,
    kappa: float,
    normalization: dict[str, float],
    n_train: int,
    n_test: int,
    smoke: bool,
) -> Path:
    ckpt_dir = Path(getattr(args, "checkpoint_dir", CHECKPOINT_DIR))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path_in_dir(kappa, seed, ckpt_dir, smoke=smoke)
    payload = build_checkpoint_payload(model, seed, args, kappa, normalization, n_train, n_test)
    torch.save(payload, path)
    sidecar = dict(payload["metadata"])
    sidecar["checkpoint"] = str(path.relative_to(REPO_ROOT))
    write_json(sidecar, path.with_suffix(".json"))
    return path


@dataclass
class LoadedFNO:
    model: torch.nn.Module
    metadata: dict[str, Any]
    checkpoint: Path
    device: str = "cpu"

    def __call__(self, a_field: np.ndarray) -> np.ndarray:
        return self.predict(a_field)

    def predict(self, a_field: np.ndarray) -> np.ndarray:
        arr = np.asarray(a_field, dtype=np.float64)
        single = arr.ndim == 2
        if single:
            arr = arr[None, :, :]
        N = int(self.metadata["grid_N"])
        norm = self.metadata["normalization"]
        X = build_inputs(arr, float(norm["logmean"]), float(norm["logstd"]), N)
        pred = predict_model_physical(self.model, X, norm, self.device)
        return pred[0] if single else pred


class LoadedEnsemble(list[LoadedFNO]):
    def predict_members(self, a_field: np.ndarray) -> np.ndarray:
        return np.stack([member(a_field) for member in self], axis=0)

    def predict(self, a_field: np.ndarray) -> dict[str, np.ndarray]:
        members = self.predict_members(a_field)
        return {
            "members": members,
            "mean": members.mean(axis=0),
            "sigma": members.std(axis=0) if members.shape[0] > 1 else np.zeros_like(members[0]),
        }


def _validate_checkpoint_metadata(
    metadata: dict,
    path: Path,
    *,
    kappa: float,
    grid_N: int | None,
    n_train: int | None,
    n_test: int | None,
    data_seed: int | None,
) -> None:
    """Guard against loading a checkpoint that does not match the requested config.

    Filenames encode only kappa/seed (not grid size), so without this check a 32x32
    checkpoint could be silently loaded for a 64x64 run, or a checkpoint trained with
    a different (data_seed, n_train) -- which changes the test-field RNG draw -- could
    be reused on non-identical test fields. The checkpoint metadata carries all the
    fields needed to detect these, so validate every one that is available.
    (Codex review 2026-06-06, Item 1.)
    """
    if "kappa" in metadata and abs(float(metadata["kappa"]) - float(kappa)) > 1e-9:
        raise ValueError(f"{path}: checkpoint kappa={metadata['kappa']} != requested {kappa}")
    if grid_N is not None and "grid_N" in metadata and int(metadata["grid_N"]) != int(grid_N):
        raise ValueError(
            f"{path}: checkpoint grid_N={metadata['grid_N']} != requested N={grid_N} "
            f"(filenames omit N -- wrong-resolution checkpoint)"
        )
    training = metadata.get("training", {})
    if n_train is not None and "n_train" in training and int(training["n_train"]) != int(n_train):
        raise ValueError(
            f"{path}: checkpoint n_train={training['n_train']} != requested {n_train} "
            f"(reuse would evaluate on a different test-field RNG sequence)"
        )
    if n_test is not None and "n_test" in training and int(training["n_test"]) != int(n_test):
        raise ValueError(f"{path}: checkpoint n_test={training['n_test']} != requested {n_test}")
    # data_seed is only present on checkpoints saved after the 2026-06-06 fix; verify
    # when available, otherwise field identity rests on the documented default + the
    # byte-identical non-rung-4 metric check used to validate the original reuse run.
    if data_seed is not None and "data_seed" in metadata and int(metadata["data_seed"]) >= 0 \
            and int(metadata["data_seed"]) != int(data_seed):
        raise ValueError(
            f"{path}: checkpoint data_seed={metadata['data_seed']} != requested {data_seed} "
            f"(reuse would regenerate different test fields)"
        )


def load_ensemble(
    kappa: float,
    seeds: list[int] | None = None,
    checkpoint_dir: Path | str = CHECKPOINT_DIR,
    device: str = "cpu",
    smoke: bool = False,
    grid_N: int | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    data_seed: int | None = None,
) -> LoadedEnsemble:
    """Load checkpointed FNO members as callables with an ensemble predict path.

    When grid_N / n_train / n_test / data_seed are supplied, each member's saved
    metadata is validated against them so a mismatched checkpoint raises instead of
    silently producing predictions on the wrong grid or wrong test fields.
    """
    ckpt_dir = Path(checkpoint_dir)
    if seeds is None:
        prefix = "fno_smoke" if smoke else "fno"
        pattern = f"{prefix}_kappa{kappa_label(kappa)}_seed*.pt"
        paths = sorted(
            ckpt_dir.glob(pattern),
            key=lambda p: int(re.search(r"_seed(\d+)\.pt$", p.name).group(1)),  # type: ignore[union-attr]
        )
    else:
        paths = [checkpoint_path_in_dir(kappa, seed, ckpt_dir, smoke=smoke) for seed in seeds]
    if not paths:
        raise FileNotFoundError(f"no checkpoints found for kappa={kappa:g} in {ckpt_dir}")

    members = LoadedEnsemble()
    for idx, path in enumerate(paths):
        payload = _torch_load(path, device)
        metadata = dict(payload.get("metadata", {}))
        _validate_checkpoint_metadata(
            metadata, path, kappa=kappa, grid_N=grid_N,
            n_train=n_train, n_test=n_test, data_seed=data_seed,
        )
        if seeds is not None and "seed" in metadata and int(metadata["seed"]) != int(seeds[idx]):
            raise ValueError(f"{path}: checkpoint seed={metadata['seed']} != requested {seeds[idx]}")
        arch = dict(payload.get("arch", metadata.get("arch", {})))
        model = FNO2d(
            modes=int(arch["modes"]),
            width=int(arch["width"]),
            n_layers=int(arch["n_layers"]),
            in_ch=int(arch["in_ch"]),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        members.append(LoadedFNO(model=model, metadata=metadata, checkpoint=path, device=device))
    return members


def train_one_checkpointed(
    seed: int,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    args: argparse.Namespace,
    kappa: float,
    normalization: dict[str, float],
    n_train: int,
    n_test: int,
    device: str,
    smoke: bool,
) -> tuple[np.ndarray, Path, float]:
    set_global_seed(seed)
    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.layers, in_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(Ytr, dtype=torch.float32, device=device)
    lossfn = torch.nn.MSELoss()

    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = lossfn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()

    pred = predict_model_physical(model, Xte, normalization, device)
    path = save_checkpoint(model, seed, args, kappa, normalization, n_train, n_test, smoke=smoke)

    loaded = load_ensemble(
        kappa, seeds=[seed], device=device, smoke=smoke,
        checkpoint_dir=Path(getattr(args, "checkpoint_dir", CHECKPOINT_DIR)),
        grid_N=int(args.N), n_train=n_train, n_test=n_test, data_seed=int(args.data_seed),
    )[0]
    loaded_pred = predict_model_physical(loaded.model, Xte, normalization, device)
    roundtrip = float(np.max(np.abs(pred.astype(np.float64) - loaded_pred.astype(np.float64))))
    if not np.isfinite(roundtrip) or roundtrip > 1.0e-5:
        raise RuntimeError(f"checkpoint round-trip exceeded tolerance: {roundtrip:.6e}")
    return pred, path, roundtrip


def top_indices(x: np.ndarray, q: float) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    k = max(1, int(np.ceil(q * flat.size)))
    return np.argpartition(flat, -k)[-k:]


def localization_metrics_top10(score: np.ndarray, abs_error: np.ndarray) -> dict[str, float]:
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    if s.size != e.size:
        raise ValueError(f"score and abs_error sizes differ: {s.size} vs {e.size}")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(e)):
        raise ValueError("score and abs_error must be finite")

    if np.all(s == s[0]) or np.all(e == e[0]):
        rho = 0.0
    else:
        rho = spearmanr(s, e).statistic
        if not np.isfinite(rho):
            rho = 0.0

    y = np.zeros(e.size, dtype=np.int8)
    y[top_indices(e, 0.10)] = 1
    try:
        auc = float(roc_auc_score(y, s))
    except ValueError:
        auc = 0.5

    top_score = top_indices(s, 0.10)
    total_error = float(e.sum())
    capture = float(e[top_score].sum() / total_error) if total_error > 0.0 else 0.0
    overlap = float(len(set(top_score.tolist()) & set(top_indices(e, 0.10).tolist())) / max(1, len(top_score)))
    return {
        "spearman": float(rho),
        "auroc": float(auc),
        "error_capture_top10": float(capture),
        "top10_overlap": overlap,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty or non-finite array")
    mean = float(arr.mean())
    if arr.size == 1 or n_boot <= 0:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    draws = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = arr[draws].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "mean": mean,
        "ci95_low": float(np.quantile(boot, alpha / 2.0)),
        "ci95_high": float(np.quantile(boot, 1.0 - alpha / 2.0)),
    }


def aggregate_rung_metrics(
    per_sample: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    metric_names = ["spearman", "auroc", "error_capture_top10", "top10_overlap", "time_s", "matvecs"]
    # Cost columns are averaged, not bootstrapped: they are deterministic properties of
    # the method, not sampled quantities. Kept as floats -- rung 4's cost is a measured
    # AMG cycle complexity, and rounding it to an int hides the real number.
    cost_names = ["matvecs", "matvecs_with_setup"]
    for rung, block in per_sample.groupby("rung", sort=False):
        row: dict[str, Any] = {}
        for metric in metric_names:
            if metric in cost_names:
                row[metric] = float(block[metric].mean())
            else:
                row[metric] = bootstrap_mean_ci(block[metric].to_numpy(dtype=np.float64), rng, n_boot)
        for cost in cost_names:
            if cost in block.columns:
                row[cost] = float(block[cost].mean())
        backend_values = [str(x) for x in block["backend"].unique().tolist() if str(x)]
        row["backend"] = backend_values[0] if backend_values else ""
        out[str(rung)] = row
    return out


def make_feature_table_row(
    sample_idx: int,
    a_field: np.ndarray,
    u_true: np.ndarray,
    f_field: np.ndarray,
    ensemble: np.ndarray,
    sigma: np.ndarray,
    ladder: dict[str, dict[str, Any]],
) -> dict[str, float | int | str]:
    u_hat = np.asarray(ensemble, dtype=np.float64).mean(axis=0)
    err = u_hat - np.asarray(u_true, dtype=np.float64)
    abs_error = np.abs(err)
    denom = float(np.linalg.norm(np.asarray(u_true, dtype=np.float64).reshape(-1)) + 1e-12)

    row: dict[str, float | int | str] = {
        "sample": int(sample_idx),
        "rel_l2_error": float(np.linalg.norm(err.reshape(-1)) / denom),
        "mean_abs_error": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "rung4_backend": str(ladder["4_vcycle"].get("backend", "")),
    }
    row.update(exp8_field_stats("ens_sigma", sigma))
    row.update(exp8_field_stats("res_abs", ladder["1_abs_r"]["score"]))
    row.update(exp8_field_stats("res_smooth", ladder["2_smooth_abs_r"]["score"]))
    row.update(exp8_field_stats("ladder_jac20", ladder["3e_jacobi_20"]["score"]))
    row.update(exp8_field_stats("ladder_rung4", ladder["4_vcycle"]["score"]))
    row["ladder_rung4_to_res_p95"] = float(
        np.clip(float(row["ladder_rung4_p95"]) / max(float(row["res_abs_p95"]), 1e-12), 0.0, 1e6)
    )
    row["ladder_jac20_to_res_p95"] = float(
        np.clip(float(row["ladder_jac20_p95"]) / max(float(row["res_abs_p95"]), 1e-12), 0.0, 1e6)
    )
    _ = a_field, f_field
    return row


def bootstrap_abstention_summaries(
    scores: dict[str, np.ndarray],
    high_test: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, dict[str, dict[str, float]]]:
    metric_names = [
        "area_under_handoff_uncaught_curve",
        "handoff_rate_for_half_high_errors_caught",
        "handoff_rate_for_zero_uncaught_high_errors",
    ]
    high = np.asarray(high_test, dtype=bool)
    n = high.size
    out: dict[str, dict[str, dict[str, float]]] = {key: {} for key in scores}
    if n <= 1 or n_boot <= 0:
        for key, score in scores.items():
            summary = exp8_curve_summary(exp8_pareto_curve(score, high), int(high.sum()))
            for metric in metric_names:
                val = float(summary[metric])
                out[key][metric] = {"mean": val, "ci95_low": val, "ci95_high": val}
        return out

    boot_values = {
        key: {metric: np.empty(int(n_boot), dtype=np.float64) for metric in metric_names}
        for key in scores
    }
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        high_b = high[idx]
        for key, score in scores.items():
            curve = exp8_pareto_curve(np.asarray(score, dtype=np.float64)[idx], high_b)
            summary = exp8_curve_summary(curve, int(high_b.sum()))
            for metric in metric_names:
                boot_values[key][metric][b] = float(summary[metric])

    for key, score in scores.items():
        base = exp8_curve_summary(exp8_pareto_curve(score, high), int(high.sum()))
        for metric in metric_names:
            vals = boot_values[key][metric]
            out[key][metric] = {
                "mean": float(base[metric]),
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
            }
    return out


def finite_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                record[key] = None
    return records


def run_abstention(
    feature_df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    abst_args = SimpleNamespace(
        seed=int(args.abstention_seed),
        train_fraction=float(args.train_fraction),
        high_error_quantile=float(args.high_error_quantile),
        risk_model=str(args.risk_model),
        ridge_alpha=float(args.ridge_alpha),
        operating_rates=list(args.operating_rates),
    )
    train_idx, test_idx = exp8_split_indices(len(feature_df), abst_args.seed, abst_args.train_fraction)
    threshold = float(
        np.quantile(
            feature_df.iloc[train_idx]["rel_l2_error"].to_numpy(dtype=np.float64),
            abst_args.high_error_quantile,
        )
    )
    test_error = feature_df.iloc[test_idx]["rel_l2_error"].to_numpy(dtype=np.float64)
    high_test = test_error >= threshold

    ensemble_cols, combined_cols = exp8_feature_columns(feature_df)
    ens_score, ens_diag = exp8_fit_risk_scores(feature_df, train_idx, test_idx, ensemble_cols, abst_args)
    comb_score, comb_diag = exp8_fit_risk_scores(feature_df, train_idx, test_idx, combined_cols, abst_args)
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
            summaries[key][f"{metric}_ci95_low"] = stats["ci95_low"]
            summaries[key][f"{metric}_ci95_high"] = stats["ci95_high"]

    return {
        "train_samples": int(len(train_idx)),
        "heldout_samples": int(len(test_idx)),
        "risk_model": abst_args.risk_model,
        "high_error_metric": "relative_l2_error",
        "high_error_quantile_on_train": abst_args.high_error_quantile,
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
        "summary": summaries,
        "summary_ci": ci,
        "operating_points": {
            key: exp8_operating_points(curve, abst_args.operating_rates) for key, curve in curves.items()
        },
        "curves": {key: finite_records(curve) for key, curve in curves.items()},
    }


def assert_finite_payload(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_payload(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_finite_payload(item, f"{path}[{i}]")
    elif isinstance(value, (int, float, np.generic)):
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"non-finite value at {path}: {value}")


def process_kappa(args: argparse.Namespace, kappa: float) -> dict[str, Any]:
    t0 = time.time()
    device = "cpu"
    ckpt_dir = Path(getattr(args, "checkpoint_dir", CHECKPOINT_DIR))
    n_train = int(args.ntrain)
    n_test = int(args.ntest)
    rng = np.random.default_rng(int(args.data_seed))
    print(f"kappa={kappa:g}: generating Family-C train/test data", flush=True)
    a_tr, _f_tr, u_tr = make_contrast_dataset(args.N, n_train, float(kappa), rng)
    a_te, f_te, u_te = make_contrast_dataset(args.N, n_test, float(kappa), rng)

    normalization = {
        "logmean": float(np.log(a_tr).mean()),
        "logstd": float(np.log(a_tr).std()),
        "umean": float(u_tr.mean()),
        "ustd": float(u_tr.std()),
    }
    Xtr = build_inputs(a_tr, normalization["logmean"], normalization["logstd"], args.N)
    Xte = build_inputs(a_te, normalization["logmean"], normalization["logstd"], args.N)
    Ytr = ((u_tr - normalization["umean"]) / normalization["ustd"]).astype(np.float32)

    if bool(getattr(args, "reuse_checkpoints", False)):
        # Refresh mode: skip training and reuse the saved checkpoints (identical predictions),
        # so the ladder is recomputed with the CURRENT rung-4 backend (e.g. true AMG once pyamg
        # is installed) on the SAME test fields. Train data above is still generated only to
        # reproduce the normalization and the test-field RNG sequence.
        loaded = load_ensemble(
            float(kappa), seeds=[int(s) for s in args.seeds], device=device,
            smoke=bool(args.smoke), checkpoint_dir=ckpt_dir,
            grid_N=int(args.N), n_train=n_train, n_test=n_test, data_seed=int(args.data_seed),
        )
        preds = np.asarray(loaded.predict(a_te)["members"], dtype=np.float32)
        if preds.shape != (len(args.seeds), n_test, args.N, args.N):
            raise RuntimeError(f"reuse-checkpoints: unexpected ensemble prediction shape {preds.shape}")
        fields_fp = hashlib.sha1(np.ascontiguousarray(a_te, dtype=np.float64).tobytes()).hexdigest()[:12]
        print(f"kappa={kappa:g} [reuse-ckpt]: test-field fingerprint={fields_fp} "
              f"(N={args.N}, n_train={n_train}, data_seed={args.data_seed})", flush=True)
        checkpoint_paths = [
            str(checkpoint_path_in_dir(float(kappa), int(seed), ckpt_dir, smoke=bool(args.smoke)).relative_to(REPO_ROOT))
            for seed in args.seeds
        ]
        roundtrips = [0.0]
        load_max_abs = 0.0
        for member_idx, seed in enumerate(args.seeds):
            seed_rel = rel_l2(preds[member_idx].astype(np.float64), u_te.astype(np.float64))
            print(
                f"kappa={kappa:g} seed={seed} [reuse-ckpt]: rel-L2 mean={seed_rel.mean():.4f}",
                flush=True,
            )
    else:
        preds = np.empty((len(args.seeds), n_test, args.N, args.N), dtype=np.float32)
        checkpoint_paths = []
        roundtrips = []
        for member_idx, seed in enumerate(args.seeds):
            pred, path, roundtrip = train_one_checkpointed(
                int(seed),
                Xtr,
                Ytr,
                Xte,
                args,
                float(kappa),
                normalization,
                n_train,
                n_test,
                device,
                smoke=bool(args.smoke),
            )
            preds[member_idx] = pred
            checkpoint_paths.append(str(path.relative_to(REPO_ROOT)))
            roundtrips.append(roundtrip)
            seed_rel = rel_l2(pred.astype(np.float64), u_te.astype(np.float64))
            print(
                f"kappa={kappa:g} seed={seed}: rel-L2 mean={seed_rel.mean():.4f} "
                f"checkpoint={path.name} roundtrip={roundtrip:.3e}",
                flush=True,
            )

        loaded = load_ensemble(
            float(kappa), seeds=[int(s) for s in args.seeds], device=device,
            smoke=bool(args.smoke), checkpoint_dir=ckpt_dir,
        )
        loaded_pred = loaded.predict(a_te)["members"]
        loaded_pred = np.moveaxis(loaded_pred, 0, 0)
        load_max_abs = float(np.max(np.abs(loaded_pred.astype(np.float64) - preds.astype(np.float64))))
        if not np.isfinite(load_max_abs) or load_max_abs > 1.0e-5:
            raise RuntimeError(f"load_ensemble prediction mismatch: {load_max_abs:.6e}")

    ens_mean = preds.astype(np.float64).mean(axis=0)
    sigma_ens = preds.astype(np.float64).std(axis=0) if len(args.seeds) > 1 else np.zeros_like(ens_mean)
    rel = rel_l2(ens_mean, u_te.astype(np.float64))

    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    identity_errors = []
    for i in range(n_test):
        A = assemble(np.asarray(a_te[i], dtype=np.float64))
        u_hat = np.asarray(ens_mean[i], dtype=np.float64)
        u_true = np.asarray(u_te[i], dtype=np.float64)
        f_i = np.asarray(f_te[i], dtype=np.float64)
        err = u_hat - u_true
        abs_error = np.abs(err)
        r_vec = residual(A, u_hat, f_i)
        exact_from_residual = solve(A, r_vec).reshape(args.N, args.N)
        identity_errors.append(float(np.max(np.abs(exact_from_residual - err))))

        ladder = compute_ladder(
            A=A,
            u_hat=u_hat,
            f=f_i,
            u_test=u_true,
            u_pred_ensemble=np.asarray(preds[:, i], dtype=np.float64),
            sigma_ens=np.asarray(sigma_ens[i], dtype=np.float64),
        )
        oracle_identity = float(np.max(np.abs(ladder["5_oracle"]["score"] - np.abs(err))))
        if oracle_identity > 1.0e-8:
            raise RuntimeError(f"oracle residual identity failed at sample {i}: {oracle_identity:.3e}")

        for rung, payload in ladder.items():
            metrics = localization_metrics_top10(payload["score"], abs_error)
            rows.append(
                {
                    "sample": int(i),
                    "rung": str(rung),
                    "spearman": metrics["spearman"],
                    "auroc": metrics["auroc"],
                    "error_capture_top10": metrics["error_capture_top10"],
                    "top10_overlap": metrics["top10_overlap"],
                    "time_s": float(payload["time_s"]),
                    # float, not int: rung 4's cost is a measured cycle complexity
                    # (e.g. 3.64), and truncating it to 3 hides the real number.
                    "matvecs": float(payload["matvecs"]),
                    # Cost as actually run, including the per-sample AMG setup that
                    # cycle complexity omits. NaN for rungs with no setup phase.
                    "matvecs_with_setup": float(
                        payload.get("matvecs_with_setup", payload["matvecs"])
                    ),
                    "backend": str(payload.get("backend", "")),
                }
            )
        feature_rows.append(
            make_feature_table_row(
                sample_idx=i,
                a_field=np.asarray(a_te[i], dtype=np.float64),
                u_true=u_true,
                f_field=f_i,
                ensemble=np.asarray(preds[:, i], dtype=np.float64),
                sigma=np.asarray(sigma_ens[i], dtype=np.float64),
                ladder=ladder,
            )
        )
        if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == n_test:
            print(f"kappa={kappa:g}: ladder/abstention features {i + 1}/{n_test}", flush=True)

    max_identity_error = float(np.max(identity_errors))
    if max_identity_error > 1.0e-8:
        raise RuntimeError(f"same-A residual identity failed: {max_identity_error:.3e}")

    boot_rng = np.random.default_rng(int(args.bootstrap_seed) + int(round(float(kappa) * 1000.0)))
    per_sample = pd.DataFrame(rows)
    feature_df = pd.DataFrame(feature_rows)
    rungs = aggregate_rung_metrics(per_sample, boot_rng, int(args.bootstrap))
    abstention = run_abstention(feature_df, args, boot_rng)
    rel_summary = bootstrap_mean_ci(rel, boot_rng, int(args.bootstrap))

    payload = {
        "kappa": float(kappa),
        "family": "contrast",
        "grid_N": int(args.N),
        "n_train": n_train,
        "n_test": n_test,
        "seeds": [int(s) for s in args.seeds],
        "normalization": normalization,
        "checkpoint_paths": checkpoint_paths,
        "roundtrip_max_abs": float(np.max(roundtrips)),
        "load_ensemble_max_abs": load_max_abs,
        "same_A_residual_identity_max_abs": max_identity_error,
        "rel_l2": rel_summary,
        "rungs": rungs,
        "abstention": abstention,
        "wall_time_s": float(time.time() - t0),
    }
    assert_finite_payload(payload)
    return payload


def make_csv_rows(per_kappa: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kappa_key, block in per_kappa.items():
        for rung, metrics in block["rungs"].items():
            row: dict[str, Any] = {
                "kappa": float(kappa_key),
                "rung": rung,
                "backend": metrics.get("backend", ""),
                "matvecs": metrics["matvecs"],
                "matvecs_with_setup": metrics.get("matvecs_with_setup", metrics["matvecs"]),
            }
            for metric in ["spearman", "auroc", "error_capture_top10", "top10_overlap", "time_s"]:
                stats = metrics[metric]
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_ci95_low"] = stats["ci95_low"]
                row[f"{metric}_ci95_high"] = stats["ci95_high"]
            rows.append(row)
    return rows


def make_summary_rows(per_kappa: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kappa_key, block in per_kappa.items():
        r = block["rungs"]
        abst = block["abstention"]["summary"]
        rows.append(
            {
                "kappa": float(kappa_key),
                "rel_l2_mean": block["rel_l2"]["mean"],
                "raw_abs_r_rho": r["1_abs_r"]["spearman"]["mean"],
                "raw_abs_r_auroc": r["1_abs_r"]["auroc"]["mean"],
                "jacobi20_rho": r["3e_jacobi_20"]["spearman"]["mean"],
                "jacobi20_auroc": r["3e_jacobi_20"]["auroc"]["mean"],
                "rung4_rho": r["4_vcycle"]["spearman"]["mean"],
                "rung4_auroc": r["4_vcycle"]["auroc"]["mean"],
                "oracle_rho": r["5_oracle"]["spearman"]["mean"],
                "oracle_auroc": r["5_oracle"]["auroc"]["mean"],
                "abst_area_combined": abst["combined"]["area_under_handoff_uncaught_curve"],
                "abst_area_ensemble_only": abst["ensemble_only"]["area_under_handoff_uncaught_curve"],
            }
        )
    return rows


def figure_paths(out_prefix: str) -> tuple[Path, Path]:
    if out_prefix == DEFAULT_OUT_PREFIX:
        return (
            REPO_ROOT / "figures" / "fig_real_fno_multi_kappa_localization.png",
            REPO_ROOT / "figures" / "fig_real_fno_multi_kappa_abstention.png",
        )
    name = Path(out_prefix).name
    return (
        REPO_ROOT / "figures" / f"fig_{name}_localization.png",
        REPO_ROOT / "figures" / f"fig_{name}_abstention.png",
    )


def plot_localization(per_kappa: dict[str, Any], out_path: Path) -> None:
    kappas = np.array([float(k) for k in per_kappa.keys()], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    colors = {
        "raw_abs_r": "tab:orange",
        "jacobi_20": "tab:green",
        "rung4": "tab:blue",
        "oracle": "tab:purple",
    }
    labels = {
        "raw_abs_r": "raw |r|",
        "jacobi_20": "Jacobi-20",
        "rung4": "rung-4",
        "oracle": "oracle",
    }
    for ax, metric, ylabel in zip(axes, ["spearman", "auroc"], ["Spearman rho", "AUROC top-10%"]):
        for key, rung in SELECTED_RUNGS.items():
            means = np.array([per_kappa[str(k)]["rungs"][rung][metric]["mean"] for k in kappas], dtype=np.float64)
            lows = np.array([per_kappa[str(k)]["rungs"][rung][metric]["ci95_low"] for k in kappas], dtype=np.float64)
            highs = np.array([per_kappa[str(k)]["rungs"][rung][metric]["ci95_high"] for k in kappas], dtype=np.float64)
            ax.plot(kappas, means, marker="o", color=colors[key], label=labels[key])
            ax.fill_between(kappas, lows, highs, color=colors[key], alpha=0.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("kappa")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_abstention(per_kappa: dict[str, Any], out_path: Path) -> None:
    kappas = [float(k) for k in per_kappa.keys()]
    n = len(kappas)
    fig, axes_arr = plt.subplots(1, n, figsize=(max(5.2, 4.8 * n), 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes_arr)
    for ax, kappa in zip(axes, kappas):
        block = per_kappa[str(kappa)]["abstention"]
        for key, color, label in [
            ("ensemble_only", "tab:orange", "ensemble-only"),
            ("combined", "tab:blue", "combined"),
        ]:
            curve = pd.DataFrame(block["curves"][key])
            area = block["summary"][key]["area_under_handoff_uncaught_curve"]
            ax.step(
                curve["handoff_rate"],
                curve["uncaught_high_error_rate"],
                where="post",
                color=color,
                label=f"{label} (area={area:.3f})",
            )
        ax.set_title(f"kappa={kappa:g}")
        ax.set_xlabel("solver handoff rate")
        ax.set_ylabel("uncaught high-error rate")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappas", type=float, nargs="+", default=[10.0, 100.0, 500.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--ntrain", type=int, default=700)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Skip training; load the saved FNO checkpoints and recompute the ladder/abstention "
        "on the SAME test fields. Use to refresh rung-4 with the current backend (e.g. true AMG "
        "once pyamg is installed) without retraining. Pair with a distinct --out prefix.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(CHECKPOINT_DIR),
        help="Directory for FNO checkpoints. Use a distinct dir for a different grid size (e.g. "
        "results/checkpoints_64) so 64x64 checkpoints do not overwrite the 32x32 ones "
        "(filenames encode kappa/seed but NOT N).",
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260606)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--high-error-quantile", type=float, default=0.85)
    parser.add_argument("--risk-model", choices=["ridge", "gbr"], default="gbr")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--operating-rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--abstention-seed", type=int, default=202606068)
    args = parser.parse_args()
    if args.smoke:
        args.kappas = [100.0]
        args.seeds = [0]
        args.epochs = 3
        args.ntrain = 40
        args.ntest = 20
        args.bootstrap = min(int(args.bootstrap), 100)
        args.progress_every = min(int(args.progress_every), 5)
        if args.out == DEFAULT_OUT_PREFIX:
            args.out = "results/real_fno_multi_kappa_smoke"
    # Resolve checkpoint dir to absolute so save_checkpoint's path.relative_to(REPO_ROOT) works
    # whether the user passes a relative (e.g. results/checkpoints_64) or absolute path.
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir
    args.checkpoint_dir = str(ckpt_dir)
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(0)
    per_kappa: dict[str, Any] = {}
    for kappa in args.kappas:
        block = process_kappa(args, float(kappa))
        per_kappa[str(float(kappa))] = block

    summary_rows = make_summary_rows(per_kappa)
    csv_rows = make_csv_rows(per_kappa)
    out_prefix = Path(args.out)
    out_json = REPO_ROOT / f"{out_prefix}.json"
    out_csv = REPO_ROOT / f"{out_prefix}.csv"
    loc_fig, abst_fig = figure_paths(args.out)

    payload = {
        "experiment": "exp_real_fno_multi_kappa",
        "smoke": bool(args.smoke),
        "config": {
            "kappas": [float(k) for k in args.kappas],
            "seeds": [int(s) for s in args.seeds],
            "epochs": int(args.epochs),
            "N": int(args.N),
            "ntrain": int(args.ntrain),
            "ntest": int(args.ntest),
            "arch": {"modes": int(args.modes), "width": int(args.width), "n_layers": int(args.layers), "in_ch": 3},
            "bootstrap": int(args.bootstrap),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve().relative_to(REPO_ROOT)),
        },
        "per_kappa": per_kappa,
        "summary_table": summary_rows,
        "outputs": {
            "json": str(out_json.relative_to(REPO_ROOT)),
            "csv": str(out_csv.relative_to(REPO_ROOT)),
            "localization_figure": str(loc_fig.relative_to(REPO_ROOT)),
            "abstention_figure": str(abst_fig.relative_to(REPO_ROOT)),
        },
    }
    assert_finite_payload(payload)
    write_json(payload, out_json)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    plot_localization(per_kappa, loc_fig)
    plot_abstention(per_kappa, abst_fig)

    summary_df = pd.DataFrame(summary_rows)
    print("SUMMARY_CSV", flush=True)
    print(summary_df.to_csv(index=False), end="", flush=True)
    print(f"saved {out_json.relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {out_csv.relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {loc_fig.relative_to(REPO_ROOT)}", flush=True)
    print(f"saved {abst_fig.relative_to(REPO_ROOT)}", flush=True)
    print("MULTI_KAPPA_DONE", flush=True)
    return payload


if __name__ == "__main__":
    run(parse_args())
