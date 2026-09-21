"""Reanalyze the frozen OOD bundle without overwriting historical artifacts.

Corrects evaluation bookkeeping and frozen scale floors, labels the weighted
conformal protocol as empirical, and tests risk-model transfer with labels from
ID training fields only. New results are deliberately separate from paper-era
exp9/exp10 files; changed numbers must never be silently substituted.

Usage:
  .venv/bin/python analysis/audit_ood_protocol.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp9_real_fno_ood_conformal import (
    DATASET_ORDER, DATASET_LABELS, _base_coverage_rows, _calibrate_methods,
    _make_coverage_figure, _make_localization_figure, _weighted_rows,
)
from experiments.exp10_ood_abstention import (
    _assert_bundle_finite, _build_feature_table, _load_bundle, _run_shift,
)
from experiments.exp8_abstention import _feature_columns, _split_indices
from uncertainty.conformal import conformal_quantile
from uncertainty.weighted_conformal import (
    fit_density_ratio_classifier, weighted_conformal_quantile,
)
from experiments.exp9_real_fno_ood_conformal import coefficient_feature_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def expected_rank_curve(scores: np.ndarray, high_error: np.ndarray) -> np.ndarray:
    """Uncaught high-error fraction at each whole-field handoff budget.

    Exact ties are averaged over uniformly random orderings within each tied
    score group. This avoids arbitrary row order affecting a constant scorer.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    high = np.asarray(high_error, dtype=bool).reshape(-1)
    if scores.size == 0 or scores.shape != high.shape or not np.all(np.isfinite(scores)):
        raise ValueError("scores and labels must be nonempty, matched, and finite")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_high = high[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0.0) + 1]
    lengths = np.diff(np.r_[starts, scores.size])
    positives = np.add.reduceat(sorted_high.astype(np.float64), starts)
    caught = np.r_[0.0, np.cumsum(np.repeat(positives / lengths, lengths))]
    return np.maximum(0.0, (high.sum() - caught) / high.size)


def _rank_summary(scores: np.ndarray, high: np.ndarray) -> dict[str, float]:
    curve = expected_rank_curve(scores, high)
    n = high.size
    stats = {"area_under_handoff_uncaught_curve": float(np.trapezoid(curve, np.arange(n + 1) / n))}
    for rate in [0.1, 0.2, 0.5]:
        stats[f"uncaught_high_error_rate_at_{int(rate * 100)}pct_handoff"] = float(curve[int(np.ceil(rate * n))])
    return stats


def paired_rank_comparison(scores: dict[str, np.ndarray], high: np.ndarray, rng, n_boot: int) -> dict:
    """Paired field bootstrap; positive deltas favor combined over ensemble."""
    high = np.asarray(high, dtype=bool)
    methods = ["ensemble_only", "combined"]
    base = {method: _rank_summary(scores[method], high) for method in methods}
    metrics = list(base[methods[0]])
    delta = {metric: base["ensemble_only"][metric] - base["combined"][metric] for metric in metrics}
    boot = {metric: [] for metric in metrics}
    for _ in range(int(n_boot)):
        indices = rng.integers(0, high.size, size=high.size)
        b = {method: _rank_summary(np.asarray(scores[method])[indices], high[indices]) for method in methods}
        for metric in metrics:
            boot[metric].append(b["ensemble_only"][metric] - b["combined"][metric])
    return {
        "methods": base,
        "ensemble_minus_combined": {
            metric: {
                "estimate": delta[metric],
                "ci95_low": float(np.quantile(boot[metric], 0.025)) if n_boot else delta[metric],
                "ci95_high": float(np.quantile(boot[metric], 0.975)) if n_boot else delta[metric],
            }
            for metric in metrics
        },
        "ci_scope": "paired bootstrap over heldout fields, conditional on the fitted predictor and risk models",
        "tie_policy": "expected outcome under uniform random ordering within equal risk scores",
        "positive_delta_favors": "combined",
    }


def id_transfer(tables: dict[str, pd.DataFrame], args) -> dict:
    """Train risk models once on an ID split and freeze them for every shift."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not 0.0 < args.train_fraction < 1.0 or not 0.0 < args.high_error_quantile < 1.0:
        raise ValueError("train fraction and high-error quantile must be in (0, 1)")
    train, heldout = _split_indices(len(tables["test_id"]), args.seed, args.train_fraction)
    if train.size < 2 or heldout.size < 2:
        raise ValueError("ID training and evaluation each require at least two fields")
    id_train = tables["test_id"].iloc[train]
    ens_cols, combined_cols = _feature_columns(id_train)
    feature_sets = {"ensemble_only": ens_cols, "combined": combined_cols}
    threshold = float(np.quantile(id_train["rel_l2_error"], args.high_error_quantile))
    fitted = {}
    for method, columns in feature_sets.items():
        if args.risk_model == "gbr":
            model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=2,
                                              min_samples_leaf=6, random_state=args.seed)
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        x = np.log1p(np.clip(id_train[columns].to_numpy(dtype=np.float64), 0.0, 1e6))
        model.fit(x, id_train["rel_l2_error"].to_numpy(dtype=np.float64))
        fitted[method] = model
    per_shift = {}
    for offset, dataset in enumerate(DATASET_ORDER):
        table = tables[dataset].iloc[heldout] if dataset == "test_id" else tables[dataset]
        high = table["rel_l2_error"].to_numpy(dtype=np.float64) >= threshold
        scores = {}
        for method, columns in feature_sets.items():
            x = np.log1p(np.clip(table[columns].to_numpy(dtype=np.float64), 0.0, 1e6))
            scores[method] = fitted[method].predict(x)
        compare = paired_rank_comparison(scores, high, np.random.default_rng(args.bootstrap_seed + offset), args.bootstrap)
        per_shift[dataset] = {
            "n_heldout": len(table), "high_error_count": int(high.sum()), "high_error_rate": float(high.mean()),
            "high_error_threshold": threshold, "comparison": compare,
            "risk_scores": scores, "high_error_labels": high,
            "risk_target_labels_used_for_training": False,
            "ranking_discrimination_defined": bool(0 < high.sum() < high.size),
            "unique_risk_scores": {method: int(np.unique(values).size) for method, values in scores.items()},
        }
    return {
        "training_distribution": "test_id only", "training_indices": train.tolist(),
        "id_heldout_indices": heldout.tolist(), "n_training": len(train),
        "threshold_source": "85th percentile by default of ID risk-training relative L2 errors; fixed for every target",
        "threshold": threshold, "risk_model": args.risk_model, "feature_sets": feature_sets,
        "models_frozen_across_shifts": True, "solver_handoff_executed": False,
        "handoff_unit": "whole_field", "per_shift": per_shift,
    }


def _load_all_blocks(bundle: Path) -> dict:
    blocks = _load_bundle(bundle)
    _assert_bundle_finite(blocks)
    with np.load(bundle, allow_pickle=False) as data:
        blocks["calib"] = {key.removeprefix("calib_"): np.asarray(data[key]) for key in data.files if key.startswith("calib_")}
        for name in DATASET_ORDER:
            blocks[name]["f"] = np.asarray(data[f"{name}_f"])
    for name in ["calib", *DATASET_ORDER]:
        blocks[name].update(name=name, label=DATASET_LABELS.get(name, "CALIB"), family="R" if name == "test_r" else "B" if name == "test_b" else "C")
    return blocks


def refresh_full_amg(blocks: dict, out: Path) -> dict:
    """Recompute all rung-4 fields with a deterministic, complete hierarchy."""
    import pyamg
    from solver.assemble_fd import assemble
    from solver.multigrid import vcycle, last_vcycle_backend, last_vcycle_cost

    provenance = {
        "pyamg_version": pyamg.__version__,
        "max_levels": None,
        "rng_seed_rule": "20260912 + dataset_index * 1000 + field_index",
        "source_sha256": {name: _sha256(REPO_ROOT / name) for name in ["solver/assemble_fd.py", "solver/multigrid.py", "solver/relaxation.py"]},
        "per_dataset": {},
    }
    refreshed = {}
    for dataset_index, name in enumerate(["calib", *DATASET_ORDER]):
        block = blocks[name]
        original = np.asarray(block["rung4"], dtype=np.float64)
        current = np.empty_like(original)
        records = []
        for index in range(current.shape[0]):
            A = assemble(block["a"][index])
            residual = A @ block["u_hat_mean"][index].reshape(-1) - block["f"][index].reshape(-1)
            np.random.seed(20260912 + dataset_index * 1000 + index)
            current[index] = np.abs(vcycle(A, residual, max_levels=None)).reshape(current[index].shape)
            backend, cost = last_vcycle_backend(), last_vcycle_cost()
            if backend != "pyamg_smoothed_aggregation" or int(cost["coarsest_unknowns"]) > 10:
                raise RuntimeError(f"full-AMG requirement failed for {name} field {index}: {backend}, {cost}")
            error = (block["u_hat_mean"][index] - block["u_true"][index]).reshape(-1)
            identity = np.linalg.norm(A @ error - residual) / max(np.linalg.norm(residual), 1e-15)
            if identity > 1e-8:
                raise RuntimeError(f"saved-prediction residual identity failed for {name}/{index}: {identity}")
            relative_change = np.linalg.norm(current[index] - original[index]) / max(np.linalg.norm(original[index]), 1e-15)
            records.append({"field": index, "backend": backend, "relative_change_from_frozen": float(relative_change),
                            "residual_identity_relative_error": float(identity), **cost})
        block["rung4"] = current
        block["rung4_backend"] = np.asarray([record["backend"] for record in records])
        refreshed[f"{name}_rung4"] = current
        provenance["per_dataset"][name] = {
            "n_fields": len(records), "per_field": records,
            "relative_change_max": max(record["relative_change_from_frozen"] for record in records),
            "relative_change_mean": float(np.mean([record["relative_change_from_frozen"] for record in records])),
        }
        print(f"Refreshed deterministic full-AMG fields: {name} ({len(records)})", flush=True)
    np.savez_compressed(out / "rung4_refreshed.npz", **refreshed)
    provenance["refreshed_rung4_sha256"] = _sha256(out / "rung4_refreshed.npz")
    return provenance


def make_transfer_figure(transfer: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), constrained_layout=True)
    titles = {"test_id": "Held-out ID", "test_r": "Roughness shift", "test_b": "Checkerboard", "test_cs": "Geometry shift"}
    for axis, dataset in zip(axes, DATASET_ORDER):
        block = transfer["per_shift"][dataset]
        high = np.asarray(block["high_error_labels"], dtype=bool)
        x = np.arange(high.size + 1) / high.size
        for method, color, label in [("ensemble_only", "#c9792f", "Ensemble"), ("combined", "#176b91", "Combined")]:
            y = expected_rank_curve(np.asarray(block["risk_scores"][method]), high)
            axis.plot(x, y, color=color, linewidth=1.8, label=label)
        axis.plot(x, high.mean() * (1 - x), color="#888888", linestyle="--", linewidth=1, label="Random selection")
        axis.set(title=f"{titles[dataset]}\n{high.sum()}/{high.size} high-error fields", xlabel="Fraction of whole fields selected", xlim=(0, 1), ylim=(0, max(0.11, high.mean() * 1.08)))
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Uncaught high-error fraction")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _target_mass_diagnostic(blocks, args):
    source = blocks["calib"]
    source_scores = source["abs_e"].reshape(source["abs_e"].shape[0], -1).max(axis=1)
    source_x = coefficient_feature_matrix(source["a"])
    outputs = {}
    for name, offset in [("test_r", 17), ("test_b", 29), ("test_cs", 41)]:
        target = blocks[name]
        ratio = fit_density_ratio_classifier(source_x, coefficient_feature_matrix(target["a"]),
                                            random_state=args.seed + offset, model=args.ratio_model)
        qhat = np.asarray([weighted_conformal_quantile(source_scores, ratio.source_weights, args.alpha, test_weight=weight)
                           for weight in ratio.target_weights])
        scores = target["abs_e"].reshape(target["abs_e"].shape[0], -1).max(axis=1)
        outputs[name] = {
            "coverage": float(np.mean(scores <= qhat)), "infinite_threshold_fraction": float(np.isinf(qhat).mean()),
            "target_specific_qhat": qhat,
            "caveat": "estimated in-sample classifier ratios and summary covariates; empirical only, no coverage guarantee",
        }
    return outputs


def run(args):
    bundle = (REPO_ROOT / args.input).resolve()
    historical = REPO_ROOT / "results/exp9_ood_conformal.json"
    historical_abst = REPO_ROOT / "results/exp10_ood_abstention.json"
    out = (REPO_ROOT / args.out_dir).resolve()
    if out == REPO_ROOT / "results":
        raise ValueError("use a dedicated audited output directory, not the frozen results root")
    frozen = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in [bundle, historical, historical_abst]}
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading preserved OOD bundle {bundle}", flush=True)
    blocks = _load_all_blocks(bundle)
    amg_provenance = refresh_full_amg(blocks, out) if args.refresh_amg else {
        "mode": "frozen bundle features; see separate full-AMG rerun for deterministic hierarchy/cost provenance"
    }
    old = json.loads(historical.read_text())
    exp9args = SimpleNamespace(alpha=old["alpha"], seed=202606069, ratio_model="logistic", weight_clip_quantile=0.995,
                              test_weight=1.0, bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed)
    methods, medians = _calibrate_methods(blocks["calib"], exp9args.alpha)
    print("Evaluating corrected scale floors and coverage confidence intervals", flush=True)
    rows = _base_coverage_rows(blocks, methods, medians, exp9args)
    weighted_rows, weighted_diagnostics = _weighted_rows(blocks["calib"], blocks, medians, exp9args)
    rows += weighted_rows
    by_key = {(r["test_set"], r["method"]): r for r in old["coverage"]}
    changes = []
    for row in rows:
        original = by_key[(row["test_set"], row["method"])]
        for metric in ["marginal_coverage", "fieldwise_coverage", "mean_width", "width_error_spearman", "width_error_auroc_top10"]:
            delta = float(row[metric]) - float(original[metric])
            if abs(delta) > 1e-12:
                changes.append({"test_set": row["test_set"], "method": row["method"], "metric": metric,
                                "historical": original[metric], "audited": row[metric], "delta": delta})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(out / "coverage_audited.csv", index=False)
    _make_coverage_figure(coverage, out / "coverage_audited.png", exp9args.alpha)
    _make_localization_figure(coverage, out / "localization_audited.png")
    tables = {dataset: _build_feature_table(dataset, blocks[dataset]) for dataset in DATASET_ORDER}
    # Preserve historical within-shift fits while adding protocol metadata and
    # paired contrasts. Target labels are explicitly required in this diagnostic.
    within = {}
    historical_args = SimpleNamespace(**vars(args))
    historical_args.seed = 202606070
    for index, dataset in enumerate(DATASET_ORDER):
        print(f"Auditing supervised within-shift risk ranking: {dataset}", flush=True)
        block = _run_shift(dataset, tables[dataset], historical_args, np.random.default_rng(args.bootstrap_seed + index), index * 101)
        block["paired_rank_comparison"] = paired_rank_comparison(
            {key: np.asarray(value) for key, value in block["heldout_risk_scores"].items()},
            np.asarray(block["heldout_high_error_labels"]), np.random.default_rng(args.bootstrap_seed + index), args.bootstrap)
        within[dataset] = block
    print("Evaluating risk models fitted only on ID labels across shifts", flush=True)
    transfer = id_transfer(tables, args)
    make_transfer_figure(transfer, out / "id_transfer.png")
    report = {
        "experiment": "audited_ood_protocol", "input_sha256": frozen,
        "historical_outputs_preserved": True,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "rung4_provenance": amg_provenance,
        "corrections": [
            "calibration scale floors reused unchanged across all target batches",
            "constant-width AUROC equals 0.5 instead of 0",
            "masked coverage CIs cluster-bootstrap the reported pooled-cell ratio",
            "dependent pooled cells carry no exchangeable-cell conformal guarantee",
            "weighted source and target ratios share a common normalization",
            "fixed-test-mass clipped weighted CP explicitly labeled empirical",
            "within-shift risk fitting distinguished from frozen ID-only risk transfer",
        ],
        "coverage": rows, "numerical_changes_from_historical": changes,
        "weighted_cp": weighted_diagnostics,
        "estimated_target_specific_mass_diagnostic": _target_mass_diagnostic(blocks, exp9args),
        "within_shift_supervised_ranking": within,
        "id_only_risk_transfer": transfer,
        "bootstrap": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
    }
    for name, digest in frozen.items():
        if _sha256(REPO_ROOT / name) != digest:
            raise RuntimeError(f"frozen input changed during audit: {name}")
    (out / "audit.json").write_text(json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    concise = []
    for dataset in DATASET_ORDER:
        b = transfer["per_shift"][dataset]
        delta = b["comparison"]["ensemble_minus_combined"]["area_under_handoff_uncaught_curve"]
        concise.append({"dataset": dataset, "n": b["n_heldout"], "high_error_rate": b["high_error_rate"],
                        "area_delta_ensemble_minus_combined": delta["estimate"], "ci95_low": delta["ci95_low"], "ci95_high": delta["ci95_high"]})
    pd.DataFrame(concise).to_csv(out / "id_transfer_summary.csv", index=False)
    print(pd.DataFrame(concise).to_string(index=False), flush=True)
    print(f"Preserved all frozen inputs. Wrote {out / 'audit.json'}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/ood_features_bundle.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/ood_audit_20260912"))
    parser.add_argument("--refresh-amg", action="store_true", help="Recompute all rung-4 fields with deterministic full hierarchies")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026091201)
    parser.add_argument("--seed", type=int, default=202606070)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--high-error-quantile", type=float, default=0.85)
    parser.add_argument("--risk-model", choices=["ridge", "gbr"], default="gbr")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--operating-rates", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
