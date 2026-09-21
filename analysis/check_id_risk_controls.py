"""Cheap-feature controls for the fixed-ID risk-transfer diagnostic.

Uses the same training split, threshold, and GBR settings as audit_ood_protocol.
All target error labels are reserved for evaluation. The controls check whether
ensemble-plus-physics beats simple prediction and coefficient summaries.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.audit_ood_protocol import _clean, _load_all_blocks, _sha256, paired_rank_comparison, _rank_summary
from experiments.exp10_ood_abstention import _build_feature_table
from experiments.exp8_abstention import _feature_columns, _field_stats, _split_indices
from experiments.exp9_real_fno_ood_conformal import DATASET_ORDER, coefficient_feature_matrix
from experiments.exp7_weighted_cp import FEATURE_NAMES


def control_feature_tables(blocks):
    tables = {}
    for dataset in DATASET_ORDER:
        block = blocks[dataset]
        table = _build_feature_table(dataset, block)
        for name in ["u_hat_mean"]:
            feature_rows = [_field_stats("prediction_magnitude", np.abs(field)) for field in block[name]]
            table = pd.concat([table, pd.DataFrame(feature_rows)], axis=1)
        coefficients = coefficient_feature_matrix(block["a"])
        for index, name in enumerate(FEATURE_NAMES):
            table[f"coefficient_{name}"] = coefficients[:, index]
        tables[dataset] = table
    return tables


def _transform(table, columns):
    values = table[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite risk features")
    values = np.clip(values, -1e6, 1e6)
    return np.sign(values) * np.log1p(np.abs(values))


def evaluate_controls(tables, *, seed=202606070, bootstrap=2000, bootstrap_seed=2026091201):
    train, heldout = _split_indices(len(tables["test_id"]), seed, 0.6)
    training = tables["test_id"].iloc[train]
    ens, combined = _feature_columns(training)
    prediction = [column for column in training if column.startswith("prediction_magnitude_")]
    geometry = [column for column in training if column.startswith("coefficient_")]
    amg = [column for column in training if column.startswith("ladder_rung4_") and "_to_" not in column]
    cheap = ens + prediction + geometry
    sets = {
        "ensemble_only": ens,
        "prediction_magnitude_only": prediction,
        "coefficient_geometry_only": geometry,
        "ensemble_prediction_geometry": cheap,
        "amg_only": amg,
        "combined": combined,
        "ensemble_prediction_geometry_amg": cheap + amg,
        "all_features": combined + prediction + geometry,
    }
    threshold = float(np.quantile(training["rel_l2_error"], 0.85))
    models = {}
    for method, columns in sets.items():
        model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=2,
                                          min_samples_leaf=6, random_state=seed)
        model.fit(_transform(training, columns), training["rel_l2_error"].to_numpy(dtype=np.float64))
        models[method] = model
    out = {}
    for offset, dataset in enumerate(DATASET_ORDER):
        table = tables[dataset].iloc[heldout] if dataset == "test_id" else tables[dataset]
        high = table["rel_l2_error"].to_numpy(dtype=np.float64) >= threshold
        scores = {method: model.predict(_transform(table, sets[method])) for method, model in models.items()}
        contrasts = {}
        for baseline, augmented in [("ensemble_only", "combined"),
                                     ("ensemble_prediction_geometry", "combined"),
                                     ("ensemble_prediction_geometry", "ensemble_prediction_geometry_amg"),
                                     ("ensemble_prediction_geometry", "all_features"),
                                     ("prediction_magnitude_only", "amg_only")]:
            comparison = paired_rank_comparison({"ensemble_only": scores[baseline], "combined": scores[augmented]},
                                                high, np.random.default_rng(bootstrap_seed + offset), bootstrap)
            contrasts[f"{baseline}_minus_{augmented}"] = comparison["ensemble_minus_combined"]
        out[dataset] = {"n_heldout": len(table), "high_error_count": int(high.sum()),
                        "methods": {method: _rank_summary(values, high) for method, values in scores.items()},
                        "paired_differences": contrasts, "risk_scores": scores, "high_error_labels": high}
    return {
        "protocol": "ID-only risk-model fitting; all target labels used only for evaluation",
        "training_indices": train.tolist(), "id_heldout_indices": heldout.tolist(),
        "training_fraction": 0.6, "high_error_training_quantile": 0.85, "threshold": threshold,
        "risk_model": "GradientBoostingRegressor", "risk_model_parameters": models["combined"].get_params(),
        "feature_sets": sets, "feature_transform": "signed log1p; identical to original log1p on nonnegative features",
        "bootstrap": bootstrap, "bootstrap_seed": bootstrap_seed,
        "confidence_interval_scope": "paired heldout fields, conditional on fitted FNO/risk models and training-derived threshold",
        "positive_difference_favors": "second method in pair",
        "solver_handoff_executed": False, "per_shift": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/ood_features_bundle.npz"))
    parser.add_argument("--amg-audit-dir", type=Path, default=Path("results/ood_audit_fullamg_20260912"))
    parser.add_argument("--out", type=Path, default=Path("results/ood_audit_fullamg_20260912/id_risk_controls.json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    bundle = (REPO / args.input).resolve()
    amg_features = (REPO / args.amg_audit_dir / "rung4_refreshed.npz").resolve()
    out = (REPO / args.out).resolve()
    blocks = _load_all_blocks(bundle)
    with np.load(amg_features, allow_pickle=False) as data:
        for dataset in DATASET_ORDER:
            blocks[dataset]["rung4"] = np.asarray(data[f"{dataset}_rung4"])
    report = evaluate_controls(control_feature_tables(blocks), bootstrap=args.bootstrap)
    report["input_sha256"] = {str(path.relative_to(REPO)): _sha256(path) for path in [bundle, amg_features]}
    report["source_sha256"] = {name: _sha256(REPO / name) for name in [
        "analysis/check_id_risk_controls.py", "analysis/audit_ood_protocol.py",
        "experiments/exp10_ood_abstention.py", "experiments/exp8_abstention.py",
        "experiments/exp7_weighted_cp.py",
    ]}
    import sklearn
    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    for dataset, block in report["per_shift"].items():
        print(dataset, {method: round(stats["area_under_handoff_uncaught_curve"], 6) for method, stats in block["methods"].items()}, flush=True)
        key = "ensemble_prediction_geometry_minus_ensemble_prediction_geometry_amg"
        print("incremental AMG", block["paired_differences"][key]["area_under_handoff_uncaught_curve"], flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
