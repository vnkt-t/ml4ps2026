"""Reconstruct declared training coefficient draws and audit frozen test separation.

This is evidence about the saved artifacts and current training implementation,
not a retrospective proof of every historical training action. No solves,
predictions, fitting, or target-label-dependent decisions occur here.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.generate_contrast import family_C_field


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_hashes(fields):
    return {hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest() for x in fields}


def run():
    report = {
        "experiment": "frozen_checkpoint_split_audit",
        "scope": "reconstructed seed-0 coefficient draws, normalization metadata agreement, exact coefficient overlap; no retrospective guarantee of unrecorded historical actions",
        "training_draws": 700, "declared_training_data_seed": 0,
        "source_sha256": {name: sha256(REPO/name) for name in [
            "analysis/audit_submission_splits.py", "data/generate_contrast.py",
            "experiments/exp_real_fno_multi_kappa.py", "experiments/exp_submission_validation.py"]},
        "per_configuration": {}, "ood": {},
    }
    inputs = {}
    for n, checkpoint_dir in [(32, "checkpoints"), (64, "checkpoints_64")]:
        for kappa in [10, 100, 500]:
            rng = np.random.default_rng(0)
            training = np.stack([family_C_field(n, kappa, rng)[0] for _ in range(700)])
            train_hashes = field_hashes(training)
            logs = np.log(training)
            expected = {"logmean": float(logs.mean()), "logstd": float(logs.std())}
            original = np.stack([family_C_field(n, kappa, rng)[0] for _ in range(200)])
            fresh_rng = np.random.default_rng(20260912)
            fresh = np.stack([family_C_field(n, kappa, fresh_rng)[0] for _ in range(200)])
            blocks = {}
            for label, expected_fields, directory in [
                ("original", original, f"submission_{n}"),
                ("fresh", fresh, f"submission_fresh_{n}"),
            ]:
                path = REPO / "results" / directory / f"predictions_k{kappa}.npz"
                inputs[str(path.relative_to(REPO))] = sha256(path)
                with np.load(path, allow_pickle=False) as data:
                    a = data["a"]
                    if not np.array_equal(a, expected_fields):
                        raise AssertionError(f"reconstructed {label} fields differ from {path}")
                    signature = json.loads(str(data["signature"].item()))
                hashes = field_hashes(a)
                overlap = len(train_hashes & hashes)
                if overlap:
                    raise AssertionError(f"training/test coefficient overlap in {path}: {overlap}")
                blocks[label] = {"n_fields": len(a), "unique_coefficient_fields": len(hashes),
                    "training_overlap": overlap, "reconstruction_exact": True,
                    "test_seed": signature["test_seed"], "skip_fields": signature["skip_fields"]}
            if field_hashes(original) & field_hashes(fresh):
                raise AssertionError("original/fresh coefficient overlap")
            checkpoints = []
            directories = [checkpoint_dir]
            if n == 32 and kappa == 100:
                directories.append("checkpoints_longtrain")
            for directory, seed in itertools.product(directories, [0, 1, 2]):
                path = REPO / "results" / directory / f"fno_kappa{kappa}_seed{seed}.pt"
                sidecar = path.with_suffix(".json")
                for artifact in [path, sidecar]:
                    inputs[str(artifact.relative_to(REPO))] = sha256(artifact)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                metadata = payload["metadata"]
                saved_sidecar = json.loads(sidecar.read_text())
                for key, value in metadata.items():
                    if key in saved_sidecar and saved_sidecar[key] != value:
                        raise AssertionError(f"sidecar metadata mismatch: {path}, {key}")
                if metadata["grid_N"] != n or metadata["kappa"] != kappa or metadata["seed"] != seed:
                    raise AssertionError(f"checkpoint identity mismatch: {path}")
                if metadata["training"]["n_train"] != 700:
                    raise AssertionError(f"training count mismatch: {path}")
                if "data_seed" in metadata and metadata["data_seed"] != 0:
                    raise AssertionError(f"checkpoint has another data seed: {path}")
                diffs = {key: abs(metadata["normalization"][key] - val) for key, val in expected.items()}
                if max(diffs.values()) > 1e-12:
                    raise AssertionError(f"training-only coefficient normalization mismatch: {path}")
                checkpoints.append({"path": str(path.relative_to(REPO)),
                    "data_seed_embedded": metadata.get("data_seed"), "epochs": metadata["training"]["epochs"],
                    "input_normalization_abs_difference": diffs, "sidecar_agrees_with_embedded_metadata": True})
            report["per_configuration"][f"N{n}_k{kappa}"] = {
                "unique_training_coefficient_fields": len(train_hashes),
                "original_fresh_overlap": 0, "evaluation": blocks, "checkpoints": checkpoints}
            if n == 32 and kappa == 100:
                path = REPO / "results/ood_features_bundle.npz"
                inputs[str(path.relative_to(REPO))] = sha256(path)
                with np.load(path, allow_pickle=False) as data:
                    ood_hashes = {name: field_hashes(data[f"{name}_a"]) for name in
                                  ["calib", "test_id", "test_cs", "test_r", "test_b"]}
                    for name, hashes in ood_hashes.items():
                        overlap = len(train_hashes & hashes)
                        if overlap:
                            raise AssertionError(f"training/OOD overlap: {name}")
                        report["ood"][name] = {"n_fields": len(data[f"{name}_a"]),
                            "unique_coefficient_fields": len(hashes), "training_overlap": overlap,
                            "seed": int(data[f"{name}_seed"])}
                    pair_overlaps = {f"{left}__{right}": len(ood_hashes[left] & ood_hashes[right])
                                     for left, right in itertools.combinations(ood_hashes, 2)}
                    if any(pair_overlaps.values()):
                        raise AssertionError(f"overlap among OOD sets: {pair_overlaps}")
                    report["ood_pairwise_overlap"] = pair_overlaps
    for name, digest in {**report["source_sha256"], **inputs}.items():
        if sha256(REPO/name) != digest:
            raise AssertionError(f"input/source changed during audit: {name}")
    report["input_sha256"] = inputs
    report["input_and_source_hashes_unchanged"] = True
    report["limitations"] = [
        "Old 80-epoch checkpoints omit an embedded data seed; seed 0 is supported by exact input normalization and original-field reconstruction plus the saved code/default protocol.",
        "Target normalization was inspected in training code as using u_tr only; this audit does not recompute 700 PDE training targets.",
        "Exact coefficient disjointness is weaker than statistical independence or an audit of unrecorded historical model selection.",
        "Original fields were used in exploratory development; fresh fields are independent draws used with the frozen predictors, not new training seeds.",
        "The same random geometry is reused across contrasts and resolutions; condition-wise confidence intervals must not be pooled as independent geometries.",
        "Checkerboard has two unique fields; resampling 200 draws quantifies only the finite phase-mixture diagnostic.",
    ]
    output = REPO / "results/submission_split_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"PASS: 6 grid/contrast configurations, 21 checkpoint payloads, 5 OOD sets; saved {output.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    run()
