"""Independently replay retained timing statistics, raw rows, and provenance.

This imports no benchmark implementation. It reconstructs paired medians and
field-bootstrap intervals directly from the retained CSV. Historical source
snapshots are authoritative: later working-source edits are reported, not
silently substituted for the measured implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[setting] = "1"

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


def close(actual, expected, context):
    if not np.allclose(actual, expected, rtol=2e-11, atol=2e-13):
        raise AssertionError(f"{context}: reconstructed {actual!r}, retained {expected!r}")


def audit(directory, previous, artifact_only=False):
    summary_path = directory / "benchmark.json"
    raw_path = directory / "raw_times.csv"
    saved = json.loads(summary_path.read_text())
    frame = pd.read_csv(raw_path)
    require(saved["schema_version"] == 2, "expected final flux-inclusive schema 2")
    require(saved["input_and_source_hashes_unchanged"] is True, "run did not verify stable inputs/sources")
    protocol = saved["protocol"]
    require(protocol["thread_limit_requested"] == 1, "single-thread protocol missing")
    for setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        require(protocol["thread_environment"][setting] == "1", f"{setting} was not limited")
    for pool in saved["versions"]["detected_thread_pools"]:
        require(pool["num_threads"] == 1, f"detected numerical pool was not limited: {pool}")
    close(saved["average_cpu_cores_used"], saved["process_cpu_s"] / saved["benchmark_wall_s"], "CPU/wall ratio")
    require(np.isfinite(frame.time_s).all() and (frame.time_s > 0).all(), "timings must be positive finite")
    require(np.isfinite(frame.spearman).all(), "correlations must be finite")
    keys = ["N", "kappa", "sample", "method", "repeat"]
    require(not frame.duplicated(keys).any(), "duplicate field/method/repetition rows")
    require(len(frame.groupby(["N", "kappa"])) == len(saved["results"]), "extra or missing raw configurations")

    skipped_inputs = []
    if previous is not None and artifact_only and not previous.exists():
        skipped_inputs.append({"path": str(previous), "reason": "prior timing summary absent from artifact-only package"})
        earlier = None
    else:
        earlier = json.loads(previous.read_text()) if previous is not None else None
    checks = {"configurations": 0, "method_confidence_intervals": 0,
              "direct_flux_comparisons": 0, "previous_localization_comparisons": 0,
              "adaptive_field_methods": 0, "raw_rows": len(frame)}
    expected_rows = 0
    for key, result in saved["results"].items():
        group = frame[(frame.N == result["N"]) & (frame.kappa == result["kappa"])]
        n_fields = result["n_fields"]
        methods = set(result["methods"])
        require(set(group.method) == methods, f"{key}: method set mismatch")
        require(set(group["sample"]) == set(range(n_fields)), f"{key}: field selection mismatch")
        require(set(group["repeat"]) == set(range(protocol["repeats"])), f"{key}: repeat IDs mismatch")
        counts = group.groupby(["sample", "method"]).size()
        require(len(counts) == n_fields * len(methods) and (counts == protocol["repeats"]).all(),
                f"{key}: incomplete repeated field/method data")
        expected_rows += n_fields * len(methods) * protocol["repeats"]
        # Explicit field ordering and median over repeats are independent of
        # the original script's row order and groupby(sort=False).
        times = group.pivot_table(index="sample", columns="method", values="time_s", aggfunc="median").sort_index()
        draws = np.random.default_rng(protocol["seed"]).integers(
            0, n_fields, size=(protocol["bootstrap_replicates"], n_fields))
        for method, retained in result["methods"].items():
            values = times[method].to_numpy()
            ratio = values / times["oracle"].to_numpy()
            bootstrap_medians = np.median(ratio[draws], axis=1)
            interval = np.percentile(bootstrap_medians, [2.5, 97.5])
            close(interval, [retained["ratio_ci95_low"], retained["ratio_ci95_high"]], f"{key}/{method}: ratio CI")
            close(np.median(ratio), retained["median_paired_ratio_to_fresh_oracle"], f"{key}/{method}: ratio")
            close(1000 * np.median(values), retained["median_time_ms"], f"{key}/{method}: median ms")
            close(1000 * np.mean(values), retained["mean_time_ms"], f"{key}/{method}: mean ms")
            block = group[group.method == method]
            close(block.spearman.mean(), retained["mean_spearman"], f"{key}/{method}: correlation")
            if earlier is not None and method in earlier["results"][key]["methods"]:
                close(retained["mean_spearman"], earlier["results"][key]["methods"][method]["mean_spearman"],
                      f"{key}/{method}: preceding controlled run")
                checks["previous_localization_comparisons"] += 1
            if method in ("adaptive_rank90", "adaptive_flux90"):
                require(np.isfinite(block[["iterations", "target_met", "bound_checks", "certified_high_recall"]]).all().all(),
                        f"{key}/{method}: nonfinite adaptive metadata")
                require((block.target_met == 1).all() and (block.certified_high_recall >= .9).all(),
                        f"{key}/{method}: declared target not met")
                require(block.groupby("sample").iterations.nunique().max() == 1,
                        f"{key}/{method}: nonrepeatable stopping iteration")
                close(block.iterations.mean(), retained["mean_iterations"], f"{key}/{method}: mean iterations")
                close(block.bound_checks.mean(), retained["mean_bound_checks"], f"{key}/{method}: mean checks")
                close(block.target_met.mean(), retained["target_met_fraction"], f"{key}/{method}: target fraction")
                checks["adaptive_field_methods"] += n_fields
            checks["method_confidence_intervals"] += 1
        ratio = times["adaptive_flux90"].to_numpy() / times["adaptive_rank90"].to_numpy()
        difference = 1000 * (times["adaptive_flux90"] - times["adaptive_rank90"]).to_numpy()
        retained = result["adaptive_flux_vs_poisson"]
        close(np.median(ratio), retained["median_paired_time_ratio"], f"{key}: flux paired ratio")
        close(np.percentile(np.median(ratio[draws], axis=1), [2.5, 97.5]),
              [retained["ratio_ci95_low"], retained["ratio_ci95_high"]], f"{key}: flux ratio CI")
        close(difference.mean(), retained["mean_paired_time_difference_ms"], f"{key}: flux time difference")
        close(np.percentile(np.mean(difference[draws], axis=1), [2.5, 97.5]),
              [retained["difference_ci95_low_ms"], retained["difference_ci95_high_ms"]], f"{key}: flux difference CI")
        checks["direct_flux_comparisons"] += 1
        checks["configurations"] += 1
    require(len(frame) == expected_rows, "unexpected raw row total")

    working_source_changes = []
    for name, digest in saved["source_sha256"].items():
        require(sha256(directory / "source_snapshot" / name) == digest, f"source snapshot SHA mismatch: {name}")
        if not (REPO / name).exists() or sha256(REPO / name) != digest:
            working_source_changes.append(name)
    caches_verified = 0
    for record in saved["provenance"]:
        cache_path = Path(record["cache"])
        if not cache_path.is_absolute():
            cache_path = REPO / cache_path
        if artifact_only and not cache_path.exists():
            skipped_inputs.append({"path": str(cache_path), "reason": "prediction cache absent from artifact-only package"})
            continue
        require(sha256(cache_path) == record["cache_sha256"], f"prediction cache SHA mismatch: {cache_path}")
        caches_verified += 1
    return {"status": "ARTIFACT_ONLY_PASS" if artifact_only else "PASS", "checks": checks,
            "benchmark": str(summary_path), "benchmark_sha256": sha256(summary_path),
            "raw_csv_sha256": sha256(raw_path), "audit_source_sha256": sha256(__file__),
            "historical_source_snapshots_verified": True,
            "prediction_caches_verified": not artifact_only,
            "prediction_cache_hash_checks_performed": caches_verified,
            "artifact_only": artifact_only, "full_verification_passed": not artifact_only,
            "skipped_inputs": skipped_inputs,
            "working_source_changes_since_timing": working_source_changes,
            "previous_benchmark": str(previous) if previous is not None else None,
            "scope": "artifact/statistical replay; not an independent wall-time rerun or a proof of machine idleness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=REPO / "results/submission_timing_flux")
    parser.add_argument("--previous", type=Path, default=REPO / "results/submission_timing_singlethread/benchmark.json")
    parser.add_argument("--skip-previous", action="store_true")
    parser.add_argument("--artifact-only", action="store_true",
                        help="verify retained statistics/snapshots, allowing omitted prediction caches and prior timing summary; report partial verification")
    parser.add_argument("--report", type=Path, default=REPO / "results/submission_timing_audit.json")
    args = parser.parse_args()
    try:
        report = audit(args.directory, None if args.skip_previous else args.previous, args.artifact_only)
    except Exception as exc:
        report = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in ("PASS", "ARTIFACT_ONLY_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
