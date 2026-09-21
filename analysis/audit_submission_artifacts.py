"""Independently audit the current submission from retained field-level evidence.

Reconstructs bootstrap draws and paired differences without importing experiment
summary helpers; evaluates TPFA identities/energy through independent face sums;
checks all retained decision ledgers and replays a declared deterministic subset
with a dense one-dimensional eigendecomposition of the reference Poisson operator.
This does not retrain networks or prove historical model-selection independence.
The separate audit_paper_numbers.py is only a historical hardcoded-value ledger.
Full mode exports new audit-derived diagnostics under results/submission_audit_diagnostics;
it never edits the original per-field experiment CSVs or prediction caches.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg, LinearOperator
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

PRIMARY = [f"submission{fresh}_{n}" for fresh in ("", "_fresh") for n in (32, 64)]
EXPECTED_METHODS = {"ensemble", "raw", "smoothed", "prediction_magnitude", "boundary_distance",
                    "poisson", "amg", "oracle"}
EXPECTED_METHODS |= {f"jacobi{k}" for k in (1, 5, 10, 20, 40, 80)}
EXPECTED_METHODS |= {f"{method}{k}" for method in ("cg", "pcg", "poisson_pcg") for k in (5, 10, 20, 40, 80)}
FOOTER = "Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute."
ORIGINAL_STYLE_HASH = "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a"


class Audit:
    def __init__(self, artifact_only=False):
        self.section = "initialization"
        self.counts = Counter()
        self.failures = []
        self.notes = []
        self.inputs = {}
        self.details = {}
        self.artifact_only = artifact_only
        self.skipped = []

    def check(self, condition, name, **details):
        self.counts[self.section] += 1
        if not bool(condition):
            self.failures.append({"section": self.section, "check": name, **details})

    def close(self, actual, expected, name, atol=2e-12, rtol=2e-10):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=atol, rtol=rtol)
        self.check(ok, name, maximum_absolute_difference=float(np.max(np.abs(a-b))) if a.shape == b.shape and a.size else None)

    def track(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = REPO/path
        name = str(path.relative_to(REPO))
        digest = sha256(path)
        self.inputs.setdefault(name, digest)
        return path

    def read(self, path):
        return json.loads(self.track(path).read_text())

    def frame(self, path):
        return pd.read_csv(self.track(path))

    def skip(self, name, reason):
        self.skipped.append({"section": self.section, "check": name, "reason": reason, "verified": False})

    def available(self, path, name):
        path = Path(path)
        if not path.is_absolute(): path = REPO/path
        if path.exists(): return True
        if self.artifact_only:
            self.skip(name, "Full frozen input is unavailable in this artifact-only package: "+str(path.relative_to(REPO)))
            return False
        raise FileNotFoundError(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_matches(audit, path, expected):
    path = audit.track(path)
    if sha256(path) == expected:
        return True
    if path != REPO/"data/generate_contrast.py":
        return False
    note = audit.read("reproducibility/generator_documentation_update.json")
    previous = audit.track(note["original_source"])
    current = audit.track(note["current_source"])
    class WithoutDocstrings(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    node.body = node.body[1:]
            return node
    def executable(source):
        return ast.dump(WithoutDocstrings().visit(ast.parse(source)), include_attributes=False)
    equal = executable(previous.read_text()) == executable(current.read_text())
    audit.check(equal, "generator documentation update has identical executable AST")
    return (equal and sha256(previous) == expected == note["original_sha256"]
            and sha256(current) == note["current_sha256"])


@lru_cache(maxsize=24)
def bootstrap_weights(n, replicates, seed):
    # Reconstruct the declared integer draws, then independently aggregate them
    # as per-field frequency weights. This avoids the producer's indexing helper.
    draws = np.random.default_rng(seed).integers(0, n, size=(replicates, n))
    weights = np.zeros((replicates, n), dtype=float)
    np.add.at(weights, (np.repeat(np.arange(replicates), n), draws.reshape(-1)), 1.)
    return weights/n


def check_statistics(audit, values, expected, weights, name):
    values = np.asarray(values, dtype=float)
    audit.check(values.ndim == 1 and len(values) == weights.shape[1] and np.isfinite(values).all(), name+" finite field vector")
    if values.ndim != 1 or len(values) != weights.shape[1] or not np.isfinite(values).all():
        return
    # Avoid platform BLAS exception-flag leakage on this small reduction.
    means = np.einsum("ij,j->i", weights, values, optimize=False)
    actual = np.r_[values.mean(), np.quantile(means, [.025, .975])]
    wanted = [expected[k] for k in ("mean", "ci95_low", "ci95_high")]
    audit.close(actual, wanted, name+" mean and independently reconstructed percentile CI")


def canonical_fields(audit, frame, n, name):
    audit.check(len(frame) == n and not frame["sample"].duplicated().any(), name+" one row per field")
    audit.check(set(frame["sample"]) == set(range(n)), name+" exact field IDs")
    return frame.sort_values("sample")


def face_quantities(a, field):
    """Independent vectorized TPFA action and positive energy allocation."""
    n = a.shape[0]
    wx = 2*a[:-1]*a[1:]/(a[:-1]+a[1:])*n*n
    wy = 2*a[:, :-1]*a[:, 1:]/(a[:, :-1]+a[:, 1:])*n*n
    dx, dy = field[:-1]-field[1:], field[:, :-1]-field[:, 1:]
    action, energy = np.zeros_like(field), np.zeros_like(field)
    action[:-1] += wx*dx; action[1:] -= wx*dx
    action[:, :-1] += wy*dy; action[:, 1:] -= wy*dy
    energy[:-1] += .5*wx*dx**2; energy[1:] += .5*wx*dx**2
    energy[:, :-1] += .5*wy*dy**2; energy[:, 1:] += .5*wy*dy**2
    for idx in (0, -1):
        action[idx, :] += 2*n*n*a[idx, :]*field[idx, :]
        action[:, idx] += 2*n*n*a[:, idx]*field[:, idx]
        energy[idx, :] += 2*n*n*a[idx, :]*field[idx, :]**2
        energy[:, idx] += 2*n*n*a[:, idx]*field[:, idx]**2
    return action, energy


def load_cache(audit, path):
    with np.load(audit.track(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def audit_primary(audit):
    audit.section = "primary_field_statistics_and_identities"
    directories = PRIMARY+["submission_longtrain_fresh_32"]
    diagnostic_base = Path("results/submission_audit_diagnostics")
    diagnostic_manifest = {"origin": "Audit-derived post-run diagnostics, not the original experiment ledger",
                           "audit_source_sha256": sha256(Path(__file__)), "numpy_version": np.__version__, "per_configuration": {}}
    if audit.artifact_only and (REPO/diagnostic_base/"manifest.json").exists():
        diagnostic_manifest = audit.read(diagnostic_base/"manifest.json")
        audit.check(diagnostic_manifest["audit_source_sha256"] == sha256(Path(__file__)), "audit-derived diagnostics identify this exact audit implementation")
    for directory in directories:
        base = Path("results")/directory
        doc = audit.read(base/"summary.json")
        expected_contrasts = {"100"} if "longtrain" in directory else {"10", "100", "500"}
        audit.check(set(doc["per_kappa"]) == expected_contrasts, directory+" all declared contrasts")
        protocol = doc["protocol"]
        for kappa, block in doc["per_kappa"].items():
            label = f"{directory}/k{kappa}"
            n = block["n_fields"]
            audit.check(n == 200, label+" declared 200 fields")
            weights = bootstrap_weights(n, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
            frame = audit.frame(base/f"metrics_k{kappa}.csv")
            norms = audit.frame(base/f"norms_k{kappa}.csv")
            audit.check(set(frame.method) == EXPECTED_METHODS == set(block["methods"]), label+" complete method coverage")
            audit.check(not frame.duplicated(["sample", "method"]).any() and len(frame) == n*len(EXPECTED_METHODS), label+" rectangular method ledger")
            for method, summaries in block["methods"].items():
                fields = canonical_fields(audit, frame[frame.method == method], n, label+"/"+method)
                for metric, stats in summaries.items():
                    check_statistics(audit, fields[metric], stats, weights, label+"/"+method+"/"+metric)
            wide = frame.pivot(index="sample", columns="method", values="spearman").sort_index()
            for pair, stats in block["paired_spearman_differences"].items():
                left, right = pair.split("_minus_")
                check_statistics(audit, wide[left]-wide[right], stats, weights, label+"/paired/"+pair)
            audit.check(set(norms.target) == {"abs_error", "face_energy", "face_flux_magnitude"}, label+" norm target coverage")
            for target, stats in block["raw_residual_norm_targets"].items():
                fields = canonical_fields(audit, norms[norms.target == target], n, label+"/norm/"+target)
                check_statistics(audit, fields.spearman, stats, weights, label+"/norm/"+target)

            audit.check(block["identity_relative_error_max"] <= 1e-8, label+" reported inverse identity tolerance")
            audit.check(block["energy_allocation_relative_error_max"] <= 1e-10, label+" reported energy identity tolerance")
            audit.close(frame[frame.method == "oracle"].relative_l2_correction_error.max(), block["identity_relative_error_max"], label+" inverse identity / oracle ledger")
            declared = block["provenance"]
            fresh = "fresh" in directory
            audit.check(declared["test_seed"] == (20260912 if fresh else 0) and declared["skip_fields"] == (0 if fresh else 700), label+" saved split metadata")
            audit.check(declared["n_test"] == n and declared["checkpoint_training_seed"] == 0 and len(declared["checkpoint_hashes"]) == 3, label+" saved field/model cardinalities")
            diagnostic_key = directory+"_k"+kappa
            if audit.artifact_only and diagnostic_key in diagnostic_manifest["per_configuration"]:
                record = diagnostic_manifest["per_configuration"][diagnostic_key]
                diagnostic = canonical_fields(audit, audit.frame(record["fields_csv"]), n, label+" audit-derived diagnostics")
                audit.check(record["n_fields"] == n and record["grid_N"] == declared["grid_N"], label+" audit-derived field/grid metadata")
                audit.check(sha256(audit.track(record["fields_csv"])) == record["fields_csv_sha256"], label+" audit-derived CSV provenance hash")
                audit.check((diagnostic.truth_l2_norm > 0).all() and (diagnostic.error_l2_norm > 0).all(), label+" positive diagnostic norms")
                audit.close(diagnostic.relative_l2_prediction_error, diagnostic.error_l2_norm/diagnostic.truth_l2_norm, label+" diagnostic relative-L2 definition")
                check_statistics(audit, diagnostic.relative_l2_prediction_error, block["relative_l2_prediction_error"], weights, label+" audit-derived prediction relative-L2 CI")
                audit.check((diagnostic.same_A_relative_error <= 1e-8).all() and (diagnostic.reference_PDE_relative_error <= 1e-8).all() and (diagnostic.face_energy_identity_relative_error <= 1e-10).all(), label+" retained independent diagnostic tolerances")
                if (REPO/record["prediction_cache"]).exists():
                    audit.check(sha256(audit.track(record["prediction_cache"])) == record["prediction_cache_sha256"], label+" audit-derived diagnostics describe present cache")
            elif audit.artifact_only:
                audit.skip(label+" audit-derived prediction-error CI", "No audit-derived diagnostic ledger is included; full NPZ can reconstruct this statistic.")
            if not audit.available(base/f"predictions_k{kappa}.npz", label+" full cache, independent TPFA/energy replay and input provenance"):
                audit.details[label] = {"fields": n, "methods": len(EXPECTED_METHODS), "saved_statistics_verified": True, "independent_cached_field_replay_verified": False}
                continue

            data = load_cache(audit, base/f"predictions_k{kappa}.npz")
            coeff, forcing, truth = [data[key] for key in ("a", "f", "u")]
            prediction = data["members"].astype(float).mean(axis=0)
            signature = json.loads(str(data["signature"].item()))
            N = signature["grid_N"]
            audit.check(coeff.shape == forcing.shape == truth.shape == prediction.shape == (n, N, N), label+" cache dimensions")
            audit.check(data["members"].shape == (3, n, N, N), label+" exactly three frozen members")
            provenance = block["provenance"]
            audit.check(all(provenance.get(key) == val for key, val in signature.items()), label+" cache/summary provenance agreement")
            audit.check(hashlib.sha256(coeff.tobytes()).hexdigest() == provenance["test_field_sha256"], label+" exact coefficient hash")
            fresh = "fresh" in directory
            audit.check(signature["test_seed"] == (20260912 if fresh else 0) and signature["skip_fields"] == (0 if fresh else 700), label+" declared evaluation split")
            for path, digest in signature["checkpoint_hashes"].items():
                if audit.available(path, label+" frozen checkpoint "+path):
                    audit.check(sha256(audit.track(path)) == digest, label+" frozen checkpoint "+path)
            for key, path in [("generator_sha256", "data/generate_contrast.py"), ("solver_sha256", "solver/assemble_fd.py"), ("model_sha256", "models/tiny_fno.py")]:
                audit.check(source_matches(audit, path, signature[key]), label+" source identity or verified documentation-only snapshot "+key)
            errors = prediction-truth
            relative = np.linalg.norm(errors.reshape(n, -1), axis=1)/np.linalg.norm(truth.reshape(n, -1), axis=1)
            check_statistics(audit, relative, block["relative_l2_prediction_error"], weights, label+"/prediction relative L2")
            same, pde, energy_miss, raw_rhos, energy_rhos = [], [], [], [], []
            for a, f, u, pred, error in zip(coeff, forcing, truth, prediction, errors):
                applied_error, energy = face_quantities(a, error)
                r = face_quantities(a, pred)[0]-f
                same.append(np.linalg.norm(applied_error-r)/max(np.linalg.norm(r), 1e-15))
                pde.append(np.linalg.norm(face_quantities(a, u)[0]-f)/np.linalg.norm(f))
                quadratic = float(np.sum(error*applied_error))
                energy_miss.append(abs(energy.sum()-quadratic)/quadratic)
                raw_rhos.append(spearmanr(np.abs(r).ravel(), np.abs(error).ravel()).statistic)
                energy_rhos.append(spearmanr(np.abs(r).ravel(), energy.ravel()).statistic)
                audit.check(np.isfinite(energy).all() and np.min(energy) >= 0 and quadratic > 0, label+" positive independent face energy")
            audit.check(max(same) <= 1e-8 and max(pde) <= 1e-8, label+" independent same-A and reference PDE residual", same_A_max=max(same), reference_residual_max=max(pde))
            audit.check(max(energy_miss) <= 1e-10 and block["energy_allocation_relative_error_max"] <= 1e-10, label+" independent energy identity", recomputed_max=max(energy_miss))
            audit.check(block["identity_relative_error_max"] <= 1e-8, label+" reported inverse identity tolerance")
            audit.close(frame[frame.method == "oracle"].relative_l2_correction_error.max(), block["identity_relative_error_max"], label+" inverse identity / oracle ledger")
            audit.close(raw_rhos, wide["raw"], label+" independent raw-error correlations", atol=2e-8)
            energy_ledger = norms[norms.target == "face_energy"].sort_values("sample").spearman
            # Tiny roundoff can change ranks of exactly tied low-energy cells.
            audit.close(energy_rhos, energy_ledger, label+" independent face-energy correlations", atol=2e-6)
            costs = block["amg_cost_per_field"]
            audit.check(len(costs) == n and {c["sample"] for c in costs} == set(range(n)), label+" AMG field coverage")
            audit.check(all(c["n_levels"] >= 2 and c["coarsest_unknowns"] < N*N for c in costs), label+" genuine AMG hierarchy")
            audit.details[label] = {"fields": n, "methods": len(EXPECTED_METHODS), "same_A_relative_max": max(same), "reference_PDE_relative_max": max(pde), "face_energy_identity_max": max(energy_miss)}
            if not audit.artifact_only:
                # Separate output namespace: original experimental ledgers are immutable.
                derived = pd.DataFrame({"sample": np.arange(n),
                    "error_l2_norm": np.linalg.norm(errors.reshape(n, -1), axis=1),
                    "truth_l2_norm": np.linalg.norm(truth.reshape(n, -1), axis=1),
                    "error_max_abs": np.abs(errors).reshape(n, -1).max(axis=1),
                    "relative_l2_prediction_error": relative,
                    "same_A_relative_error": same, "reference_PDE_relative_error": pde,
                    "face_energy_identity_relative_error": energy_miss,
                    "independent_raw_error_spearman": raw_rhos,
                    "independent_raw_face_energy_spearman": energy_rhos})
                destination = REPO/diagnostic_base/f"fields_{diagnostic_key}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                derived.to_csv(destination, index=False)
                audit.track(destination)
                cache_path = base/f"predictions_k{kappa}.npz"
                diagnostic_manifest["per_configuration"][diagnostic_key] = {
                    "fields_csv": str(destination.relative_to(REPO)), "fields_csv_sha256": sha256(destination),
                    "prediction_cache": str(cache_path), "prediction_cache_sha256": sha256(REPO/cache_path),
                    "n_fields": n, "grid_N": N, "origin": "independent audit recomputation from frozen cache"}
        scope = "saved statistics; unavailable full inputs explicitly noted" if audit.artifact_only else "all field summaries, paired intervals, and independent identities"
        print(f"Audited {directory}: {scope}", flush=True)
    if not audit.artifact_only:
        path = REPO/diagnostic_base/"manifest.json"
        path.write_text(json.dumps(diagnostic_manifest, indent=2, sort_keys=True)+"\n")
        audit.track(path)


def check_decision_counts(audit, frame, N, name, recall):
    top = int(np.ceil(.1*N*N))
    for col in ("false_high", "false_low", "bound_violations"):
        if col in frame:
            audit.check((frame[col] == 0).all(), name+" "+col+" zero in retained ledger")
    audit.check(((frame.certified_high_count >= 0) & (frame.certified_high_count <= top)).all(), name+" high counts bounded")
    audit.close(frame[recall], frame.certified_high_count/top, name+" recall denominator")
    if "certified_low_count" in frame:
        audit.check(((frame.certified_low_count >= 0) & (frame.certified_low_count <= N*N-top)).all(), name+" low counts bounded")
        audit.close(frame.resolved_fraction, (frame.certified_high_count+frame.certified_low_count)/(N*N), name+" resolved counts")


def audit_rank(audit):
    audit.section = "rank_enclosure_ledgers"
    for primary in PRIMARY:
        directory = primary.replace("submission", "rank_bounds")
        base = Path("results")/directory
        doc = audit.read(base/"summary.json")
        for k, record in doc["per_kappa"].items():
            name = f"{directory}/k{k}"
            n, N = record["n_fields"], record["grid_N"]
            frame = audit.frame(base/f"fields_k{k}.csv")
            weights = bootstrap_weights(n, doc["bootstrap_replicates"], doc["bootstrap_seed"])
            audit.check(set(frame.iterations) == {int(k) for k in record["budgets"]}, name+" budget coverage")
            check_decision_counts(audit, frame, N, name, "certified_high_recall")
            for budget, metrics in record["budgets"].items():
                block = canonical_fields(audit, frame[frame.iterations == int(budget)], n, name+"/"+budget)
                for metric, stats in metrics.items():
                    check_statistics(audit, block[metric], stats, weights, name+"/"+budget+"/"+metric)
            for col, key in [("false_high", "false_high_total"), ("false_low", "false_low_total"), ("bound_violations", "bound_violation_total")]:
                audit.check(int(frame[col].sum()) == record[key] == 0, name+" aggregate "+key)
            audit.close(frame.max_bound_excess_over_error_scale.max(), record["max_numeric_bound_excess_relative"], name+" maximum recorded excess")
            if audit.available(record["prediction_cache"], name+" cache hash and ground-truth tolerance scale"):
                data = load_cache(audit, record["prediction_cache"])
                scales = np.maximum(np.abs(data["members"].astype(float).mean(axis=0)-data["u"]).reshape(n, -1).max(axis=1), 1e-12)
                ratios = 1e-10+1e-13/scales
                audit.check(np.all(frame.max_bound_excess_over_error_scale.to_numpy() <= ratios[frame["sample"]]+1e-15), name+" every raw excess below declared evaluation tolerance")
                audit.check(sha256(audit.track(record["prediction_cache"])) == record["prediction_cache_sha256"], name+" retained cache hash")
            else:
                audit.check(np.all(frame.max_bound_excess_over_error_scale <= 1e-10), name+" retained relative excess below tolerance's universal lower bound")


def audit_ood_rank(audit):
    audit.section = "OOD_rank_enclosure_ledgers"
    base = Path("results/rank_bounds_ood")
    doc = audit.read(base/"summary.json")
    audit.check(set(doc["per_dataset"]) == {"test_id", "test_cs", "test_r", "test_b"}, "all OOD rank datasets")
    if audit.available(doc["input_bundle"], "OOD rank input bundle hash"):
        audit.check(sha256(audit.track(doc["input_bundle"])) == doc["input_bundle_sha256"], "OOD rank input bundle hash")
    for name, record in doc["per_dataset"].items():
        n, N = record["n_sampled_fields"], record["grid_N"]
        frame = audit.frame(base/f"fields_{name}.csv")
        audit.check(sha256(audit.track(base/f"fields_{name}.csv")) == record["per_field_csv_sha256"], name+" retained OOD CSV hash")
        audit.check(set(frame.method) == set(record["methods"]), name+" OOD method coverage")
        weights = bootstrap_weights(n, record["bootstrap_replicates"], record["bootstrap_seed"])
        for method, metrics in record["methods"].items():
            block = canonical_fields(audit, frame[frame.method == method], n, name+"/"+method)
            for metric, stats in metrics.items():
                if isinstance(stats, dict) and "mean" in stats:
                    check_statistics(audit, block[metric].astype(float), stats, weights, name+"/"+method+"/"+metric)
                elif metric.endswith("_total"):
                    audit.check(int(block[metric[:-6]].sum()) == stats, name+"/"+method+"/"+metric)
                elif metric == "max_positive_bound_excess_over_error_scale":
                    audit.close(block[metric].max(), stats, name+"/"+method+" maximum excess")
                elif metric == "executed_iterations_range":
                    audit.close([block.iterations_executed.min(), block.iterations_executed.max()], stats, name+"/"+method+" iteration range")
        for pair, stats in record["paired_spearman_differences"].items():
            left, right = pair.split("_minus_")
            wide = frame.pivot(index="sample", columns="method", values="spearman").sort_index()
            check_statistics(audit, wide[left]-wide[right], stats, weights, name+" paired "+pair)
        decisions = frame[frame.certified_high_count.notna()]
        check_decision_counts(audit, decisions, N, name, "certified_top10_recall")
        for col in ("false_high_count", "false_low_count", "bound_violations_beyond_numeric_tolerance", "magnitude_violations_beyond_numeric_tolerance"):
            audit.check((decisions[col] == 0).all(), name+" "+col)
        audit.check((decisions.max_positive_bound_excess <= decisions.numeric_tolerance).all(), name+" every raw excess within numeric evaluation tolerance")
        audit.check((frame.same_A_identity_relative_error <= 1e-8).all(), name+" OOD same-A tolerance")


def audit_flux_directory(audit, directory):
    audit.section = "flux_enclosure_ledgers"
    base = Path("results")/directory
    doc = audit.read(base/"summary.json")
    config = doc["configuration"]
    # The producer offsets the seed by source enumeration, not sorted JSON keys.
    order = [f"{Path(d).name}_k{k:g}" for d in config["cache_dirs"] for k in config["kappas"]]
    if config["ood_bundle"]:
        order += ["ood_"+key for key in ("test_id", "test_cs", "test_r", "test_b")]
    audit.check(set(order) == set(doc["per_source"]), "flux exact source coverage")
    for name, record in doc["per_source"].items():
        frame = audit.frame(base/f"fields_{name}.csv")
        n, N = record["n_fields"], record["grid_N"]
        weights = bootstrap_weights(n, config["bootstrap"], config["bootstrap_seed"]+order.index(name))
        audit.check(sha256(audit.track(base/f"fields_{name}.csv")) == record["fields_csv_sha256"], name+" field CSV hash")
        audit.check(set(frame.bound) == {"baseline", "equilibrated_flux"}, name+" both bounds")
        check_decision_counts(audit, frame, N, name, "certified_top10_recall")
        audit.check((frame.same_A_identity_relative_error <= 1e-8).all(), name+" same-A tolerance")
        audit.check((frame.positive_flux_excess_over_baseline_bound <= frame.numeric_tolerance).all(), name+" stronger bound ordering tolerance")
        audit.check((frame.baseline_flags_lost_by_flux == 0).all(), name+" no baseline flags lost")
        audit.check((frame.flux_to_baseline_bound_ratio <= 1+1e-12).all(), name+" flux/baseline ratio")
        available = audit.available(record["input"], name+" input hash, coefficient minimum and true-error tolerance scale")
        if available:
            data = load_cache(audit, record["input"])
            if name.startswith("ood_"):
                prefix = name.removeprefix("ood_")
                error = data[prefix+"_u_hat_mean"]-data[prefix+"_u_true"]
                coefficient = data[prefix+"_a"]
            else:
                error = data["members"].astype(float).mean(axis=0)-data["u"]
                coefficient = data["a"]
            minima = coefficient[:n].reshape(n, -1).min(axis=1)
            contrasts = coefficient[:n].reshape(n, -1).max(axis=1)/minima
            audit.close(frame.a_min, minima[frame["sample"]], name+" actual per-field coercivity minimum")
            audit.close(frame.actual_contrast, contrasts[frame["sample"]], name+" actual per-field contrast")
            scales = np.maximum(np.abs(error[:n]).reshape(n, -1).max(axis=1), 1e-12)
            row_scales = scales[frame["sample"]]
            expected_tolerance = 1e-10*row_scales+1e-13
            audit.close(frame.numeric_tolerance, expected_tolerance, name+" evaluation-only numeric tolerance", atol=1e-24, rtol=1e-12)
            audit.check(np.all(frame.max_positive_bound_excess_over_error_scale.to_numpy()*row_scales <= expected_tolerance+1e-16), name+" every raw bound excess within declared tolerance")
        for budget, entries in record["budgets"].items():
            for kind in ("baseline", "equilibrated_flux"):
                block = canonical_fields(audit, frame[(frame.iterations == int(budget)) & (frame.bound == kind)], n, name+"/"+budget+"/"+kind)
                for metric, stats in entries[kind].items():
                    if isinstance(stats, dict) and "mean" in stats:
                        check_statistics(audit, block[metric].astype(float), stats, weights, name+"/"+budget+"/"+kind+"/"+metric)
                    elif metric.endswith("_total"):
                        audit.check(int(block[metric[:-6]].sum()) == stats, name+"/"+budget+"/"+kind+"/"+metric)
                    elif metric == "max_positive_bound_excess_over_error_scale":
                        audit.close(block[metric].max(), stats, name+" maximum excess")
            for metric, stats in entries["paired_flux_minus_baseline"].items():
                wide = frame[frame.iterations == int(budget)].pivot(index="sample", columns="bound", values=metric).astype(float).sort_index()
                check_statistics(audit, wide.equilibrated_flux-wide.baseline, stats, weights, name+"/"+budget+" paired "+metric)
        if available:
            audit.check(sha256(audit.track(record["input"])) == record["input_sha256"], name+" input hash")
    audit.notes.append("Flux raw positive floating-point exceedances are retained and audited separately from tolerance violations; zero tolerance violations does not mean outward-rounded certification.")


def audit_flux(audit):
    for directory in ("flux_bounds_all", "flux_bounds_unet"):
        audit_flux_directory(audit, directory)


def audit_unet(audit):
    """Check the architecture control independently of its summary producer."""
    audit.section = "unet_robustness_statistics_and_predictions"
    base = Path("results/unet_robustness")
    doc = audit.read(base/"summary.json")
    config, protocol, evaluation = doc["configuration"], doc["protocol"], doc["evaluation"]
    n, N = config["n_test"], protocol["grid_N"]
    audit.check(n == 200 and N == 32 and protocol["kappa"] == 100, "U-Net declared fresh grid/contrast/cardinality")
    audit.check(config["n_train"] == 700 and config["seeds"] == [0, 1, 2] and config["epochs"] == 200 and config["width"] == 16, "U-Net fixed training configuration")
    audit.check(protocol["training_data_seed"] == 0 and protocol["test_data_seed"] == 20260912, "U-Net declared training/fresh seeds")
    weights = bootstrap_weights(n, config["bootstrap"], 20260912)
    frame = audit.frame(base/"fields.csv")
    methods = {"raw", "prediction_magnitude", "ensemble", "cg20", "diagonal_pcg20", "amg", "poisson_pcg20"}
    audit.check(set(frame.method) == methods == set(evaluation["methods"]), "U-Net complete seven-method coverage")
    audit.check(len(frame) == n*len(methods) and not frame.duplicated(["sample", "method"]).any(), "U-Net rectangular per-field ledger")
    for method, summaries in evaluation["methods"].items():
        block = canonical_fields(audit, frame[frame.method == method], n, "U-Net/"+method)
        for metric, stats in summaries.items():
            check_statistics(audit, block[metric], stats, weights, "U-Net/"+method+"/"+metric)
        if method in {"cg20", "diagonal_pcg20", "poisson_pcg20"}:
            audit.check(((block.iterations_executed >= 0) & (block.iterations_executed <= 20)).all(), "U-Net/"+method+" bounded executed iterations")
    wide = frame.pivot(index="sample", columns="method", values="spearman").sort_index()
    for pair, stats in evaluation["paired_spearman_differences"].items():
        left, right = pair.split("_minus_")
        check_statistics(audit, wide[left]-wide[right], stats, weights, "U-Net paired "+pair)
    costs = evaluation["amg_cost_per_field"]
    audit.check(len(costs) == n and {c["sample"] for c in costs} == set(range(n)), "U-Net AMG field coverage")
    audit.check(all(c["n_levels"] == 4 and c["coarsest_unknowns"] == 4 for c in costs), "U-Net actual full AMG hierarchy")
    audit.check(evaluation["same_A_identity_relative_error_max"] <= 1e-8, "U-Net reported inverse identity gate")
    for path, expected in doc["source_sha256"].items():
        audit.check(source_matches(audit, path, expected), "U-Net producing source identity: "+path)
    audit.check(doc["source_and_comparison_cache_hashes_unchanged"], "U-Net producer reports stable run inputs")
    audit.check(len(doc["checkpoints"]) == 3, "U-Net three frozen checkpoints")
    all_checkpoints = True
    for j, row in enumerate(doc["checkpoints"]):
        history = row["training_history"]
        audit.check(row["seed"] == config["seeds"][j] and row["checkpoint_roundtrip_max_abs"] == 0, "U-Net seed and exact checkpoint round trip")
        audit.check([h["epoch"] for h in history] == list(range(1, config["epochs"]+1)), "U-Net full fixed-epoch training history")
        audit.check(all(np.isfinite(h["normalized_training_mse"]) and h["normalized_training_mse"] >= 0 for h in history), "U-Net finite training-only losses")
        path = Path(row["path"])
        if audit.available(path, "U-Net frozen checkpoint "+str(path)):
            audit.check(sha256(audit.track(path)) == row["sha256"], "U-Net frozen checkpoint hash "+str(path))
        else:
            all_checkpoints = False
        if audit.available(path.with_suffix(".json"), "U-Net checkpoint metadata "+str(path)):
            meta = audit.read(path.with_suffix(".json"))
            audit.check(meta["seed"] == row["seed"] and meta["epochs"] == config["epochs"] and meta["width"] == config["width"], "U-Net checkpoint sidecar training configuration")
            audit.check(meta["normalization"] == doc["normalization"] and meta["source_sha256"] == doc["source_sha256"] and meta["training_arrays_sha256"] == doc["training_arrays_sha256"], "U-Net checkpoint sidecar normalization and training provenance")
    diagnostic_base = Path("results/submission_audit_diagnostics")
    diagnostic_path = diagnostic_base/"fields_unet_robustness_k100.csv"
    manifest_path = diagnostic_base/"unet_manifest.json"
    if audit.artifact_only:
        if audit.available(manifest_path, "U-Net audit-derived relative-L2 ledger provenance"):
            manifest = audit.read(manifest_path)
            audit.check(manifest["audit_source_sha256"] == sha256(Path(__file__)), "U-Net audit-derived diagnostics identify this audit implementation")
            diagnostic = canonical_fields(audit, audit.frame(diagnostic_path), n, "U-Net audit-derived prediction diagnostics")
            audit.check(sha256(audit.track(diagnostic_path)) == manifest["fields_csv_sha256"], "U-Net audit-derived CSV provenance hash")
            audit.close(diagnostic.relative_l2_prediction_error, diagnostic.error_l2_norm/diagnostic.truth_l2_norm, "U-Net retained relative-L2 definition")
            check_statistics(audit, diagnostic.relative_l2_prediction_error, evaluation["relative_l2_prediction_error"], weights, "U-Net retained prediction relative-L2 CI")
            audit.close(diagnostic.independent_raw_error_spearman, wide.raw, "U-Net retained independent raw-error correlations", atol=2e-8)
            if (REPO/manifest["prediction_cache"]).exists():
                audit.check(sha256(audit.track(manifest["prediction_cache"])) == manifest["prediction_cache_sha256"], "U-Net diagnostic cache provenance")
    cache_path = base/"predictions_k100.npz"
    has_cache = audit.available(cache_path, "U-Net all-field numerical prediction/identity replay")
    comparison_path = doc["frozen_fno_comparison_cache"]
    has_comparison = audit.available(comparison_path, "U-Net identical fresh-field FNO comparison")
    if has_comparison:
        audit.check(sha256(audit.track(comparison_path)) == doc["frozen_fno_comparison_cache_sha256"], "U-Net comparison cache hash")
    if has_cache:
        data = load_cache(audit, cache_path)
        a, f, truth, members = [data[key] for key in ("a", "f", "u", "members")]
        prediction = members.astype(float).mean(axis=0)
        audit.check(a.shape == f.shape == truth.shape == prediction.shape == (n, N, N) and members.shape == (3, n, N, N), "U-Net cached array dimensions")
        for name, value in [("a", a), ("f", f), ("truth", truth), ("members", members)]:
            audit.check(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == doc["evaluation_arrays_sha256"][name], "U-Net exact numerical array hash: "+name)
        signature = json.loads(str(data["signature"].item()))
        audit.check(signature["configuration"] == config and signature["normalization"] == doc["normalization"], "U-Net cache/summary configuration and normalization")
        audit.check(signature["checkpoint_sha256"] == [row["sha256"] for row in doc["checkpoints"]], "U-Net prediction cache identifies frozen members")
        if has_comparison:
            comparison = load_cache(audit, comparison_path)
            for key, value in [("a", a), ("f", f), ("u", truth)]:
                audit.check(np.array_equal(value, comparison[key]), "U-Net exact same FNO fresh "+key+" fields")
        errors = prediction-truth
        error_norm = np.linalg.norm(errors.reshape(n, -1), axis=1)
        truth_norm = np.linalg.norm(truth.reshape(n, -1), axis=1)
        relative = error_norm/truth_norm
        check_statistics(audit, relative, evaluation["relative_l2_prediction_error"], weights, "U-Net independent all-field prediction relative-L2 CI")
        for j, seed in enumerate(config["seeds"]):
            member_relative = np.linalg.norm((members[j]-truth).reshape(n, -1), axis=1)/truth_norm
            audit.close(member_relative.mean(), evaluation["per_member_relative_l2"][str(seed)], "U-Net independently recomputed member error "+str(seed))
        sigma = members.astype(float).std(axis=0)
        same, pde, energies, raw_rhos = [], [], [], []
        for i in range(n):
            applied, energy = face_quantities(a[i], errors[i])
            residual = face_quantities(a[i], prediction[i])[0]-f[i]
            same.append(np.linalg.norm(applied-residual)/max(np.linalg.norm(residual), 1e-15))
            pde.append(np.linalg.norm(face_quantities(a[i], truth[i])[0]-f[i])/np.linalg.norm(f[i]))
            quadratic = np.sum(errors[i]*applied)
            energies.append(abs(energy.sum()-quadratic)/quadratic)
            scores = {"raw": np.abs(residual), "prediction_magnitude": np.abs(prediction[i]), "ensemble": sigma[i]}
            for method, score in scores.items():
                rho = spearmanr(score.ravel(), np.abs(errors[i]).ravel()).statistic
                audit.close(rho, wide.loc[i, method], f"U-Net field {i} independent {method} Spearman", atol=2e-8)
                if method == "raw": raw_rhos.append(rho)
        audit.check(max(same) <= 1e-8 and max(pde) <= 1e-8 and max(energies) <= 1e-10, "U-Net independent same-A/PDE/energy tolerances", same_A_max=max(same), PDE_max=max(pde), energy_max=max(energies))
        for method in ("cg20", "diagonal_pcg20", "poisson_pcg20", "amg"):
            block = frame[frame.method == method].sort_values("sample")
            audit.close(block.corrected_prediction_relative_l2, relative*block.relative_l2_correction_error, "U-Net corrected/uncorrected norm identity: "+method)
        if not audit.artifact_only:
            diagnostic = pd.DataFrame({"sample": np.arange(n), "error_l2_norm": error_norm, "truth_l2_norm": truth_norm,
                "relative_l2_prediction_error": relative, "same_A_relative_error": same, "reference_PDE_relative_error": pde,
                "face_energy_identity_relative_error": energies, "independent_raw_error_spearman": raw_rhos})
            (REPO/diagnostic_base).mkdir(parents=True, exist_ok=True)
            diagnostic.to_csv(REPO/diagnostic_path, index=False)
            manifest = {"origin": "Audit-derived post-run diagnostics, not the original experiment ledger", "audit_source_sha256": sha256(Path(__file__)),
                "fields_csv": str(diagnostic_path), "fields_csv_sha256": sha256(REPO/diagnostic_path), "prediction_cache": str(cache_path),
                "prediction_cache_sha256": sha256(REPO/cache_path), "n_fields": n, "grid_N": N}
            (REPO/manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
            audit.track(diagnostic_path); audit.track(manifest_path)
    verifier = audit.track("analysis/verify_unet_artifacts.py")
    normalization_path = "results/checkpoints/fno_kappa100_seed0.json"
    has_normalization = audit.available(normalization_path, "U-Net independent FNO training-normalization reference")
    if has_cache and has_comparison and all_checkpoints and has_normalization:
        audit.track(normalization_path)
        process = subprocess.run([sys.executable, str(verifier)], cwd=REPO, text=True, capture_output=True)
        audit.check(process.returncode == 0, "U-Net independent checkpoint prediction verifier", output=(process.stdout+process.stderr)[-2500:])
        report = audit.read(base/"artifact_verification.json")
        audit.check(report["verification_source_sha256"] == sha256(verifier) and report["n_fields"] == n and report["fresh_fields_match_fno_cache_exactly"], "U-Net verifier provenance and fresh-field comparison")
        audit.check(len(report["checks"]) == 3 and all(c["max_abs_prediction_difference"] == 0 for c in report["checks"]), "U-Net all three checkpoints reproduce all saved predictions exactly")
        for path, expected in report["input_sha256"].items():
            audit.check(sha256(audit.track(path)) == expected, "U-Net independent verifier input hash: "+path)
    else:
        audit.skip("U-Net exact checkpoint/prediction consistency", "At least one required checkpoint, cached prediction, comparison cache or reference-normalization input is unavailable; retained verifier report alone is not a rerun.")
    audit.details["unet_robustness"] = {"fields": n, "methods": len(methods), "saved_CIs_reconstructed": True,
        "independent_all_field_raw_and_relative_L2_verified": has_cache,
        "checkpoint_verifier_executed": has_cache and has_comparison and all_checkpoints and has_normalization}


def audit_adaptive(audit):
    audit.section = "adaptive_stopping_ledgers"
    for directory in ("adaptive_rank_stopping", "adaptive_rank_flux"):
        base = Path("results")/directory
        doc = audit.read(base/"summary.json")
        protocol = doc["protocol"]
        for name, record in doc["results"].items():
            n, N = record["n_fields"], record["grid_N"]
            top = int(np.ceil(protocol["q"]*N*N))
            target = int(np.ceil(protocol["target_certified_high_recall"]*top))
            frame = canonical_fields(audit, audit.frame(base/f"fields_{name}.csv"), n, directory+"/"+name)
            weights = bootstrap_weights(n, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
            check_decision_counts(audit, frame, N, name, "certified_high_recall")
            for metric, stats in record["metrics"].items():
                check_statistics(audit, frame[metric], stats, weights, directory+"/"+name+"/"+metric)
            audit.check(np.array_equal(frame.target_met.astype(bool), frame.certified_high_count >= target), name+" target computed from strict count")
            audit.check(np.array_equal(frame.target_met.astype(bool), frame.status == "rank_target"), name+" status / target agreement")
            audit.check((frame.iterations <= protocol["maxiter"]).all(), name+" maximum budget")
            audit.close(frame.fft_transforms, 2*(frame.preconditioner_applications+frame.bound_poisson_solves), name+" FFT accounting")
            audit.close(frame.extra_operator_applications, frame.bound_poisson_solves, name+" bound/operator accounting")
            audit.check({str(k): v for k, v in Counter(frame.iterations).items()} == record["iteration_distribution"], name+" iteration histogram")
            audit.check(dict(Counter(frame.status)) == record["status_counts"], name+" status histogram")
            audit.close(frame.iterations.median(), record["iteration_median"], name+" iteration median")
            audit.close(frame.iterations.quantile(.9), record["iteration_p90"], name+" iteration p90")
            histories = audit.read(base/f"histories_{name}.json")
            audit.check(len(histories) == n and {h["sample"] for h in histories} == set(range(n)), name+" complete histories")
            for history in histories:
                row = frame.iloc[history["sample"]]
                steps = history["history"]
                audit.check(bool(steps), name+" nonempty stopping history")
                if not steps:
                    continue
                audit.check(all(s["certified_high_count"] < target for s in steps[:-1]), name+" stopped at first successful recorded check")
                audit.check(steps[-1]["iterations"] == row.iterations and steps[-1]["certified_high_count"] == row.certified_high_count, name+" final history matches returned decision")
                audit.check(all(s["iterations"] % protocol["check_every"] == 0 for s in steps[:-1]), name+" declared check cadence")
                audit.check(len(steps) == row.bound_poisson_solves, name+" each recorded check accounted")
            if audit.available(record["cache"], name+" adaptive frozen cache hash"):
                audit.check(sha256(audit.track(record["cache"])) == record["cache_sha256"], name+" frozen cache hash")


@lru_cache(maxsize=2)
def independent_reference(N):
    one = np.diag(np.full(N, 2.))+np.diag(np.full(N-1, -1.), 1)+np.diag(np.full(N-1, -1.), -1)
    one[0, 0] += 1; one[-1, -1] += 1
    eigen, Q = np.linalg.eigh(one*N*N)
    inverse = 1/(eigen[:, None]+eigen[None, :])
    def multiply(left, right):
        return np.einsum("ik,kj->ij", left, right, optimize=False)
    diagonal = multiply(multiply(Q**2, inverse), (Q**2).T)
    def solve(rhs):
        transformed = multiply(multiply(Q.T, np.asarray(rhs).reshape(N, N)), Q)
        return multiply(multiply(Q, transformed*inverse), Q.T)
    return solve, diagonal


def independent_enclosure(a, rhs, z, kind):
    N = a.shape[0]
    solve, diagonal = independent_reference(N)
    defect = rhs-face_quantities(a, z)[0]
    p = solve(defect)
    if kind == "poisson":
        beta = np.sqrt(np.maximum(diagonal*np.sum(defect*p), 0))/a.min()
    else:
        wx = 2*a[:-1]*a[1:]/(a[:-1]+a[1:])
        wy = 2*a[:, :-1]*a[:, 1:]/(a[:, :-1]+a[:, 1:])
        energy = N*N*(np.sum(np.diff(p, axis=0)**2/wx)+np.sum(np.diff(p, axis=1)**2/wy))
        energy += 2*N*N*sum(np.sum(p[idx, :]**2/a[idx, :])+np.sum(p[:, idx]**2/a[:, idx]) for idx in (0, -1))
        beta = np.sqrt(np.maximum(diagonal*energy/a.min(), 0))
    lower, upper = np.maximum(np.abs(z)-beta, 0).ravel(), (np.abs(z)+beta).ravel()
    top = int(np.ceil(.1*N*N))
    high = lower > np.sort(upper)[::-1][top]
    low = upper < np.sort(lower)[::-1][top-1]
    return beta, high, low


def audit_decision_replay(audit):
    audit.section = "independent_selected_field_decision_replay"
    # Selection is fixed by index, never by success, rank quality or bound size.
    replay_count = 0
    for directory in PRIMARY:
        for k in (10, 100, 500):
            data = load_cache(audit, Path("results")/directory/f"predictions_k{k}.npz")
            prediction = data["members"].astype(float).mean(axis=0)
            adaptive_rows = {}
            for adaptive, kind in (("adaptive_rank_stopping", "poisson"), ("adaptive_rank_flux", "flux")):
                path = Path("results")/adaptive/f"fields_{directory}_k{k}.csv"
                if (REPO/path).exists():
                    adaptive_rows[kind] = audit.frame(path).set_index("sample")
            for i in (0, len(prediction)//2, len(prediction)-1):
                a, truth, pred = data["a"][i], data["u"][i], prediction[i]
                N, n = a.shape[0], a.size
                rhs = face_quantities(a, pred)[0]-data["f"][i]
                error = pred-truth
                solve, _ = independent_reference(N)
                operator = LinearOperator((n, n), matvec=lambda x: face_quantities(a, x.reshape(N, N))[0].ravel(), dtype=float)
                preconditioner = LinearOperator((n, n), matvec=lambda x: solve(x).ravel(), dtype=float)
                top = int(np.ceil(.1*n)); true_top = np.zeros(n, bool)
                true_top[np.argsort(np.abs(error).ravel())[-top:]] = True
                tolerance = 1e-10*max(float(np.abs(error).max()), 1e-12)+1e-13
                budgets = {5, 10, 20, 40} | {int(rows.loc[i, "iterations"]) for rows in adaptive_rows.values()}
                for budget in sorted(budgets):
                    z, info = cg(operator, rhs.ravel(), M=preconditioner, maxiter=budget, rtol=8*np.finfo(float).eps, atol=0)
                    audit.check(info >= 0 and np.isfinite(z).all(), f"{directory}/{k}/{i}/{budget} independent solve")
                    for kind in ("poisson", "flux"):
                        beta, high, low = independent_enclosure(a, rhs, z.reshape(N, N), kind)
                        label = f"{directory}/k{k}/field{i}/budget{budget}/{kind}"
                        audit.check(not np.any(high & ~true_top) and not np.any(low & true_top), label+" independently recomputed strict flags")
                        audit.check(np.max(np.abs(error-z.reshape(N, N))-beta) <= tolerance, label+" independently recomputed enclosure tolerance")
                        if kind in adaptive_rows and budget == adaptive_rows[kind].loc[i, "iterations"]:
                            target = int(np.ceil(.9*top))
                            audit.check(bool(high.sum() >= target) == bool(adaptive_rows[kind].loc[i, "target_met"]), label+" independently replayed adaptive target")
                        replay_count += 1
    audit.details["independent_decision_replay"] = {"field_selection": "indices 0, n//2, n-1 in all 12 grid/contrast/split conditions", "budgets": "5,10,20,40 plus the selected fields' actual adaptive stopping iterations", "bound_variants": ["poisson", "flux"], "evaluations": replay_count, "independence": "vectorized face action, dense 1D eigensystem Poisson inverse and independently sorted interval thresholds; no production bound or decision helper imported"}


def audit_split_evidence(audit):
    audit.section = "training_split_evidence"
    split = audit.read("results/submission_split_audit.json")
    audit.check(len(split["per_configuration"]) == 6, "six reconstructed grid/contrast split configurations")
    for name, block in split["per_configuration"].items():
        audit.check(block["unique_training_coefficient_fields"] == 700 and block["original_fresh_overlap"] == 0, name+" training and evaluation split ledger")
        for part, item in block["evaluation"].items():
            audit.check(item["n_fields"] == 200 and item["training_overlap"] == 0 and item["reconstruction_exact"], name+"/"+part+" reconstructed disjoint coefficient draws")
        for checkpoint in block["checkpoints"]:
            audit.check(checkpoint["sidecar_agrees_with_embedded_metadata"] and max(checkpoint["input_normalization_abs_difference"].values()) <= 1e-12, name+" checkpoint normalization evidence")
    for name, digest in {**split["input_sha256"], **split["source_sha256"]}.items():
        if audit.available(name, "split evidence input hash: "+name):
            audit.check(source_matches(audit, name, digest), "split evidence remains current: "+name)
    audit.notes.extend(split["limitations"])


def audit_submission_files(audit):
    audit.section = "generated_assets_and_submission_format"
    completed = subprocess.run([sys.executable, "analysis/build_submission_assets.py", "--check"], cwd=REPO, text=True, capture_output=True)
    audit.check(completed.returncode == 0, "builder --check macros/table/source hashes", output=(completed.stdout+completed.stderr)[-3000:])
    for path in ("paper/generated/numbers.tex", "paper/generated/localization_table.tex", "paper/generated/numerical_sources.json"):
        audit.track(path)
    official = audit.track("paper/neurips_2026.official.sty").read_text()
    modified = audit.track("paper/neurips_2026.sty").read_text()
    old = r"Submitted to \@neuripsordinal\/ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Do not distribute."
    audit.check(sha256(REPO/"paper/neurips_2026.official.sty") == ORIGINAL_STYLE_HASH, "exact independently obtained official style checksum")
    audit.check(official.count(old) == 1 and modified == official.replace(old, FOOTER), "only authorized footer differs from official style")
    tex = audit.track("paper/draft_ml4ps.tex").read_text()
    audit.check(r"\usepackage{neurips_2026}" in tex and not re.search(r"\\usepackage\[[^\]]*(?:final|preprint)", tex), "anonymous official submission style mode")
    audit.check(r"\author{Anonymous Author(s)}" in tex, "anonymous author declaration")
    audit.check(not re.search(r"\\(?:usepackage(?:\[[^\]]*\])?\{geometry\}|setlength\{\\(?:textwidth|textheight|oddsidemargin|topmargin)\}|renewcommand\{\\(?:normalsize|baselinestretch)\})", tex), "no manuscript template-dimension/font overrides")
    pdf = audit.track("paper/draft_ml4ps.pdf")
    proc = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True)
    audit.check(proc.returncode == 0, "PDF text extractable", stderr=proc.stderr[-1000:])
    pages = proc.stdout.split("\f")
    if pages and not pages[-1].strip(): pages.pop()
    references = [(i, re.search(r"(?m)^\s*(?:\d+\s+)?References\s*$", page)) for i, page in enumerate(pages)]
    references = [(i, match) for i, match in references if match]
    audit.check(bool(references), "PDF reference heading found")
    if references:
        i, match = references[0]
        before = pages[i][:match.start()]
        meaningful = re.sub(r"(?m)^\s*\d+\s*$", "", before).strip()
        body_pages = i+1 if meaningful else i
        audit.check(body_pages <= 4, "main content no more than four pages", main_content_pages=body_pages, total_pdf_pages=len(pages))
        audit.details["PDF"] = {"total_pages": len(pages), "last_main_content_page": body_pages, "reference_start_page": i+1}
    audit.check("??" not in proc.stdout, "PDF has no unresolved-reference placeholders")
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}", (REPO/"paper/generated/numbers.tex").read_text()))
    used_commands = set(re.findall(r"\\([A-Za-z]+)", tex))
    for name in sorted(used_commands & macros.keys()):
        value = macros[name]
        if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            audit.check(bool(re.search(r"(?<![\d.])"+re.escape(value)+r"(?!\d|\.\d)", proc.stdout)), "PDF contains generated numeric value for "+name, expected_value=value)
    local_names = {getpass.getuser(), Path.home().name}
    private_name = any(re.search(r"\b"+re.escape(name)+r"\b", proc.stdout, re.I) for name in local_names if len(name) > 2)
    audit.check(not private_name and not re.search(r"/Users/|github\.com/(?!anonymous)[\w.-]+/", proc.stdout, re.I), "PDF text contains no detected local username/path/nonanonymous GitHub identity")
    audit.check(" ".join(FOOTER.split()) in " ".join(proc.stdout.split()), "correct workshop footer visible in PDF")
    if audit.available("paper/draft_ml4ps.log", "latest LaTeX log undefined-reference/error scan"):
        log = audit.track("paper/draft_ml4ps.log").read_text(errors="replace")
        bad = re.findall(r"[^\n]*(?:Undefined control sequence|undefined references|Citation .+ undefined|Reference .+ undefined|! LaTeX Error)[^\n]*", log)
        audit.check(not bad, "latest LaTeX log has no undefined references or errors", messages=bad[:15])
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    author = re.search(r"(?m)^Author:[ \t]*(.*)$", info.stdout)
    audit.check(author is None or author.group(1).strip() in ("", "Anonymous Author(s)"), "PDF author metadata anonymous", author=author.group(1) if author else None)


def audit_timing(audit):
    audit.section = "independent_controlled_timing_audit"
    audit.track("analysis/audit_submission_timing.py")
    command = [sys.executable, "analysis/audit_submission_timing.py"]
    if audit.artifact_only: command.append("--artifact-only")
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    audit.check(completed.returncode == 0, "independent raw-timing/bootstrap checker", output=(completed.stdout+completed.stderr)[-2500:])
    report = audit.read("results/submission_timing_audit.json")
    audit.check(report["status"] in ({"PASS", "ARTIFACT_ONLY_PASS"} if audit.artifact_only else {"PASS"}), "retained timing artifact checks passed")
    for item in report.get("skipped_inputs", []):
        audit.skip("timing full input verification", str(item))
    audit.check(sha256(audit.track("results/submission_timing_flux/benchmark.json")) == report["benchmark_sha256"], "timing audit describes final flux-inclusive benchmark")
    audit.track("results/submission_timing_flux/raw_times.csv")
    audit.details["timing_audit"] = {"checks": report["checks"], "report": "results/submission_timing_audit.json", "scope": report["scope"]}


def audit_fault_detection(audit):
    audit.section = "auditor_fault_injection"
    probe = Audit()
    check_statistics(probe, np.arange(8.), {"mean": 0., "ci95_low": 0., "ci95_high": 0.}, bootstrap_weights(8, 100, 42), "deliberately corrupted summary")
    audit.check(bool(probe.failures), "auditor rejects a corrupted field statistic")
    probe = Audit()
    canonical_fields(probe, pd.DataFrame({"sample": [0, 0, 2]}), 3, "deliberately duplicated field")
    audit.check(bool(probe.failures), "auditor rejects a duplicated/missing field")
    probe = Audit()
    check_decision_counts(probe, pd.DataFrame({"false_high": [1], "false_low": [0], "bound_violations": [0], "certified_high_count": [1], "certified_high_recall": [0.]}), 4, "deliberately corrupted decision", "certified_high_recall")
    audit.check(len(probe.failures) >= 2, "auditor rejects false flags and inconsistent top-set counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO/"results/submission_artifact_audit.json")
    parser.add_argument("--skip-replay", action="store_true", help="Diagnostic quick pass only; records that independent decision replay was omitted")
    parser.add_argument("--artifact-only", action="store_true", help="Verify saved statistics and submission artifacts; explicitly report unavailable frozen inputs as not verified, never as a full pass")
    args = parser.parse_args()
    started = time.monotonic()
    audit = Audit(artifact_only=args.artifact_only)
    audit.track(Path(__file__))
    with threadpool_limits(limits=1):
        stages = [audit_fault_detection, audit_primary, audit_unet, audit_rank, audit_ood_rank, audit_flux, audit_adaptive, audit_split_evidence, audit_timing]
        if not args.skip_replay and not args.artifact_only: stages.append(audit_decision_replay)
        else: audit.skip("independent selected-field decision replay", "Explicit artifact-only/skip-replay mode; full numerical replay is not verified in this run.")
        stages.append(audit_submission_files)
        for stage in stages:
            try:
                stage(audit)
            except Exception as exc:
                audit.check(False, stage.__name__+" completed", exception=f"{type(exc).__name__}: {exc}")
            print(f"{stage.__name__}: {len(audit.failures)} cumulative failures", flush=True)
    audit.section = "audit_input_stability"
    for name, digest in audit.inputs.items():
        audit.check(sha256(REPO/name) == digest, "unchanged during audit: "+name)
    full_pass = not audit.failures and not audit.skipped and not args.artifact_only
    status = "FAIL" if audit.failures else "PASS" if full_pass else "ARTIFACT_ONLY_PASS"
    payload = {"experiment": "independent_submission_artifact_audit", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": status, "mode": "artifact_only" if args.artifact_only else "full", "passed": full_pass,
               "full_verification_passed": full_pass, "artifact_checks_passed": not audit.failures,
               "skipped_not_verified": audit.skipped, "checks": sum(audit.counts.values()), "checks_by_section": dict(audit.counts),
               "failures": audit.failures, "details": audit.details, "notes_and_limits": audit.notes,
               "input_sha256": audit.inputs, "audit_source_sha256": sha256(Path(__file__)), "wall_time_s": time.monotonic()-started,
               "scope": "All primary original/fresh32/64 and U-Net per-field summaries and paired CIs; all retained rank/flux/adaptive ledgers including U-Net flux sensitivity; independent identity/energy recomputation; declared selected-field decision replay; U-Net exact checkpoint/prediction verifier; split audit evidence hashes; generated assets and machine-checkable PDF/template requirements. Visual quality and historical decisions still require human review."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"{status}: {payload['checks']} checks, {len(audit.failures)} failures, {len(audit.skipped)} not verified; {args.out}")
    if audit.failures:
        for failure in audit.failures[:15]: print(json.dumps(failure))
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
