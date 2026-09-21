"""Build anonymous review and frozen-input archives from an explicit allowlist.

Original research artifacts are never edited. Package copies remove local path
prefixes and update transitive file-hash references; the derivative manifest
records every changed byte stream. Numerical arrays and checkpoint payloads are
unchanged; a named NPZ metadata signature may have local path prefixes removed.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["solver", "models", "data", "uncertainty", "experiments", "analysis", "tests", "reproducibility"]
RESULT_DIRS = [
    "submission_32", "submission_64", "submission_fresh_32", "submission_fresh_64",
    "rank_bounds_32", "rank_bounds_64", "rank_bounds_fresh_32", "rank_bounds_fresh_64",
    "rank_bounds_ood", "flux_bounds_all", "flux_bounds_unet", "adaptive_rank_stopping", "adaptive_rank_flux",
    "submission_timing", "submission_timing_singlethread", "submission_timing_flux",
    "ood_audit_fullamg_20260912", "submission_longtrain_fresh_32", "unet_robustness",
    "submission_audit_diagnostics",
    "checkpoints", "checkpoints_64", "checkpoints_longtrain",
]
TOP_FILES = ["README.md", "REPRODUCIBILITY.md", "SUPPLEMENT_METHODS.md", "requirements-lock.txt", "Makefile", "pytest.ini",
             "paper/draft_ml4ps.tex", "paper/draft_ml4ps.pdf", "paper/draft_ml4ps.log",
             "paper/references.bib", "paper/neurips_2026.sty", "paper/neurips_2026.official.sty",
             "paper/visual_references_20260912.md", "paper/independent_citation_review_20260912.md",
             "paper/FIGURE_GUIDE.md", "paper/feedback_revision_review.json",
             "paper/citations_primary_snapshot.json", "paper/citation_check.json",
             "paper/independent_citation_registry_20260912.json",
             "results/ood_features_bundle.npz", "results/submission_split_audit.json",
             "results/submission_artifact_audit.json", "results/submission_timing_audit.json",
             "results/submission_replay_comparison.json", "results/frozen_error_contrast.json",
             "results/adaptive_current_replay_comparison.json",
             "results/frozen_error_contrast.csv", "results/fno_longtrain_kappa100.json",
             "results/fno_longtrain_kappa100.csv", "results/exp9_ood_conformal.json",
             "results/exp10_ood_abstention.json", "results/check_damped_jacobi.json",
             "results/real_fno_multi_kappa_amgfix.json", "results/real_fno_64x64_amgfix.json"]
TEXT_SUFFIXES = {".py", ".json", ".csv", ".md", ".txt", ".tex", ".sty", ".svg", ".log", ".patch"}
REPRESENTATIVE_CACHE = "results/submission_fresh_32/predictions_k100.npz"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files():
    files = {REPO/name for name in TOP_FILES if (REPO/name).is_file()}
    for name in SOURCE_DIRS:
        files.update(p for p in (REPO/name).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix in TEXT_SUFFIXES)
    for name in RESULT_DIRS:
        files.update(p for p in (REPO/"results"/name).rglob("*") if p.is_file()
                     and p.suffix in TEXT_SUFFIXES|{".npz", ".pt", ".pdf", ".png", ".svg"})
    files.update(p for p in (REPO/"paper/generated").glob("*") if p.is_file())
    files.update(p for p in (REPO/"paper/figures").glob("submission_*") if p.is_file())
    return sorted(files, key=lambda p: str(p.relative_to(REPO)))


def has_identity(text):
    tokens = [str(REPO), str(Path.home())]
    if any(token in text for token in tokens):
        return True
    username = getpass.getuser()
    return bool(len(username) > 2 and re.search(r"\b"+re.escape(username)+r"\b", text, re.I))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, val in value.items():
            yield from strings(key)
            yield from strings(val)
    elif isinstance(value, (tuple, list)):
        for val in value:
            yield from strings(val)


def copy_binary_anonymous(path, target):
    changed = []
    if path.suffix == ".npz":
        arrays = {}
        with np.load(path, allow_pickle=False) as cache:
            for name in cache.files:
                values = cache[name]
                if values.dtype.kind in "US":
                    if any(has_identity(str(value)) for value in values.reshape(-1)):
                        if name not in {"signature", "metadata"} or values.ndim != 0:
                            raise ValueError(f"identifying non-metadata string: {path.name}/{name}")
                        raw = str(values.item())
                        json.loads(raw)
                        clean = sanitize(raw)
                        json.loads(clean)
                        if has_identity(clean):
                            raise ValueError(f"identifying metadata remains: {path.name}/{name}")
                        values = np.asarray(clean)
                        changed.append(name)
                elif values.dtype.kind == "O":
                    raise ValueError(f"uninspected object array: {path.name}/{name}")
                arrays[name] = values
        if changed:
            np.savez_compressed(target, **arrays)
            with np.load(path, allow_pickle=False) as before, np.load(target, allow_pickle=False) as after:
                for name in before.files:
                    if name not in changed and not np.array_equal(before[name], after[name]):
                        raise AssertionError(f"numerical array changed during anonymization: {name}")
        else:
            shutil.copy2(path, target)
    elif path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if any(has_identity(value) for value in strings(payload)):
            raise ValueError(f"identifying checkpoint metadata: {path.name}")
        shutil.copy2(path, target)
    return changed


def sanitize(text):
    return text.replace(str(REPO)+"/", "").replace(str(REPO), ".").replace(str(Path.home())+"/", "~/")


def remap_hashes(stage, originals):
    """Maintain referential checksums after path-only JSON transformations."""
    aliases = {digest: name for name, digest in originals.items()}
    for _ in range(12):
        current = {str(p.relative_to(stage)): sha(p) for p in stage.rglob("*") if p.is_file()}
        aliases.update({digest: name for name, digest in current.items()})
        changed = False
        def rewrite(value):
            if isinstance(value, str) and value in aliases:
                return current.get(aliases[value], value)
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            if isinstance(value, dict):
                return {k: rewrite(v) for k, v in value.items()}
            return value
        for path in stage.rglob("*.json"):
            original_text = path.read_text()
            payload = json.loads(original_text)
            updated = rewrite(payload)
            if updated != payload:
                path.write_text(json.dumps(updated, indent=2, sort_keys=True, allow_nan=False)+"\n")
                changed = True
        if not changed:
            return
    raise RuntimeError("checksum dependency graph did not stabilize")


def run_checked(command, stage, label):
    process = subprocess.run(command, cwd=stage, text=True, capture_output=True,
                             env={**os.environ, "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
                                  "VECLIB_MAXIMUM_THREADS":"1", "MKL_NUM_THREADS":"1"})
    if process.returncode:
        raise RuntimeError(f"{label} failed:\n"+(process.stdout+process.stderr)[-7000:])
    print(label+": PASS", flush=True)


def write_archive(path, stage, files):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(files):
            name = "ml4ps2026-review/"+str(file.relative_to(stage))
            info = zipfile.ZipInfo(name, date_time=(2026,9,12,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, file.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"archive CRC failure: {bad}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO/"submission")
    parser.add_argument("--allow-incomplete", action="store_true", help="Development-only package before the U-Net robustness run completes")
    parser.add_argument("--skip-full-audit", action="store_true", help="Development-only staging; final package requires the full audit")
    args = parser.parse_args()
    if not args.allow_incomplete and not (REPO/"results/unet_robustness/summary.json").exists():
        raise FileNotFoundError("U-Net robustness summary is required for the final package")
    (REPO/"tmp").mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="anonymous_review_", dir=REPO/"tmp"))
    original_hashes = {}
    binary_metadata_derivatives = {}
    for source in selected_files():
        name = str(source.relative_to(REPO))
        target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hashes[name] = sha(source)
        if source.suffix in {".npz", ".pt"}:
            changed = copy_binary_anonymous(source, target)
            if changed:
                binary_metadata_derivatives[name] = changed
        elif source.suffix in TEXT_SUFFIXES or source.name == "Makefile":
            target.write_text(sanitize(source.read_text()))
        else:
            shutil.copy2(source, target)
    remap_hashes(stage, original_hashes)
    (stage/"REVIEW_PACKAGE.md").write_text("""# Anonymous review package

This archive contains the manuscript, editable vector figures, implementation,
tests, per-field diagnostic records and an independent artifact checker.
The companion frozen-inputs archive supplies the remaining coefficient/prediction
caches and model checkpoints. Extract both archives into the same parent folder
for complete frozen-model verification or replay. The small archive retains one
N32/contrast100 cache so the manuscript figures can be rebuilt directly.

Install requirements-lock.txt, then run:

```sh
python -m pytest -q -W error::RuntimeWarning
python analysis/build_submission_assets.py --check
python analysis/audit_submission_citations.py
python analysis/audit_submission_artifacts.py --artifact-only
```

Artifact-only success explicitly does not certify omitted input replays. With
the companion archive extracted, omit --artifact-only for the full audit.
REPRODUCIBILITY.md documents retraining and fresh-output-directory replay.
The packaging manifest distinguishes original research bytes from anonymous
derivatives: local absolute paths were removed from text metadata, transitive
file checksums were updated, and numerical arrays/checkpoints were unchanged.
No paper or code was uploaded automatically.
""")
    run_checked([sys.executable, "analysis/build_submission_assets.py"], stage, "anonymous generated assets")
    run_checked([sys.executable, "analysis/audit_submission_citations.py"], stage, "anonymous citation audit")
    if not args.skip_full_audit:
        run_checked([sys.executable, "analysis/audit_submission_artifacts.py"], stage, "anonymous full artifact audit")
        run_checked([sys.executable, "-m", "pytest", "-q", "-W", "error::RuntimeWarning"], stage, "anonymous test suite")
    # Audit tools may record their newly resolved staging paths. Normalize only
    # distribution copies after verification, then maintain their hash links.
    before_cleanup = {str(p.relative_to(stage)): sha(p) for p in stage.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts}
    for path in stage.rglob("*"):
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == "Makefile"):
            original_text = path.read_text()
            clean = sanitize(original_text.replace(str(stage)+"/", "").replace(str(stage), "."))
            if clean != original_text:
                path.write_text(clean)
    remap_hashes(stage, before_cleanup)
    all_files = [p for p in stage.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and ".pytest_cache" not in p.parts and ".cache" not in p.parts]
    for path in all_files:
        if path.suffix in TEXT_SUFFIXES or path.name == "Makefile":
            if has_identity(path.read_text()):
                raise ValueError(f"remaining identifying text: {path.relative_to(stage)}")
    transformed = {str(p.relative_to(stage)): {"original_sha256": original_hashes.get(str(p.relative_to(stage))),
                    "packaged_sha256":sha(p)} for p in all_files
                   if original_hashes.get(str(p.relative_to(stage))) != sha(p)}
    manifest = {"scope":"Anonymous derivative of retained evidence; numerical arrays and checkpoints unchanged; named NPZ metadata signatures may have path-only edits",
                "full_staging_audit_performed":not args.skip_full_audit, "incomplete_development_build":args.allow_incomplete,
                "binary_metadata_derivatives":binary_metadata_derivatives,
                "transformations":transformed,
                "files":{str(p.relative_to(stage)):sha(p) for p in all_files}}
    manifest_path = stage/"PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    all_files.append(manifest_path)
    drifted = [name for name, digest in original_hashes.items() if sha(REPO/name) != digest]
    if drifted:
        raise RuntimeError("source artifacts changed during packaging: "+", ".join(drifted))
    selected_now = {str(path.relative_to(REPO)) for path in selected_files()}
    if selected_now != set(original_hashes):
        raise RuntimeError("source selection changed during packaging: "+
                           ", ".join(sorted(selected_now.symmetric_difference(original_hashes))))
    inputs = [p for p in all_files if p.suffix in {".npz", ".pt"} and str(p.relative_to(stage)) != REPRESENTATIVE_CACHE]
    review = [p for p in all_files if p not in inputs]
    args.out.mkdir(parents=True, exist_ok=True)
    review_zip, inputs_zip = [args.out/name for name in ["ml4ps2026_review.zip", "ml4ps2026_frozen_inputs.zip"]]
    write_archive(review_zip, stage, review)
    write_archive(inputs_zip, stage, inputs)
    report = {"status":"PASS", "full_staging_audit_performed":not args.skip_full_audit,
              "incomplete_development_build":args.allow_incomplete,
              "staging_directory":str(stage.relative_to(REPO)),
              "archives":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in [review_zip,inputs_zip]},
              "review_files":len(review), "input_files":len(inputs)}
    (args.out/"package_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
