"""Regenerate Tables 1–2 and Figures 1–2 without changing retained evidence.

The small review archive suffices. Existing builders run in an isolated copy;
no training, manuscript compilation, or full numerical audit is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
BUILDERS = ["build_submission_assets.py", "build_alternate_figure.py", "build_robustness_assets.py"]
MANIFESTS = ["figure_manifest.json", "alternate_figure_manifest.json", "robustness_manifest.json"]
NUMERICAL_OUTPUTS = ["numbers.tex", "localization_table.tex", "robustness_numbers.tex",
                     "robustness_table.tex", "representative_field.json"]
PAPER_OUTPUTS = {
    "Table 1": ["paper/generated/localization_table.tex"],
    "Table 2": ["paper/generated/robustness_table.tex"],
    "Figure 1": [f"paper/figures/submission_maps_alternate.{ext}" for ext in ("pdf", "svg", "png")],
    "Figure 2": [f"paper/figures/submission_budget_bounds.{ext}" for ext in ("pdf", "svg", "png")],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_files():
    names = {"analysis/reproduce_paper_assets.py", "analysis/submission_figure_style.py"}
    names.update("analysis/" + name for name in BUILDERS)
    # Read the builders' declared inputs rather than duplicate a list of
    # scientific result directories in this convenience wrapper.
    for name in MANIFESTS:
        manifest = json.loads((REPO / "paper/generated" / name).read_text())
        names.update(manifest["inputs"])
    for directory in ("solver", "uncertainty"):
        names.update(str(path.relative_to(REPO)) for path in (REPO / directory).glob("*.py"))
    for name in sorted(names):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Expected a repository-relative input: " + name)
        if not (REPO / name).is_file():
            raise FileNotFoundError("Missing review input: " + name)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "tmp/reproduced_paper",
                        help="New output directory; an existing directory is never overwritten")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("Choose a new --out directory; already exists: " + str(output))
    started = time.perf_counter()
    names = input_files()
    inputs = {name: sha(REPO / name) for name in names}
    numerical = {"paper/generated/" + name: sha(REPO / "paper/generated" / name)
                 for name in NUMERICAL_OUTPUTS}
    frozen_pdfs = {str(path.relative_to(REPO)): sha(path) for path in
                   (REPO / "paper").glob("draft_ml4ps*.pdf")}
    original_figures = {name: sha(REPO / name) for group in ("Figure 1", "Figure 2")
                        for name in PAPER_OUTPUTS[group]}
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    records = []
    with tempfile.TemporaryDirectory(prefix=".asset_build_", dir=output) as temporary:
        stage = Path(temporary)
        for name in names:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(stage / ".cache/mpl")}
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[name] = "1"
        for check_only in (False, True):
            for name in BUILDERS:
                command = [sys.executable, "analysis/" + name] + (["--check"] if check_only else [])
                before = time.perf_counter()
                run = subprocess.run(command, cwd=stage, env=env, text=True, capture_output=True)
                label = name.removesuffix(".py") + ("_check" if check_only else "")
                transcript = (run.stdout + run.stderr).replace(str(stage), "<isolated-build>")
                (logs / (label + ".log")).write_text(transcript)
                records.append({"command": ["python"] + command[1:], "exit_code": run.returncode,
                                "wall_time_s": time.perf_counter() - before})
                if run.returncode:
                    raise RuntimeError(label + " failed; see " + str(logs / (label + ".log")))
                print(label + ": PASS", flush=True)
        for name, expected in numerical.items():
            if sha(stage / name) != expected:
                raise ValueError("Regenerated numerical content differs from retained paper: " + name)
        for directory in ("generated", "figures"):
            shutil.copytree(stage / "paper" / directory, output / "paper" / directory)
    for name, expected in {**inputs, **numerical, **frozen_pdfs, **original_figures}.items():
        if sha(REPO / name) != expected:
            raise RuntimeError("Retained input changed during reproduction: " + name)
    figures_identical = {name: sha(output / name) == expected for name, expected in original_figures.items()}
    generated = {str(path.relative_to(output)): sha(path) for path in (output / "paper").rglob("*") if path.is_file()}
    report = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Regenerated tables, numerical LaTeX and figures from retained records; numerical text and representative-field decisions match exactly. This is not training or a full experiment audit.",
              "paper_outputs": PAPER_OUTPUTS, "input_sha256": inputs, "output_sha256": generated,
              "numerical_outputs_identical": True, "retained_manuscript_pdfs_unchanged": frozen_pdfs,
              "figure_files_byte_identical": figures_identical,
              "rendering_note": "Renderer/font/library differences can change figure bytes across environments; inspect the per-file comparisons.",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "cpu_threads_requested": 1,
                              "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")}},
              "commands": records, "wall_time_s": time.perf_counter() - started}
    (output / "REPRODUCTION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    matches = sum(figures_identical.values())
    print(f"PASS: Tables 1–2 and Figures 1–2 regenerated in {report['wall_time_s']:.2f}s; "
          f"numerical content unchanged; {matches}/{len(figures_identical)} figure files byte-identical.", flush=True)
    print("Outputs: " + str(output), flush=True)


if __name__ == "__main__":
    main()
