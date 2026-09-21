"""Generate the compact predictor/coefficient robustness table from verified data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/"paper/generated"
SOURCES = [REPO/name for name in [
    "results/submission_fresh_32/summary.json",
    "results/submission_longtrain_fresh_32/summary.json",
    "results/unet_robustness/summary.json",
    "results/smooth_robustness/summary.json",
    "results/smooth_robustness/verification.json",
]]


def hashes(paths):
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def interval(row, scale=1, places=3):
    return f"{scale*row['mean']:.{places}f} [{scale*row['ci95_low']:.{places}f}, {scale*row['ci95_high']:.{places}f}]"


def build():
    documents = [json.loads(path.read_text()) for path in SOURCES]
    if documents[4]["status"] != "PASS":
        raise ValueError("smooth artifact verification must pass before paper integration")
    records = [documents[0]["per_kappa"]["100"], documents[1]["per_kappa"]["100"],
               documents[2]["evaluation"], documents[3]["evaluation"]]
    labels = ["Binary FNO (80)", "Binary FNO (240)", "Binary U-Net (200)", "Smooth FNO (80)"]
    lines = ["% Generated from independently verified retained ensemble summaries.",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Predictor (epochs) & Rel. $L^2$ error (\%) & Residual $\rho$ & PCG-20 $\rho$ \\",
             r"\midrule"]
    for label, record in zip(labels, records):
        values = [interval(record["relative_l2_prediction_error"], scale=100, places=2),
                  interval(record["methods"]["raw"]["spearman"]),
                  interval(record["methods"]["poisson_pcg20"]["spearman"], places=5)]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    smooth = records[-1]
    specifications = {
        "SmoothRawRho": (smooth["methods"]["raw"]["spearman"], 1, 3),
        "SmoothPoissonRho": (smooth["methods"]["poisson_pcg20"]["spearman"], 1, 4),
        "SmoothRelativePercent": (smooth["relative_l2_prediction_error"], 100, 2),
        "SmoothPairedRhoGain": (smooth["paired_spearman_differences"]["poisson_pcg20_minus_raw"], 1, 3),
        "SmoothTopOverlapPercent": (smooth["methods"]["poisson_pcg20"]["top10_overlap"], 100, 1),
        "SmoothCorrectionRemainingPercent": (smooth["methods"]["poisson_pcg20"]["relative_l2_correction_error"], 100, 2),
        "SmoothFluxRecallPercent": (smooth["bounds"]["20"]["equilibrated_flux"]["certified_top10_recall"], 100, 1),
    }
    macros = {}
    for name, (record, scale, places) in specifications.items():
        for suffix, key in [("", "mean"), ("Low", "ci95_low"), ("High", "ci95_high")]:
            macros[name+suffix] = f"{scale*record[key]:.{places}f}"
    numbers = "% Generated from results/smooth_robustness/summary.json.\n" + "".join(
        rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in sorted(macros.items()))
    return {"robustness_table.tex": "\n".join(lines), "robustness_numbers.tex": numbers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contents = build()
    expected = {"inputs": hashes(SOURCES+[Path(__file__).resolve()]),
                "outputs": {str((OUT/name).relative_to(REPO)): hashlib.sha256(content.encode()).hexdigest()
                            for name, content in contents.items()},
                "scope": "Independent smooth-family retraining; intervals conditional on each trained ensemble. Binary/smooth comparisons are descriptive and unpaired."}
    manifest = OUT/"robustness_manifest.json"
    if args.check:
        if json.loads(manifest.read_text()) != expected:
            raise SystemExit("STALE robustness source manifest")
        for name, content in contents.items():
            if (OUT/name).read_text() != content:
                raise SystemExit(f"STALE robustness asset {name}")
        print("PASS: robustness table, numerical macros and source manifest")
    else:
        for name, content in contents.items():
            (OUT/name).write_text(content)
        manifest.write_text(json.dumps(expected, indent=2, sort_keys=True)+"\n")
        print("Built robustness table and numerical macros")


if __name__ == "__main__":
    main()
