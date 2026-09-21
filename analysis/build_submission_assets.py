"""Build the workshop's numerical text and figures from fresh-field artifacts.

--check detects drift in every generated numerical LaTeX fragment and source
hash manifest. Scientific quantities are never copied by hand into the paper.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from solver.assemble_fd import assemble
from solver.multigrid import vcycle
from solver.poisson import poisson_preconditioner
from uncertainty.flux_bounds import poisson_flux_bounds
from uncertainty.rank_bounds import magnitude_envelope, separate_top_set
from analysis.submission_figure_style import map_figure, budget_figure

OUT = REPO / "paper" / "generated"
FIG = REPO / "paper" / "figures"
SOURCES = [REPO / f"results/{prefix}_{n}/summary.json"
           for prefix in ["submission_fresh", "rank_bounds_fresh"] for n in [32, 64]]
SOURCES += [REPO / "results/flux_bounds_all/summary.json",
            REPO / "results/submission_timing_flux/benchmark.json",
            REPO / "results/adaptive_rank_stopping/summary.json",
            REPO / "results/adaptive_rank_flux/summary.json",
            REPO / "results/submission_longtrain_fresh_32/summary.json",
            REPO / "results/unet_robustness/summary.json"]
COLORS = {"jacobi": "#DD8452", "cg": "#4C72B0", "pcg": "#55A868", "poisson_pcg": "#8172B3"}
LABELS = {"jacobi": "Jacobi", "cg": "CG", "pcg": "diagonal PCG", "poisson_pcg": "Poisson-PCG"}
FIGURE_INPUTS = SOURCES + [REPO / name for name in [
    "analysis/build_submission_assets.py", "analysis/submission_figure_style.py",
    "solver/assemble_fd.py", "solver/poisson.py", "uncertainty/rank_bounds.py",
    "uncertainty/flux_bounds.py", "results/submission_fresh_32/predictions_k100.npz",
    "results/submission_fresh_32/metrics_k100.csv"]]


def hashes(paths):
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def number(x, decimals=3):
    return str(Decimal(str(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def interval(summary, decimals=3):
    return f"{number(summary['mean'], decimals)} [{number(summary['ci95_low'], decimals)}, {number(summary['ci95_high'], decimals)}]"


def load_all():
    docs = [json.loads(p.read_text()) for p in SOURCES[:4]]
    return docs[0]["per_kappa"], docs[1]["per_kappa"], docs[2]["per_kappa"], docs[3]["per_kappa"]


def numerical_text(m32, m64, c32, c64):
    macros = {}
    for n, doc in [(32, m32), (64, m64)]:
        for k, label in [("10", "Ten"), ("100", "Hundred"), ("500", "FiveHundred")]:
            for method, name in [("raw", "Raw"), ("prediction_magnitude", "Magnitude"),
                                 ("pcg20", "Diagonal"), ("poisson_pcg20", "Poisson"), ("amg", "AMG")]:
                macros[f"{name}{'Small' if n == 32 else 'Large'}{label}"] = number(doc[k]["methods"][method]["spearman"]["mean"])
            macros[f"Energy{'Small' if n == 32 else 'Large'}{label}"] = number(doc[k]["raw_residual_norm_targets"]["face_energy"]["mean"])
    for n, doc in [(32, c32), (64, c64)]:
        for k, label in [("100", "Hundred"), ("500", "FiveHundred")]:
            for budget, word in [("20", "Twenty"), ("40", "Forty")]:
                stats = doc[k]["budgets"][budget]["certified_high_recall"]
                stem = f"Recall{'Small' if n == 32 else 'Large'}{label}{word}"
                macros[stem] = number(100 * stats["mean"], 1)
                macros[stem + "Low"] = number(100 * stats["ci95_low"], 1)
                macros[stem + "High"] = number(100 * stats["ci95_high"], 1)
    macros["PoissonLargeFiveHundredPrecise"] = number(m64["500"]["methods"]["poisson_pcg20"]["spearman"]["mean"], 4)
    macros["PoissonCorrectionRemainingPercent"] = number(100 * m64["500"]["methods"]["poisson_pcg20"]["relative_l2_correction_error"]["mean"], 2)
    overlap = m64["500"]["methods"]["poisson_pcg20"]["top10_overlap"]
    macros["PoissonTopOverlapPercent"] = number(100 * overlap["mean"], 2)
    macros["PoissonTopOverlapPercentLow"] = number(100 * overlap["ci95_low"], 2)
    macros["PoissonTopOverlapPercentHigh"] = number(100 * overlap["ci95_high"], 2)
    for k, name in [("10", "Ten"), ("100", "Hundred"), ("500", "FiveHundred")]:
        macros[f"PredictionRelativePercent{name}"] = number(100 * m32[k]["relative_l2_prediction_error"]["mean"], 2)
    flux = json.loads(SOURCES[4].read_text())["per_source"]["submission_fresh_64_k500"]
    for k, word in [(20, "Twenty"), (40, "Forty")]:
        stats = flux["budgets"][str(k)]["equilibrated_flux"]["certified_top10_recall"]
        stem = f"FluxRecallLargeFiveHundred{word}"
        macros[stem] = number(100 * stats["mean"], 1)
        macros[stem + "Low"] = number(100 * stats["ci95_low"], 1)
        macros[stem + "High"] = number(100 * stats["ci95_high"], 1)
    timings = json.loads(SOURCES[5].read_text())["results"]
    for n, grid in [(32, "Small"), (64, "Large")]:
        for method, label in [("poisson_pcg20", "PoissonTime"), ("adaptive_rank90", "AdaptiveTime"), ("amg", "AMGTime")]:
            stats = timings[f"N{n}_k500"]["methods"][method]
            stem = f"{label}{grid}"
            macros[stem] = number(stats["median_paired_ratio_to_fresh_oracle"], 3)
            macros[stem + "Low"] = number(stats["ratio_ci95_low"], 3)
            macros[stem + "High"] = number(stats["ratio_ci95_high"], 3)
        stats = timings[f"N{n}_k500"]["adaptive_flux_vs_poisson"]
        macros[f"FluxAdaptiveTimeRatio{grid}"] = number(stats["median_paired_time_ratio"], 3)
    for path, label in [(SOURCES[6], "Adaptive"), (SOURCES[7], "FluxAdaptive")]:
        stats = json.loads(path.read_text())["results"]["submission_fresh_64_k500"]
        macros[f"{label}IterationsLargeFiveHundred"] = number(stats["metrics"]["iterations"]["mean"], 1)
    longer = json.loads(SOURCES[8].read_text())["per_kappa"]["100"]
    macros["LongtrainRelativeLTwoPercent"] = number(100 * longer["relative_l2_prediction_error"]["mean"], 2)
    macros["LongtrainRawRho"] = number(longer["methods"]["raw"]["spearman"]["mean"])
    unet = json.loads(SOURCES[9].read_text())["evaluation"]
    macros["UnetRelativeLTwoPercent"] = number(100 * unet["relative_l2_prediction_error"]["mean"], 2)
    macros["UnetRawRho"] = number(unet["methods"]["raw"]["spearman"]["mean"])
    text = "% Generated by analysis/build_submission_assets.py; do not edit.\n"
    text += "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n"

    table = ["% Generated from fresh32 summary; brackets are 95% field-bootstrap CIs.",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Score / correction & $\kappa=10$ & $\kappa=100$ & $\kappa=500$ \\", r"\midrule"]
    for method, label in [("raw", r"raw $|r|$"), ("ensemble", r"ensemble $\sigma$"),
                          ("prediction_magnitude", r"prediction $|\hat u|$"),
                          ("boundary_distance", "boundary distance"),
                          ("jacobi20", "Jacobi-20"), ("cg20", "CG-20"),
                          ("pcg20", "diagonal PCG-20"), ("poisson", r"$|B^{-1}r|$"),
                          ("amg", "one AMG V-cycle"), ("poisson_pcg20", "Poisson-PCG-20")]:
        values = [interval(m32[k]["methods"][method]["spearman"]) for k in ["10", "100", "500"]]
        table.append(label + " & " + " & ".join(values) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}", ""]
    return {"numbers.tex": text, "localization_table.tex": "\n".join(table)}


def save_figure(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def figures(m32, m64, c32, c64):
    # Pick a field using only proximity to the median raw-residual score.
    metrics = pd.read_csv(REPO / "results/submission_fresh_32/metrics_k100.csv")
    block = metrics[metrics.method == "raw"].set_index("sample")
    sample = int((block.spearman - block.spearman.median()).abs().idxmin())
    with np.load(REPO / "results/submission_fresh_32/predictions_k100.npz", allow_pickle=False) as data:
        a, f, u = (data[name][sample] for name in ["a", "f", "u"])
        pred = data["members"][:, sample].astype(np.float64).mean(axis=0)
    A = assemble(a)
    error = np.abs(pred-u)
    r = A @ pred.reshape(-1) - f.reshape(-1)
    z, info = cg(A, r, M=poisson_preconditioner(32), maxiter=20, rtol=8*np.finfo(float).eps, atol=0)
    if info < 0 or not np.isfinite(z).all():
        raise RuntimeError("representative correction failed")
    bounds = poisson_flux_bounds((r - A @ z).reshape(a.shape), a)
    lower, upper = magnitude_envelope(z, bounds.bound.reshape(-1))
    high, low = separate_top_set(lower, upper, q=.1)
    correction_rho = float(metrics[(metrics.method == "poisson_pcg20") & (metrics["sample"] == sample)].iloc[0].spearman)
    fig = map_figure(a, error, np.abs(r).reshape(a.shape), np.abs(z).reshape(a.shape),
                     high, low, float(block.loc[sample, "spearman"]), correction_rho)
    save_figure(fig, "submission_maps")
    representative = {"sample": sample, "rule": "closest to median raw Spearman on fresh kappa100/32 fields",
                      "raw_rho": float(block.loc[sample, "spearman"]), "correction_rho": correction_rho,
                      "membership_bound": "equilibrated flux, strict top-10 separation",
                      "inside_cells": int(high.sum()), "outside_cells": int(low.sum()),
                      "unresolved_cells": int((~(high | low)).sum())}
    flux = json.loads(SOURCES[4].read_text())["per_source"]["submission_fresh_64_k500"]
    save_figure(budget_figure(m64, c64, flux), "submission_budget_bounds")
    return representative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    docs = load_all()
    for path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
    artifacts = numerical_text(*docs)
    source_hashes = hashes(SOURCES)
    artifacts["numerical_sources.json"] = json.dumps(source_hashes, indent=2, sort_keys=True) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check:
        bad = [name for name, content in artifacts.items() if not (OUT / name).exists() or (OUT / name).read_text() != content]
        if bad:
            raise SystemExit("STALE GENERATED ASSETS: " + ", ".join(bad))
        manifest_path = OUT / "figure_manifest.json"
        if not manifest_path.exists():
            raise SystemExit("MISSING FIGURE MANIFEST: rebuild submission assets")
        manifest = json.loads(manifest_path.read_text())
        if manifest["inputs"] != hashes(FIGURE_INPUTS):
            raise SystemExit("STALE FIGURE INPUTS: rebuild submission assets")
        for name, digest in manifest["outputs"].items():
            path = REPO / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise SystemExit("STALE FIGURE OUTPUT: " + name)
        print("PASS: generated numerical text, source hashes and figure bytes are consistent")
        return
    for name, content in artifacts.items():
        (OUT / name).write_text(content)
    representative = figures(*docs)
    (OUT / "representative_field.json").write_text(json.dumps(representative, indent=2) + "\n")
    outputs = [FIG / f"{name}.{ext}" for name in ["submission_maps", "submission_budget_bounds"]
               for ext in ["pdf", "png", "svg"]] + [OUT / "representative_field.json"]
    (OUT / "figure_manifest.json").write_text(json.dumps({
        "inputs": hashes(FIGURE_INPUTS), "outputs": hashes(outputs),
        "rendering": "Original-resolution field heatmaps embedded without interpolation; schematic geometry, labels, curves and math remain vector objects.",
    }, indent=2, sort_keys=True) + "\n")
    print("Built workshop numerical LaTeX and vector figures from retained result summaries")


if __name__ == "__main__":
    main()
