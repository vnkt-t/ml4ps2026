"""Vector-first, data-grounded figures for the four-page ML4PS manuscript.

The perspective grid is explanatory geometry only. All quantitative field maps
use the original cell values on flat, aligned axes; no interpolation is applied.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np

INK = "#29343B"
MUTED = "#68767A"
TEAL = "#246B72"
RUST = "#AE583F"
OCHRE = "#A17A36"
BLUE = "#506B91"
PALE = "#E8E8E3"
FIELD = LinearSegmentedColormap.from_list("paper_teal", ["#F5F4EF", "#B6D0CA", "#659D9C", "#246B72", "#143C4C"])
RESIDUAL = LinearSegmentedColormap.from_list("paper_rust", ["#F5F4EF", "#E2C7AD", "#C18C65", "#AE583F", "#61372E"])
COEFFICIENT = LinearSegmentedColormap.from_list("paper_stone", ["#F0EEE7", "#B5A58C"])


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.linewidth": .55,
        "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "ml4ps-residual-rank-2026",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def _arrow(ax, start, stop, color=MUTED, style="->", lw=.7, scale=7):
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle=style, mutation_scale=scale,
                               linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def _plane(ax, origin, width=.15, height=.30, color="#ECE8DE", alpha=1):
    o = np.array(origin)
    vx, vy = np.array([width, 0.]), np.array([.035, height])
    ax.add_patch(Polygon([o, o+vx, o+vx+vy, o+vy], facecolor=color,
                         edgecolor="#89908D", lw=.65, alpha=alpha))
    for t in np.linspace(0, 1, 6)[1:-1]:
        x, y = np.stack([o+t*vx, o+t*vx+vy]).T
        ax.plot(x, y, color="#B7BCB5", lw=.35)
        x, y = np.stack([o+t*vy, o+t*vy+vx]).T
        ax.plot(x, y, color="#B7BCB5", lw=.35)


def map_figure(a, error, residual, correction, high, low, raw_rho, correction_rho):
    """A compact technical workflow above five honest cell-centered maps."""
    set_style()
    fig = plt.figure(figsize=(6.8, 2.54))
    flow = fig.add_axes([.02, .62, .96, .36])
    flow.set(xlim=(0, 1), ylim=(0, 1)); flow.axis("off")
    # A modest axonometric grid recalls a technical drawing without distorting
    # the measured maps below it.
    _plane(flow, (.025, .20), width=.11, height=.30)
    _plane(flow, (.025, .40), width=.11, height=.30, color="#E0ECE7")
    flow.text(.025, .91, r"inputs $a,\hat u$", fontsize=11)
    _arrow(flow, (.16, .51), (.255, .51))
    flow.text(.305, .70, "residual", ha="center", fontsize=11, color=INK)
    flow.text(.305, .43, r"$r=A\hat u-b$", ha="center", fontsize=12)
    _arrow(flow, (.39, .51), (.485, .51))
    flow.text(.535, .70, "correction", ha="center", fontsize=11, color=INK)
    flow.text(.535, .43, r"$z\approx A^{-1}r$", ha="center", fontsize=12)
    flow.text(.535, .28, "Poisson-PCG", ha="center", fontsize=10, color=TEAL)
    _arrow(flow, (.635, .51), (.725, .51))
    flow.text(.680, .75, r"$d=r-Az$", ha="center", fontsize=10.5, color=INK)
    flow.text(.850, .78, "enclosure", ha="center", fontsize=11, color=INK)
    flow.plot([.755, .945], [.47, .47], color=TEAL, lw=1.2)
    flow.plot([.755, .755, np.nan, .945, .945], [.39, .55, np.nan, .39, .55], color=TEAL, lw=.8)
    flow.scatter([.85], [.47], s=10, color=TEAL, zorder=3)
    flow.text(.755, .28, r"$L_i$", ha="center", fontsize=10)
    flow.text(.85, .28, r"$|z_i|$", ha="center", fontsize=10)
    flow.text(.945, .28, r"$U_i$", ha="center", fontsize=10)
    flow.plot([.004, .995], [.015, .015], color="#CDD0CA", lw=.55)

    lefts = [.022, .220, .418, .616, .814]
    fields = [a, error, residual, correction]
    titles = [r"(a) Coefficient $a$", r"(b) Error $|e|$",
              r"(c) Residual $|r|$", r"(d) Correction $|z_{20}|$", "(e) Top-10% set"]
    vmax = float(error.max())
    for j, left in enumerate(lefts):
        ax = fig.add_axes([left, .18, .164, .43])
        if j < 4:
            field = fields[j]
            lo, hi = ((1., float(a.max())) if j == 0 else (0., float(field.max()) if j == 2 else vmax))
            cmap = COEFFICIENT if j == 0 else RESIDUAL if j == 2 else FIELD
            # Native-resolution heatmap pixels avoid polygon hairline artifacts
            # in PDF viewers. Geometry, math and labels remain vector objects.
            mesh = ax.imshow(field.T, origin="lower", interpolation="none", resample=False,
                             cmap=cmap, vmin=lo, vmax=hi, extent=(0, a.shape[0], 0, a.shape[0]))
            cax = fig.add_axes([left+.013, .118, .138, .018])
            bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[lo, hi])
            bar.outline.set_visible(False)
            bar.ax.tick_params(length=1.5, width=.4, pad=1.3, labelsize=9)
            bar.ax.set_xticklabels([f"{lo:g}", f"{hi:.0f}" if hi >= 10 else f"{hi:.2g}"])
        else:
            state = np.ones_like(a, dtype=int)
            state[low.reshape(a.shape)] = 0
            state[high.reshape(a.shape)] = 2
            ax.imshow(state.T, origin="lower", interpolation="none", resample=False,
                      cmap=ListedColormap(["#BED3CF", PALE, RUST]), vmin=0, vmax=2,
                      extent=(0, a.shape[0], 0, a.shape[0]))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(.45); spine.set_color("#959D99")
        ax.set_title(titles[j], loc="center", fontsize=9.5, pad=5)
    fig.text(.500, .004, rf"$\rho={raw_rho:.3f}$", ha="center", fontsize=11, color=RUST)
    fig.text(.697, .004, rf"$\rho={correction_rho:.4f}$", ha="center", fontsize=11, color=TEAL)
    for y, color, label in [(.123, RUST, "inside"), (.078, PALE, "unresolved"), (.033, "#BED3CF", "outside")]:
        fig.add_artist(Polygon([[.83, y-.006], [.842, y-.006], [.842, y+.012], [.83, y+.012]],
                               transform=fig.transFigure, facecolor=color, edgecolor="#A9AEAA", lw=.35))
        fig.text(.85, y, label, fontsize=9, va="center")
    return fig


def budget_figure(m64, c64, flux):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.60))
    fig.subplots_adjust(left=.067, right=.985, bottom=.35, top=.86, wspace=.30)
    budgets = np.array([5, 10, 20, 40, 80])
    specs = [("jacobi", "Jacobi", OCHRE, "s", (0, (1, 1.6))),
             ("cg", "CG", BLUE, "^", (0, (4, 1.8))),
             ("pcg", "diagonal PCG", MUTED, "D", (0, (4, 1.5, 1, 1.5))),
             ("poisson_pcg", "Poisson-PCG", TEAL, "o", "-")]
    ax = axes[0]
    for key, label, color, marker, linestyle in specs:
        stats = [m64["500"]["methods"][f"{key}{k}"]["spearman"] for k in budgets]
        ax.plot(budgets, [s["mean"] for s in stats], marker=marker, ls=linestyle,
                markersize=3.6, lw=1.1, label=label, color=color)
        ax.fill_between(budgets, [s["ci95_low"] for s in stats], [s["ci95_high"] for s in stats], color=color, alpha=.10, lw=0)
    amg = m64["500"]["methods"]["amg"]["spearman"]["mean"]
    ax.axhline(amg, color=RUST, ls=(0, (3, 2)), lw=.9)
    ax.text(5.4, amg-.035, "one AMG cycle", color=RUST, fontsize=9, va="top")
    ax.set_title(r"(a) Error ranking: $64^2$, $\kappa=500$", loc="left", pad=9)
    ax.set_ylabel("Mean spatial Spearman", labelpad=4)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.37), ncol=2, frameon=False,
              borderpad=0, labelspacing=.23, handlelength=2.2, fontsize=9, columnspacing=1.0)
    ax.set_ylim(0, 1.025); ax.set_yticks([0, .25, .5, .75, 1])

    ax = axes[1]
    overlap = [m64["500"]["methods"][f"poisson_pcg{k}"]["top10_overlap"] for k in budgets]
    cert = [c64["500"]["budgets"][str(k)]["certified_high_recall"] for k in budgets]
    for stats, color, label, ls, marker in [(overlap, TEAL, "actual overlap", "-", "o"),
                                          (cert, MUTED, "Poisson enclosure", (0, (4, 1.5, 1, 1.5)), "s")]:
        ax.plot(budgets, [100*s["mean"] for s in stats], marker=marker, markersize=3.6,
                lw=1.1, color=color, label=label, ls=ls)
        ax.fill_between(budgets, [100*s["ci95_low"] for s in stats], [100*s["ci95_high"] for s in stats], color=color, alpha=.10, lw=0)
    flux_stats = [flux["budgets"][str(k)]["equilibrated_flux"]["certified_top10_recall"] for k in [20, 40]]
    ax.errorbar([20, 40], [100*s["mean"] for s in flux_stats],
                yerr=np.array([[100*(s["mean"]-s["ci95_low"]) for s in flux_stats],
                               [100*(s["ci95_high"]-s["mean"]) for s in flux_stats]]),
                color=RUST, marker="D", markersize=4, ls="--", lw=.9, capsize=2,
                label="equilibrated-flux enclosure")
    ax.set_title(r"(b) Top-$10\%$ error cells: same fields", loc="left", pad=9)
    ax.set_ylabel("Overlap / identified fraction (%)", labelpad=4)
    ax.set_ylim(0, 103); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.37), frameon=False,
              borderpad=0, labelspacing=.20, handlelength=2.2, fontsize=9)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, labels=[str(k) for k in budgets])
        ax.set_xlabel("Maximum iterations", labelpad=4)
        ax.tick_params(length=2.5, width=.5)
        ax.grid(axis="y", color="#D7DCD8", lw=.45)
        ax.set_axisbelow(True)
    return fig
