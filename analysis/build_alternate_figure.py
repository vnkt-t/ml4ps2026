"""Add vertical breathing room to the alternate paper's existing field figure.

Reuse the verified field selection and numerical construction. Only the physical
positions of the workflow and lower panels change; maps and labels keep their
original sizes. The primary figures are not overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import build_submission_assets as builder

GAP_INCHES = 0.26
NAME = "submission_maps_alternate"
MANIFEST = builder.OUT / "alternate_figure_manifest.json"
INPUTS = builder.FIGURE_INPUTS + [Path(__file__)]
OUTPUTS = [builder.FIG / f"{NAME}.{ext}" for ext in ["pdf", "png", "svg"]]


def add_workflow_gap(fig):
    width, height = fig.get_size_inches()
    positions = [ax.get_position(original=True).bounds for ax in fig.axes]
    scale = height / (height + GAP_INCHES)
    fig.set_size_inches(width, height + GAP_INCHES)
    for index, (ax, (x, y, w, h)) in enumerate(zip(fig.axes, positions)):
        lift = GAP_INCHES / (height + GAP_INCHES) if index == 0 else 0
        ax.set_position([x, y * scale + lift, w, h * scale])
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position((x, y * scale))
    for swatch in fig.artists:
        vertices = swatch.get_xy().copy()
        vertices[:, 1] *= scale
        swatch.set_xy(vertices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        saved = json.loads(MANIFEST.read_text())
        if saved["inputs"] != builder.hashes(INPUTS) or saved["outputs"] != builder.hashes(OUTPUTS):
            raise SystemExit("STALE ALTERNATE FIGURE: rebuild the layout")
        if saved["workflow_gap_inches"] != GAP_INCHES:
            raise SystemExit("STALE ALTERNATE GAP")
        print("PASS: alternate figure sources and output hashes")
        return

    original_save = builder.save_figure

    def save_alternate(fig, name):
        if name == "submission_maps":
            add_workflow_gap(fig)
            original_save(fig, NAME)
        else:
            builder.plt.close(fig)

    with patch.object(builder, "save_figure", save_alternate):
        representative = builder.figures(*builder.load_all())
    expected = json.loads((builder.OUT / "representative_field.json").read_text())
    if representative != expected:
        raise RuntimeError("Alternate layout changed the representative field")
    MANIFEST.write_text(json.dumps({
        "inputs": builder.hashes(INPUTS), "outputs": builder.hashes(OUTPUTS),
        "workflow_gap_inches": GAP_INCHES,
        "representative_field": representative,
        "scope": "Same numerical construction and physical panel/text sizes; extra vertical space below workflow only.",
    }, indent=2, sort_keys=True) + "\n")
    print("Built alternate field figure with additional workflow spacing")


if __name__ == "__main__":
    main()
