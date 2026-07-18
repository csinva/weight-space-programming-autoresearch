#!/usr/bin/env python3
"""Heatmap for a cross-dataset generalization matrix produced by ``cross_dataset.py``.

Reads the stored ``cross_vision.json`` (never re-scores anything) and renders the rows x cols
champion-by-dataset held-out matrix. Colour is per-column retention vs each column's native
champion (boxed diagonal = 1.0), so the comparison stays honest even though the metric differs by
column; each cell is annotated with the raw metric value and the retained fraction.

    python experiments/evaluation/cross_dataset_plot.py
    python experiments/evaluation/cross_dataset_plot.py --input runs/best/cross_vision/cross_vision.json
    python experiments/evaluation/cross_dataset_plot.py --output figure.pdf
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_JSON = os.path.join(REPO, "runs", "best", "cross_vision", "cross_vision.json")

INK, MUTED, FAINT, PAPER = "#22303A", "#6B7078", "#C9C1B4", "#FBF8F3"
SERIF, SANS = "DejaVu Serif", "DejaVu Sans"
LABELS = {"nsd": "NSD", "tvsd": "TVSD", "nnn": "Triple-N"}
METRIC = {"noise_normalized_r2": "r²/NC", "mean_pearson_r": "mean r"}
# A curated sand -> terracotta -> deep-brick clay ramp for the retention colour.
CLAY = ["#EFE4D6", "#E4BFA6", "#D59274", "#C0654A", "#9E3B28", "#79281B"]


def render_heatmap(document: dict, output: str) -> str:
    """Render the matrix to ``output`` (PNG) and the sibling ``.pdf``; returns the PNG path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patheffects import withSimplePatchShadow
    import numpy as np

    datasets = list(document["matrix"])
    n = len(datasets)
    raw = np.array([[document["matrix"][row][col]["mean"] for col in datasets] for row in datasets])
    retention = np.array([[document["matrix"][row][col]["retention"] for col in datasets]
                          for row in datasets], dtype=float)

    # Every cell sits in a narrow band near the native score, so a 0-anchored scale washes the
    # differences out. Stretch the colour range to just below the weakest transfer (annotations still
    # carry the exact numbers), so the honest gradient -- native (1.0) down to the worst transfer -- is
    # legible.
    cmap = LinearSegmentedColormap.from_list("clay", CLAY)
    vmin = max(0.0, float(np.nanmin(retention)) - 0.05)
    vmax = max(1.0, float(np.nanmax(retention)))
    norm = plt.Normalize(vmin, vmax)

    plt.rcParams.update({
        "font.family": SANS, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 100,
    })
    fig = plt.figure(figsize=(3.1 + 1.62 * n, 2.6 + 1.62 * n))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes((0.155, 0.115, 0.66, 0.66))
    ax.set_facecolor(PAPER)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    gap, radius = 0.075, 0.09
    shadow = withSimplePatchShadow(offset=(2.4, -2.8), alpha=0.20, shadow_rgbFace="#5E4636")
    for i in range(n):
        for j in range(n):
            frac = retention[i, j]
            diagonal = i == j
            fill = cmap(norm(frac))
            tile = FancyBboxPatch(
                (j - 0.5 + gap, i - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill, edgecolor="none", zorder=2)
            tile.set_path_effects([shadow])
            ax.add_patch(tile)
            if diagonal:                                    # a soft ivory inset marks the reference
                ax.add_patch(FancyBboxPatch(
                    (j - 0.5 + gap, i - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap,
                    boxstyle=f"round,pad=0,rounding_size={radius}",
                    facecolor="none", edgecolor=PAPER, linewidth=2.4, zorder=3))

            light = norm(frac) > 0.5
            value_color = PAPER if light else INK
            caption_color = "#F3E9DF" if light else MUTED
            ax.text(j, i - 0.11, f"{raw[i, j]:.3f}", ha="center", va="center", color=value_color,
                    fontsize=18, fontweight="bold", zorder=4)
            caption = "native reference" if diagonal else f"{frac * 100:.0f}% of native"
            ax.text(j, i + 0.235, caption, ha="center", va="center", color=caption_color,
                    fontsize=8.8, style="italic", zorder=4)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        [f"{LABELS.get(d, d)}\n{METRIC.get(document['datasets'][d]['metric'], '')}" for d in datasets],
        fontsize=11.5, color=INK)
    for tick, d in zip(ax.get_xticklabels(), datasets):     # metric line smaller + muted
        tick.set_linespacing(1.5)
    ax.set_yticklabels([LABELS.get(d, d) for d in datasets], fontsize=11.5, color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Axis-group headers (editorial small caps) + rotated row header.
    fig.text(0.485, 0.848, "E V A L U A T E D   O N   ·   H E L D - O U T   S U B J E C T S",
             ha="center", va="bottom", color=MUTED, fontsize=9.5, fontfamily=SANS)
    fig.text(0.046, 0.445, "C H A M P I O N   S E A R C H E D   O N", ha="center", va="center",
             rotation=90, color=MUTED, fontsize=9.5, fontfamily=SANS)

    # Title block (serif) with a thin editorial rule beneath it.
    fig.text(0.485, 0.955, "Vision champions transfer across datasets", ha="center", va="bottom",
             color=INK, fontsize=17.5, fontfamily=SERIF, fontweight="bold")
    fig.text(0.485, 0.915,
             "Best searched model of each dataset, scored on every dataset's held-out subjects",
             ha="center", va="bottom", color=MUTED, fontsize=10.5, fontfamily=SERIF, style="italic")
    fig.add_artist(plt.Line2D([0.155, 0.815], [0.902, 0.902], color=FAINT, linewidth=0.9,
                              transform=fig.transFigure))

    metric_note = "  ·  ".join(
        f"{LABELS.get(d, d)} {METRIC.get(document['datasets'][d]['metric'], '')}" for d in datasets)
    fig.text(0.485, 0.052,
             "Ivory-ringed diagonal is each dataset's native champion; colour = score retained vs "
             "that native champion.",
             ha="center", va="top", color=MUTED, fontsize=8.8, fontfamily=SERIF)
    fig.text(0.485, 0.03, f"Metrics differ per column ({metric_note}) — compare down columns, "
             "not across.",
             ha="center", va="top", color=MUTED, fontsize=8.8, fontfamily=SERIF, style="italic")

    # Slim, outline-free colourbar.
    cax = fig.add_axes((0.86, 0.27, 0.02, 0.36))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("retained vs native champion", color=MUTED, fontsize=8.8, labelpad=8)
    ticks = [t for t in (0.7, 0.8, 0.9, 1.0) if vmin <= t <= vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t * 100)}%" for t in ticks])
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=0, pad=4)
    cbar.outline.set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    saved = []
    for suffix in (".png", ".pdf"):
        target = os.path.splitext(output)[0] + suffix
        fig.savefig(target, facecolor=PAPER, dpi=300)
        saved.append(target)
    plt.close(fig)
    return saved[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_JSON, help="cross-dataset matrix JSON (from cross_dataset.py)")
    p.add_argument("--output", help="output .png path (a sibling .pdf is written too); "
                   "defaults next to the input JSON")
    a = p.parse_args(argv)

    if not os.path.exists(a.input):
        raise SystemExit(f"missing {a.input}; build it first with "
                         f"`python experiments/evaluation/cross_dataset.py`")
    with open(a.input) as f:
        document = json.load(f)
    output = a.output or os.path.splitext(a.input)[0] + ".png"
    figure = render_heatmap(document, output)
    print(f"saved figure -> {figure} (+ .pdf)")


if __name__ == "__main__":
    main()
