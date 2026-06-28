#!/usr/bin/env python3
"""Plot the AlexNet per-layer encoding scores produced by ``alexnet_baseline.py`` (per-layer mode).

Reads a ``results/alexnet_per-layer_*.json`` (or the latest if none is given) and renders the
AlexNet feature hierarchy as two panels -- the early-visual stream (V1-hV4) and the ventral stream
(FFA/PPA/VWFA) -- so the "early ROIs peak in conv, ventral ROIs peak in fc" story reads at a glance.
The search champion's per-stream scores are drawn as reference lines.

    python scripts/benchmark/plot_alexnet_layers.py
    python scripts/benchmark/plot_alexnet_layers.py results/alexnet_per-layer_nsd_subj1_XXXX.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Champion qq145 (NSD subj1, early+ventral; run nsd__2026-06-28) per-ROI NOISE-NORMALIZED test
# score (r^2/NC_r2, the search reward) -- the comparison point. Overall reward 0.4407. Untrained
# HMAX-style Gabor/C1 + random conv model; re-evaluated in-process (eval_champion.py), reproduces
# the DB row exactly.
CHAMPION = {"early": 0.54807, "ventral": 0.34611, "V1": 0.64456, "V2": 0.56438, "V3": 0.54084,
            "hV4": 0.48739, "FFA": 0.36687, "PPA": 0.45163, "VWFA": 0.26109}

# Palette: ivory paper, ink, two sequential stream ramps (light->deep = posterior->anterior ROI).
INK, PAPER, PANEL = "#2c2722", "#f4efe6", "#fbf8f2"
EARLY = ["V1", "V2", "V3", "hV4"]
VENTRAL = ["FFA", "PPA", "VWFA"]
COOL = ["#7fb3df", "#3d86c6", "#235e95", "#12435f"]   # V1..hV4 (light->deep posterior->anterior)
WARM = ["#e6a92e", "#dd6b1f", "#c0392b"]              # FFA, PPA, VWFA
EARLY_ANCHOR, VENT_ANCHOR = "#0d2c4a", "#6e1b12"     # stream means: darker than any sub-ROI


def _latest():
    files = sorted(glob.glob(os.path.join(RESULTS, "alexnet_per-layer_*.json")))
    if not files:
        sys.exit("no alexnet_per-layer_*.json in results/ -- run alexnet_baseline.py first")
    return files[-1]


def _panel(ax, x, layers, recs, rois, ramp, anchor, stream, title, ymax):
    def series(roi):
        return [r["roi_corr"].get(roi, {}).get("test", float("nan")) for r in recs]

    n = len(layers)
    # conv vs fc shading -- conv taps are the SRP'd "features" block, fc the "classifier" block.
    n_conv = sum(1 for l in layers if l.startswith("conv"))
    ax.axvspan(-0.5, n_conv - 0.5, color="#eae3d4", alpha=0.55, lw=0, zorder=0)
    ax.text((n_conv - 1) / 2, ymax * 0.985, "convolutional", ha="center", va="top",
            fontsize=8.5, color="#9a8f79", style="italic")
    ax.text((n_conv + n - 1) / 2, ymax * 0.985, "fully-connected", ha="center", va="top",
            fontsize=8.5, color="#9a8f79", style="italic")

    # Champion reference for this stream.
    cy = CHAMPION[stream]
    ax.axhline(cy, color=anchor, ls=(0, (5, 4)), lw=1.2, alpha=0.5, zorder=2)
    ax.text(n - 0.55, cy + ymax * 0.012, f"search champion  {cy:.3f}", color=anchor,
            fontsize=8.5, va="bottom", ha="right", alpha=0.85, fontstyle="italic")

    # Stream mean: filled anchor curve.
    sm = series(stream)
    ax.fill_between(x, sm, color=anchor, alpha=0.06, zorder=1)
    ax.plot(x, sm, color=anchor, lw=3.4, marker="o", ms=8, mfc="white", mec=anchor,
            mew=2.0, zorder=6, label=f"{stream} (mean)")

    # Sub-ROIs.
    for roi, c in zip(rois, ramp):
        s = series(roi)
        ax.plot(x, s, color=c, lw=2.0, marker="o", ms=4.5, alpha=0.95, zorder=4, label=roi)
        i = max(range(len(s)), key=lambda j: s[j])           # peak dot
        ax.scatter([i], [s[i]], s=70, color=c, edgecolor="white", lw=1.3, zorder=7)

    # Annotate the stream-mean peak.
    i = max(range(len(sm)), key=lambda j: sm[j])
    ax.annotate(f"peak · {layers[i]}\n{sm[i]:.3f}", (i, sm[i]), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=9.5, color=anchor, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=10.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", color="#ddd5c6", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdb3a0")
    ax.tick_params(colors="#6b6253")
    # Legends sit in each panel's empty corner: early lines occupy the top, ventral the bottom-right.
    loc = "upper left" if stream == "ventral" else "lower right"
    ax.legend(fontsize=9.5, frameon=False, loc=loc, labelcolor=INK, handlelength=1.6,
              borderaxespad=0.8)


def main(argv=None):
    path = (argv or sys.argv[1:])
    path = path[0] if path else _latest()
    if not os.path.isabs(path):
        path = os.path.join(RESULTS, os.path.basename(path))
    data = json.load(open(path))
    recs = data["records"]
    layers = [r["label"] for r in recs]
    x = list(range(len(layers)))

    allv = [v.get("test", 0) for r in recs for v in r["roi_corr"].values()]
    ymax = (max(allv + list(CHAMPION.values())) // 0.05 + 1) * 0.05

    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": PAPER,
                         "axes.facecolor": PANEL, "savefig.facecolor": PAPER})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.6), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.10, wspace=0.07)

    _panel(axL, x, layers, recs, EARLY, COOL, EARLY_ANCHOR, "early",
           "Early-visual stream", ymax)
    _panel(axR, x, layers, recs, VENTRAL, WARM, VENT_ANCHOR, "ventral",
           "Ventral stream", ymax)
    axL.set_ylabel("noise-normalized encoding score   (r² / NC_r²)", fontsize=12, color=INK)
    fig.supxlabel("AlexNet layer  (ImageNet-pretrained, frozen — never trained)", fontsize=12,
                  color=INK, y=0.015)

    fig.text(0.07, 0.935, "The AlexNet feature hierarchy, read out into the brain",
             fontsize=18, fontweight="bold", color=INK, ha="left")
    sub = (f"NSD subj{data.get('subject','?')}  ·  per-layer ridge encoding, noise-normalized r²/NC²  ·  "
           f"conv taps SRP→{data.get('srp_dim','?')} (visreps), fc taps raw  ·  fast ridge  ·  "
           "early ROIs peak in conv, ventral ROIs peak in fc")
    fig.text(0.07, 0.885, sub, fontsize=10.5, color="#6b6253", ha="left")

    out = path.replace(".json", ".png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
