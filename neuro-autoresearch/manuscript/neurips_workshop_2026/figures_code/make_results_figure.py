"""Figure 3 of the AgenticLS paper: held-out neural predictivity across the visual hierarchy.

One panel per dataset. Within a panel, the x-axis walks the recorded regions from
early to late visual cortex and each model is a profile line. Only individuals that
were EXCLUDED from model selection are shown, so every point is a generalization
result. The discovered model is the emphasized series; pretrained references (and,
later, classical interpretable models) are context.

Reads ``runs/best/<dataset>/evaluation.json`` and writes
``neuro_autoresearch_latex/figures/fig3_results.{pdf,png,svg}``.

Run from the repository root with the venv active::

    python manuscript/neurips_workshop_2026/figures_code/make_results_figure.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "neuro_autoresearch_latex" / "figures"

# ---------------------------------------------------------------- content ---
# Panels: (dataset id, panel title, subtitle, regions early -> late, y-axis label)
PANELS = [
    ("nsd", "Human fMRI (NSD)", "held-out subjects 5 and 7",
     ["V1", "V2", "V3", "ventral"], "Noise-normalized explained variance"),
    ("tvsd", "Macaque spiking (TVSD)", "held-out monkey N",
     ["V1", "V4", "IT"], "Mean Pearson correlation"),
]
REGION_DISPLAY = {"ventral": "Ventral"}

# Series in drawing order. role: "champion" | "pretrained" | "interpretable".
# Add classical interpretable baselines here once they exist in evaluation.json,
# e.g. ("gabor_pyramid", "Gabor pyramid", "interpretable").
SERIES = [
    ("clip_pretrained", "CLIP", "pretrained"),
    ("dinov2_pretrained", "DINOv2", "pretrained"),
    ("champion", "Discovered\nmodel", "champion"),
]

# ------------------------------------------------------------------ style ---
INK = "#1F262E"
MUTED = "#6B7680"
GRID = "#E3E7EA"
ACCENT = "#D9741A"                     # the emphasized series
PRETRAINED = {"clip_pretrained": "#3D5468", "dinov2_pretrained": "#8E9CAA"}
INTERPRETABLE = ["#2A9D8F", "#7FC8BF", "#B9E0DA"]   # reserved, teal family
STYLE = {
    "champion": dict(color=ACCENT, lw=1.9, marker="o", ms=5.8, zorder=6),
    "pretrained": dict(lw=1.25, marker="o", ms=4.6, zorder=4),
    "interpretable": dict(lw=1.25, marker="s", ms=4.2, zorder=4, ls=(0, (4, 2))),
}


def _font():
    for name in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


def heldout_mean(model: dict, roi: str) -> float:
    """Equal-weight mean over held-out individuals that have this region."""
    subjects = model["roi_scores"]["heldout"]["subjects"]
    vals = [s[roi] for s in subjects.values() if roi in s]
    return float(np.mean(vals)) if vals else np.nan


def series_color(key: str, role: str, n_interp: list[int]) -> str:
    if role == "champion":
        return ACCENT
    if role == "pretrained":
        return PRETRAINED[key]
    c = INTERPRETABLE[n_interp[0] % len(INTERPRETABLE)]
    n_interp[0] += 1
    return c


def draw_panel(ax, dataset: str, title: str, subtitle: str, regions: list[str], ylabel: str, letter: str):
    ev = json.loads((ROOT / "runs" / "best" / dataset / "evaluation.json").read_text())
    models = ev["models"]
    x = np.arange(len(regions))
    n_interp = [0]
    labels = []  # (y_end, text, color)

    for key, label, role in SERIES:
        if key not in models:
            continue
        y = np.array([heldout_mean(models[key], r) for r in regions])
        color = series_color(key, role, n_interp)
        kw = dict(STYLE[role])
        kw.setdefault("color", color)
        ax.plot(x, y, markerfacecolor=kw["color"], markeredgecolor="white", markeredgewidth=1.1,
                solid_capstyle="round", **kw)
        labels.append((y[-1], label, kw["color"], role == "champion"))

    # direct labels at the right end, nudged apart if they collide
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    labels.sort(key=lambda t: t[0])
    placed = []
    for y_end, text, color, bold in labels:
        y_lab = y_end
        if placed and y_lab - placed[-1] < 0.075 * span:
            y_lab = placed[-1] + 0.075 * span
        placed.append(y_lab)
        ax.annotate(text, xy=(x[-1], y_end), xytext=(x[-1] + 0.14, y_lab),
                    textcoords="data", va="center", ha="left", fontsize=7.6,
                    color=color, fontweight="bold" if bold else "normal",
                    annotation_clip=False)

    # value labels on the emphasized series only, and only at the two ends of the hierarchy
    champ = np.array([heldout_mean(models["champion"], r) for r in regions])
    ax.annotate(f"{champ[0]:.2f}", (x[0], champ[0]), xytext=(0, 7), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.8, color=ACCENT, fontweight="bold")
    ax.annotate(f"{champ[-1]:.2f}", (x[-1], champ[-1]), xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=6.8, color=ACCENT, fontweight="bold")

    # faint band over the regions where the discovered model leads every plotted reference
    ref_max = np.nanmax([[heldout_mean(models[k], r) for r in regions]
                         for k, _, role in SERIES if role != "champion" and k in models], axis=0)
    leads = champ > ref_max
    if leads.any():
        last = int(np.max(np.where(leads)[0]))
        right = (x[last] + x[last + 1]) / 2 if last + 1 < len(x) else x[last] + 0.35
        ax.axvspan(-0.35, right, color=ACCENT, alpha=0.06, lw=0, zorder=0)
        ax.text(right - 0.06, 0.985, "model leads", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=6.4, color=ACCENT, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_DISPLAY.get(r, r) for r in regions], fontsize=8.4, color=INK)
    ax.set_xlim(-0.35, len(regions) - 1 + 0.35)
    # a thin "early -> late" arrow under the region ticks
    ax.annotate("", xy=(0.78, -0.16), xytext=(0.22, -0.16), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0),
                annotation_clip=False)
    ax.text(0.5, -0.2, "visual hierarchy", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=7.6, color=MUTED)
    ax.margins(y=0.18)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=7.2)
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=9.2, fontweight="bold", color=INK, pad=16)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=7.2, color=MUTED, va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stem", default="fig3_results")
    ap.add_argument("--fmt", nargs="+", default=("pdf", "png", "svg"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": _font(), "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.6, 2.5))
    for ax, (ds, title, sub, regions, ylabel), letter in zip(axes, PANELS, "ABCD"):
        draw_panel(ax, ds, title, sub, regions, ylabel, letter)
    fig.subplots_adjust(left=0.1, right=0.895, top=0.8, bottom=0.24, wspace=0.62)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in args.fmt:
        p = args.out / f"{args.stem}.{ext}"
        fig.savefig(p, dpi=300, facecolor="white")
        print(p)


if __name__ == "__main__":
    main()
