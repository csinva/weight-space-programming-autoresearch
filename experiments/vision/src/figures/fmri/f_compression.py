"""Render the natural-language compression (text-bottleneck) figure for the LAION champion (erjpu).

Reads existing compression artifacts in artifacts/final/compression/erjpu__2026-11-07/
(and starter__2026-11-07/ for the search-start reference) without running model evaluation
or showing the Gabor baseline. Writes vector PDF/SVG and 450 dpi PNG to artifacts/figures/fmri/.

Usage: python src/figures/fmri/f_compression.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
COMP_DIR = REPO / "artifacts/final/compression/erjpu__2026-11-07"
START_DIR = REPO / "artifacts/final/compression/starter__2026-11-07"
OUT_DIR = REPO / "artifacts/figures/fmri"
MM = 1 / 25.4

STYLE = {
    "width": 180 * MM,
    "height": 68 * MM,
    "font": 6.5,
    "small": 5.8,
    "letter": 8,
    "grey": "#727272",
    "light": "#c4c4c4",
    "dark": "#252525",
    "highlight": "#D55E00",
    "rebuild": "#0072B2",
}

STREAMS = [
    ("pooled", "Pooled"),
    ("early visual stream", "Early"),
    ("ventral visual stream", "Ventral"),
]
ROIS = ["V1", "V2", "V3", "hV4", "FFA", "PPA"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": STYLE["font"],
    "axes.labelsize": STYLE["font"],
    "axes.titlesize": STYLE["font"],
    "xtick.labelsize": STYLE["font"],
    "ytick.labelsize": STYLE["font"],
    "legend.fontsize": STYLE["font"],
    "text.color": STYLE["dark"],
    "axes.labelcolor": STYLE["dark"],
    "xtick.color": STYLE["dark"],
    "ytick.color": STYLE["dark"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 0.45,
    "xtick.major.width": 0.45,
    "ytick.major.width": 0.45,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.labelpad": 3,
    "axes.titlepad": 3,
    "lines.linewidth": 1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 450,
})


def axes(fig, slot, rect, **kwargs):
    sx, sy, _, _ = slot
    x, y, w, h = rect
    fw, fh = fig.get_size_inches() / MM
    return fig.add_axes([(sx + x) / fw, (sy + y) / fh, w / fw, h / fh], **kwargs)


def text(fig, slot, x, y, label, **kwargs):
    fw, fh = fig.get_size_inches() / MM
    return fig.text(
        (slot[0] + x) / fw,
        (slot[1] + y) / fh,
        label,
        ha=kwargs.pop("ha", "left"),
        va=kwargs.pop("va", "bottom"),
        **kwargs,
    )


def heading(fig, slot, key, title):
    text(fig, slot, 0, slot[3] - 4.0, key.lower(), fontsize=STYLE["letter"], fontweight="bold")
    text(fig, slot, 4.5, slot[3] - 3.9, title, fontsize=7)


def get_score(row, key):
    if key == "pooled":
        return float(row["test_corr"])
    return float(row["areas"][key])


def panel_a_schematic(fig, slot, scores, runs, description):
    """Left panel: the 200-word text bottleneck and feature similarity (CKA)."""
    ax = axes(fig, slot, (0, 2, slot[2], slot[3] - 9))
    ax.set_xlim(0, slot[2])
    ax.set_ylim(0, slot[3] - 9)
    ax.axis("off")

    rebuilds = [k for k in sorted(scores) if k.startswith("rebuild_")]
    ckas = [runs["cka"][k] for k in rebuilds]
    pooled_rebuilds = [scores[k]["test_corr"] for k in rebuilds]
    orig_pooled = scores["original"]["test_corr"]
    word_count = len(description.split())

    def card(x, y, w, h, title, body, face, edge, title_color=STYLE["dark"]):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.6",
                facecolor=face,
                edgecolor=edge,
                lw=0.6,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 3.2,
            title,
            ha="center",
            va="top",
            fontsize=STYLE["font"],
            fontweight="bold",
            color=title_color,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + (h - 4.2) / 2,
            body,
            ha="center",
            va="center",
            fontsize=5.2,
            color=STYLE["dark"],
            linespacing=1.28,
            zorder=3,
        )

    y0, h0 = 13.0, 34.5
    card(
        0.5,
        y0,
        18.0,
        h0,
        "Searched model",
        f"LAION champion\n865 lines of code\n9,996 features\nPooled r = {orig_pooled:.3f}",
        "#fcf1eb",
        "#e6b89c",
        title_color=STYLE["highlight"],
    )

    excerpt = (
        "“Compressive gamma, opponent\n"
        "luminance, red-green, blue-yellow...\n"
        "3 octaves of 7×7 quadrature\n"
        "Gabors + DoGs, divisive norm...\n"
        "orientation harmonics, curvature,\n"
        "phase congruency, channel\n"
        "correlations, 5-octave geometry...”"
    )
    card(
        28.0,
        y0,
        31.5,
        h0,
        f"Text bottleneck ({word_count} words)",
        excerpt,
        "#f7f7f7",
        "#cccccc",
    )

    card(
        69.0,
        y0,
        18.5,
        h0,
        "Rebuilt models",
        f"4 fresh builders\n4/4 pass on try 1\nCKA = {min(ckas):.2f}–{max(ckas):.2f}\n"
        f"Pooled r = {np.mean(pooled_rebuilds):.3f}",
        "#eef5fa",
        "#9ec4df",
        title_color=STYLE["rebuild"],
    )

    for x_start, x_end, top_label, bot_label in [
        (19.3, 27.2, "Describe", "≤200 words\nno code"),
        (60.3, 68.2, "Rebuild", "text +\ncontract"),
    ]:
        y_mid = y0 + h0 / 2
        ax.add_patch(
            FancyArrowPatch(
                (x_start, y_mid),
                (x_end, y_mid),
                arrowstyle="-|>",
                mutation_scale=5.5,
                color=STYLE["grey"],
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
            )
        )
        ax.text(
            (x_start + x_end) / 2,
            y_mid + 2.0,
            top_label,
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
            color=STYLE["dark"],
        )
        ax.text(
            (x_start + x_end) / 2,
            y_mid - 2.0,
            bot_label,
            ha="center",
            va="top",
            fontsize=5.0,
            color=STYLE["grey"],
            linespacing=1.15,
        )

    retention_pct = 100 * np.mean(pooled_rebuilds) / orig_pooled
    ax.text(
        slot[2] / 2,
        4.2,
        f"Rebuilds retain {retention_pct:.0f}% of the searched model's pooled r\n"
        f"and {np.mean(ckas):.2f} mean linear CKA across 1,000 test images",
        ha="center",
        va="center",
        fontsize=STYLE["small"],
        color=STYLE["grey"],
        linespacing=1.25,
    )


def panel_b_encoding(fig, slot, scores, start_scores):
    """Right panel: encoding r across streams and visual areas (without Gabor baseline)."""
    rebuilds = [k for k in sorted(scores) if k.startswith("rebuild_")]

    # Left sub-axes: Pooled, Early, Ventral streams
    ax_s = axes(fig, slot, (10, 12, 25, 31))
    # Right sub-axes: Individual visual areas V1, V2, V3, hV4, FFA, PPA
    ax_r = axes(fig, slot, (40, 12, 42, 31))

    stream_xs = np.arange(len(STREAMS))
    orig_s = [get_score(scores["original"], k) for k, _ in STREAMS]
    start_s = [get_score(start_scores["original"], k) for k, _ in STREAMS]
    reb_s = np.array([[get_score(scores[r], k) for k, _ in STREAMS] for r in rebuilds])
    reb_s_mean = reb_s.mean(axis=0)

    w = 0.30
    ax_s.bar(stream_xs - w / 2, orig_s, width=w, color=STYLE["highlight"], alpha=0.90, zorder=2)
    ax_s.bar(stream_xs + w / 2, reb_s_mean, width=w, color=STYLE["rebuild"], alpha=0.84, zorder=2)
    offsets = np.linspace(-0.07, 0.07, len(rebuilds))
    for i, r_vals in enumerate(reb_s):
        ax_s.scatter(
            stream_xs + w / 2 + offsets[i],
            r_vals,
            s=8,
            facecolor="white",
            edgecolor=STYLE["dark"],
            lw=0.55,
            zorder=4,
        )
    for idx, s_val in enumerate(start_s):
        ax_s.plot(
            [idx - 0.36, idx + 0.36],
            [s_val, s_val],
            color="#555555",
            lw=0.75,
            ls=(0, (2, 1.5)),
            zorder=3,
        )

    for idx, (o_val, r_val) in enumerate(zip(orig_s, reb_s_mean)):
        pct = 100 * r_val / o_val
        ax_s.text(
            idx,
            max(o_val, r_val) + 0.024,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=STYLE["small"],
            color=STYLE["dark"],
        )

    ax_s.set_xticks(stream_xs, [label for _, label in STREAMS])
    ax_s.set_ylim(0, 0.68)
    ax_s.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_s.set_ylabel("Encoding $r$ (LAION, 5 subjects)")
    ax_s.set_title("Streams", fontsize=STYLE["font"], pad=4)
    ax_s.spines["bottom"].set_bounds(-0.45, len(STREAMS) - 0.55)

    # Right sub-axes: V1-hV4 and FFA, PPA
    positions = {"V1": 0, "V2": 1, "V3": 2, "hV4": 3, "FFA": 4.35, "PPA": 5.35}
    roi_xs = np.array([positions[r] for r in ROIS])
    orig_r = np.array([get_score(scores["original"], r) for r in ROIS])
    start_r = np.array([get_score(start_scores["original"], r) for r in ROIS])
    reb_r = np.array([[get_score(scores[rb], r) for r in ROIS] for rb in rebuilds])
    reb_r_mean = reb_r.mean(axis=0)

    early_mask = np.array([r in ("V1", "V2", "V3", "hV4") for r in ROIS])

    for i in range(len(rebuilds)):
        ax_r.plot(
            roi_xs[early_mask],
            reb_r[i, early_mask],
            color="#9ec4df",
            lw=0.6,
            alpha=0.8,
            zorder=2,
        )
        ax_r.scatter(
            roi_xs + offsets[i],
            reb_r[i],
            s=8,
            color="#9ec4df",
            edgecolor=STYLE["rebuild"],
            lw=0.4,
            zorder=3,
        )

    ax_r.plot(roi_xs[early_mask], orig_r[early_mask], color=STYLE["highlight"], lw=1.25, zorder=4)
    ax_r.scatter(roi_xs, orig_r, s=16, color=STYLE["highlight"], lw=0, zorder=5)

    ax_r.plot(roi_xs[early_mask], reb_r_mean[early_mask], color=STYLE["rebuild"], lw=1.25, zorder=4)
    ax_r.scatter(roi_xs, reb_r_mean, s=16, marker="D", color=STYLE["rebuild"], lw=0, zorder=5)

    ax_r.plot(
        roi_xs[early_mask],
        start_r[early_mask],
        color=STYLE["grey"],
        lw=0.8,
        ls=(0, (2, 1.5)),
        zorder=1,
    )
    ax_r.scatter(roi_xs, start_r, s=9, marker="_", color=STYLE["grey"], lw=0.9, zorder=2)

    ax_r.set_xticks(roi_xs, ROIS)
    ax_r.set_xlim(-0.4, 5.75)
    ax_r.set_ylim(0, 0.68)
    ax_r.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_r.tick_params(axis="y", labelleft=False)
    ax_r.set_title("Visual areas", fontsize=STYLE["font"], pad=4)
    ax_r.spines["bottom"].set_bounds(roi_xs.min(), roi_xs.max())

    handles = [
        Line2D([], [], color=STYLE["highlight"], marker="o", lw=1.2, markersize=3.5),
        Line2D([], [], color=STYLE["rebuild"], marker="D", lw=1.2, markersize=3.2),
        Line2D(
            [],
            [],
            color="#9ec4df",
            marker="o",
            markeredgecolor=STYLE["rebuild"],
            markeredgewidth=0.4,
            lw=0,
            markersize=3.0,
        ),
        Line2D([], [], color=STYLE["grey"], ls=(0, (2, 1.5)), lw=0.8),
    ]
    labels = [
        "Searched model",
        "Rebuilt mean (200 words)",
        "Individual rebuilds ($n$ = 4)",
        "Search start",
    ]
    fw, fh = fig.get_size_inches() / MM
    fig.legend(
        handles,
        labels,
        loc="lower left",
        frameon=False,
        ncol=2,
        bbox_to_anchor=((slot[0] + 10) / fw, (slot[1] + 48.0) / fh),
        borderaxespad=0,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.2,
        labelspacing=0.35,
        fontsize=STYLE["small"],
    )
    text(
        fig,
        slot,
        46,
        1.5,
        "Percentages above bars show rebuilt / searched encoding r",
        ha="center",
        fontsize=STYLE["small"],
        color=STYLE["grey"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    scores = json.loads((COMP_DIR / "laion__visual__encoding.json").read_text())
    runs = json.loads((COMP_DIR / "runs.json").read_text())
    description = (COMP_DIR / "description_200w.txt").read_text().strip()
    start_scores = json.loads((START_DIR / "laion__visual__encoding.json").read_text())

    fig = plt.figure(figsize=(STYLE["width"], STYLE["height"]))
    slot_a = (4, 3, 88, 62)
    slot_b = (95, 3, 83, 62)

    heading(fig, slot_a, "a", "Rebuilding the searched model from a 200-word description")
    panel_a_schematic(fig, slot_a, scores, runs, description)

    heading(fig, slot_b, "b", "Encoding performance of original and rebuilt models")
    panel_b_encoding(fig, slot_b, scores, start_scores)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.out / f"fmri_compression.{ext}", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out / 'fmri_compression.pdf'}")


if __name__ == "__main__":
    main()
