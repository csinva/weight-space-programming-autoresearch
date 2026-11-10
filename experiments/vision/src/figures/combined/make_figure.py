"""Combined figure: search, transfer, computations and area drivers of the searched LAION model (erjpu).

a  Search trajectory: early- and ventral-stream encoding r of the running pooled-best candidate
   (artifacts/figures/fmri/data/b.json), over gray dots for every scored candidate. Selection uses the pooled
   score, so a stream's line can fall.
b  Transfer to NSD: early- and ventral-stream encoding r of the searched model, seven interpretable baselines
   (the pixel baseline of c.json is omitted)
   and ImageNet-trained AlexNet (c.json). AlexNet uses the layer with the best LAION score for each stream,
   fixed before any NSD score: conv3 for the early stream and fc6 for the ventral stream. Points are
   subject means; bars are within-subject 95% confidence intervals. Rows are ordered by the mean of the two streams.
c  The model as a pipeline: one map per feature family on a LAION image (src/figures/combined/c_maps.py; global
   layout has no map and is drawn as a schematic).
d  Share of the gain beyond edges per family and area (d.json).
e  For hV4, FFA and PPA: the four shared images with the highest predicted area response, an image synthesized
   to maximize the area readout (src/figures/circuit/c_drivers.py) and, in pink, the largest texture operation of the area (e.json).

Only stored panel data are read. Writes artifacts/figures/combined/combined_figure.{pdf,svg,png}.
Usage: python src/figures/combined/make_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FMRI = REPO / "artifacts/figures/fmri/data"
CIRCUIT = REPO / "artifacts/figures/circuit/data"
OUT = REPO / "artifacts/figures/combined"

# Paper palette (neuro_autoresearch_latex/figures_code/palette.py). Blue marks the searched model and the
# corners-and-contours family, pink the texture-statistics family, as in the fMRI figure.
PRIMARY, SECONDARY, NEUTRAL = "#1F77B4", "#D6487E", "#6A747C"
INK, MUTED, DOT, AXIS, HAIR, TINT = "#202A32", "#64717B", "#C4CDD3", "#9AA5AD", "#DCE2E6", "#E3EFF8"
W, H = 7.5, 6.5
# The paper prints this 7.5 in figure at its 6.5 in text width (te_preprint.sty), so type prints at 0.87 of
# these sizes: 6.9, 6.2 and 8.7 pt.
BODY, SMALL, TITLE = 8, 7.2, 10
EARLY, VENTRAL, POOLED = "early visual stream", "ventral visual stream", "early and ventral visual streams"

# Panel c columns: map key (None: schematic), header, family color, short description, family key.
MAPS = [("edges", "Edges", INK, "oriented\ncontrast", "E"),
        ("color", "Color", "#A9B3BA", "color\ncontrast", "K"),
        ("curvature", "Luminance\ncurvature", NEUTRAL, "bumps and\nsaddles", "S"),
        ("corners", "Corners and\ncontours", PRIMARY, "meeting or\nbending lines", "C"),
        ("texture", "Texture\nstatistics", SECONDARY, "texture\nchanges", "T"),
        (None, "Global\nlayout", "#B7C0C7", "symmetry,\ncenter/edge", "G")]
# Panel d: family key -> (label, color, marker, line style, emphasis)
FAMILIES = {"C": ("Corners and contours", PRIMARY, "o", "-", True),
            "T": ("Texture statistics", SECONDARY, "D", "-", True),
            "K": ("Color", "#A9B3BA", "o", (0, (1, 1.5)), False),
            "S": ("Luminance curvature", NEUTRAL, "s", (0, (4, 1.5)), False),
            "G": ("Global layout", "#B7C0C7", "^", (0, (5, 1, 1, 1)), False)}
OPERATIONS = {"M": "texture boundaries", "V": "orientation/color co-occurrence", "O": "cross-scale structure",
              "R": "geometry co-occurrence", "J": "joint feature co-occurrence"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": BODY, "axes.labelsize": BODY, "xtick.labelsize": BODY, "ytick.labelsize": BODY,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def axes(fig, x, y, w, h):
    """Axes placed in inches from the bottom-left corner of the figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def text(fig, x, y, s, **kw):
    kw.setdefault("fontsize", BODY)
    kw.setdefault("va", "bottom")
    return fig.text(x / W, y / H, s, **kw)


def heading(fig, x, y, letter, title):
    text(fig, x, y, letter, fontsize=TITLE, fontweight="bold")
    text(fig, x + 0.19, y, title, fontsize=TITLE, fontweight="bold")


def tile(fig, x, y, s, image, **kw):
    ax = axes(fig, x, y, s, s)
    ax.imshow(image, interpolation="lanczos", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def panel_a(fig, x, y, w, h, b):
    best = b["running_best"]
    idx = np.array([r["index"] for r in best])
    ax = axes(fig, x, y, w, h)
    lo, hidden = 0.3, 0
    for key, label in (("early", "Early"), ("ventral", "Ventral")):
        scores = np.array([c[key] for c in b["candidates"]])
        hidden += int((scores < lo).sum())
        ax.scatter([c["index"] for c in b["candidates"]], scores, s=2.2, color=DOT, lw=0, zorder=1)
        values = np.array([r[key] for r in best])
        ax.step(idx, values, where="post", color=PRIMARY, lw=1.5)
        ax.scatter(idx[-1], values[-1], s=14, color=PRIMARY, lw=0, zorder=3, clip_on=False)
        ax.text(idx[-1] + 14, values[-1], f"{label}\n{values[-1]:.2f}", color=PRIMARY, fontweight="bold",
                va="center", ha="left", linespacing=1.1)
    ax.set_ylim(lo, 0.62)
    ax.set_xlim(0, 600)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Performance (correlation)")
    print(f"panel a: {hidden} of {2 * len(b['candidates'])} stream scores fall below {lo}")


def within_subject_ci(values):
    """Half-width of the 95% confidence interval of each row of values (models x subjects) after removing
    subject offsets (Cousineau 2005, with the Morey 2008 correction), for comparing models within subjects."""
    from scipy import stats
    m, n = values.shape
    centred = values - values.mean(0, keepdims=True) + values.mean()
    sem = centred.std(1, ddof=1) / np.sqrt(n) * np.sqrt(m / (m - 1))
    return sem * stats.t.ppf(0.975, n - 1)


def panel_b(fig, x, y, w, h, c):
    models, subjects = c["models"], c["subjects"]
    # Each stream has its own x-range, fitted to its scores, so differences between models stay visible.
    columns = ((EARLY, "Early", "alexnet_conv3", "conv3", (0.17, 0.52), (0.2, 0.3, 0.4, 0.5)),
               (VENTRAL, "Ventral", "alexnet_fc6", "fc6", (0.18, 0.46), (0.2, 0.3, 0.4)))
    rows = [k for k in models if models[k]["kind"] != "reference" and k != "pixels"]  # pixels are not shown
    rows.sort(key=lambda k: np.mean([models[k]["mean"][EARLY], models[k]["mean"][VENTRAL]]), reverse=True)
    pos = {k: len(rows) - i for i, k in enumerate(rows)}
    ref_y = -0.45
    label_w, gap = 1.72, 0.22
    col = (w - label_w - gap) / 2
    for g, (region, name, ref, layer, xlim, xticks) in enumerate(columns):
        ax = axes(fig, x + label_w + g * (col + gap), y, col, h)
        keys = rows + [ref]
        values = np.array([[models[k]["scores"][s][region] for s in subjects] for k in keys])
        mean, sem = values.mean(1), within_subject_ci(values)
        best = mean[keys.index("erjpu")]
        ax.axvline(best, color=PRIMARY, lw=0.7, alpha=0.45, zorder=1)
        for k, mu, se in zip(keys, mean, sem):
            kind = models[k]["kind"]
            yk = ref_y if kind == "reference" else pos[k]
            color = {"searched": PRIMARY, "baseline": INK, "reference": MUTED}[kind]
            ax.errorbar(mu, yk, xerr=se, color=color, lw=1.2, capsize=1.8, capthick=1.0, zorder=3)
            ax.scatter(mu, yk, s=20 if kind == "searched" else 11, zorder=4, lw=0.9, edgecolor=color,
                       color="white" if kind == "reference" else color)
            if kind == "reference":
                ax.text(mu + se + 0.012, yk, layer, va="center", ha="left", fontsize=SMALL, color=MUTED)
        ax.axhspan(pos["erjpu"] - 0.45, pos["erjpu"] + 0.45, color=TINT, lw=0, zorder=0)
        ax.axhline(0.35, lw=0.6, color=HAIR)
        ax.set_title(name, fontsize=BODY, pad=3)
        ax.set_ylim(ref_y - 0.6, len(rows) + 0.6)
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4, labelleft=g == 0)
        ax.set_yticks([pos[k] for k in rows] + [ref_y],
                      ["iWSP model" if k == "erjpu" else models[k]["label"] for k in rows] + ["AlexNet (black-box)"])
        if g == 0:
            labels = ax.get_yticklabels()
            labels[rows.index("erjpu")].set_color(PRIMARY)
            labels[rows.index("erjpu")].set_fontweight("bold")
            labels[-1].set_color(MUTED)
    cx = x + label_w + col + gap / 2
    text(fig, cx, y - 0.38, "Performance (correlation)", ha="center")


def arrow(fig, x0, y0, x1, y1):
    fig.add_artist(FancyArrowPatch((x0 / W, y0 / H), (x1 / W, y1 / H), transform=fig.transFigure, color=AXIS,
                                   lw=0.8, arrowstyle="-|>", mutation_scale=6, shrinkA=0, shrinkB=0))


def bracket(fig, x0, x1, y, label, color=MUTED):
    """Horizontal bracket opening upward, with a centred label below."""
    xs, ys = [x0, x0, x1, x1], [y + 0.05, y, y, y + 0.05]
    fig.add_artist(Line2D([v / W for v in xs], [v / H for v in ys], color=AXIS, lw=0.7))
    text(fig, (x0 + x1) / 2, y - 0.04, label, ha="center", va="top", fontsize=SMALL, color=color)


def layout_icon(fig, x, y, s):
    """Schematic for the family without a map: mirror symmetry and centre versus periphery."""
    ax = axes(fig, x, y, s, s)
    ax.set_facecolor("#EEF1F3")
    ax.add_patch(plt.Circle((0.5, 0.5), 0.2, fc="none", ec=NEUTRAL, lw=0.8))
    ax.add_patch(plt.Circle((0.5, 0.5), 0.38, fc="none", ec=NEUTRAL, lw=0.8, ls=(0, (2, 1.5))))
    ax.plot([0.5, 0.5], [0.04, 0.96], color=NEUTRAL, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_c(fig, x, top, C, x_end):
    """Pipeline: image -> one map per feature family -> linear readout (the arrow ends at panel d)."""
    image_s, s, gap, lead = 0.74, 0.585, 0.05, 0.2
    row = 1                                                      # the dollhouse image of c.npz
    y0 = top - 0.48 - s
    xc = x + image_s + lead
    mid = y0 + s / 2
    tile(fig, x, mid - image_s / 2, image_s, C["images"][row])
    arrow(fig, x + image_s + 0.04, mid, xc - 0.04, mid)
    for j, (key, label, color, desc, family) in enumerate(MAPS):
        cx = xc + j * (s + gap)
        text(fig, cx + s / 2, y0 + s + 0.09, label, ha="center", linespacing=1.05, fontsize=SMALL + 0.3,
             color=color if family in "CT" else INK, fontweight="bold" if family in "CT" else "normal")
        fig.add_artist(Line2D([cx / W, (cx + s) / W], [(y0 + s + 0.04) / H] * 2, color=color, lw=2,
                              solid_capstyle="butt"))
        if key is None:
            layout_icon(fig, cx, y0, s)
        else:
            tile(fig, cx, y0, s, C[key][row], cmap="gray", vmin=0, vmax=np.percentile(C[key][row], 99.5))
        text(fig, cx + s / 2, y0 - 0.05, desc, ha="center", va="top", fontsize=SMALL, color=MUTED, linespacing=1.1)
    yb = y0 - 0.44
    bracket(fig, xc, xc + s, yb, "Baseline")
    last = xc + len(MAPS) * (s + gap) - gap
    bracket(fig, xc + s + gap, last, yb, "Gain beyond edges (d)")
    arrow(fig, last + 0.08, mid, x_end, mid)
    text(fig, (last + 0.08 + x_end) / 2, mid + 0.06, "Linear\nreadout", ha="center", fontsize=SMALL, color=MUTED,
         linespacing=1.05)
    return yb - 0.16


def panel_d(fig, x, y, w, h, d):
    rois = d["rois"]
    xs = np.array([0, 1, 2, 3, 4.5, 5.6])
    ax = axes(fig, x, y, w, h)
    ax.axhline(0, color=HAIR, lw=0.6, zorder=0)
    for key in ("K", "S", "G", "C", "T"):
        label, color, marker, style, strong = FAMILIES[key]
        v = np.array([d["share_beyond_edges"][key][r] for r in rois]) * 100
        ax.plot(xs[:4], v[:4], color=color, lw=1.6 if strong else 0.9, ls=style, zorder=3 if strong else 2)
        ax.scatter(xs, v, s=20 if strong else 11, color=color, marker=marker, lw=0, zorder=4 if strong else 2)
    ax.text(-0.1, 56.5, "Corners and contours", color=PRIMARY, fontweight="bold", va="bottom")
    ax.text(2.8, 30.5, "Texture statistics", color=SECONDARY, fontweight="bold", va="bottom", ha="right")
    ax.set_xticks(xs, rois)
    ax.set_xlim(-0.3, 5.9)
    ax.set_ylim(-4, 64)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Share of gain beyond edges (%)")
    ax.spines["bottom"].set_bounds(0, 3)
    ax.plot([4.5, 5.6], [-4, -4], color=AXIS, lw=0.7, clip_on=False, transform=ax.transData)
    handles = [Line2D([], [], color=FAMILIES[k][1], marker=FAMILIES[k][2], ls=FAMILIES[k][3], lw=0.9, ms=3.2, mew=0)
               for k in "KSG"]
    fig.legend(handles, [FAMILIES[k][0] for k in "KSG"], loc="upper left", frameon=False, ncol=2,
               bbox_to_anchor=((x - 0.12) / W, (y - 0.24) / H), borderaxespad=0, handlelength=1.8,
               handletextpad=0.4, columnspacing=0.9, labelspacing=0.25, fontsize=SMALL, labelcolor=MUTED)


def panel_e(fig, x, top, w, C, e):
    """Per area: a 2 x 2 grid of the highest-predicted shared images and a larger synthesized image."""
    areas, between, gap, lead = [str(a) for a in C["areas"]], 0.34, 0.04, 0.2
    block = (w - 2 * between) / 3
    big = (block - lead - gap) / 2
    s = (big - gap) / 2
    y = top - 0.5 - big
    for i, area in enumerate(areas):
        x0 = x + i * (block + between)
        shares = {k: e["share_beyond_edges"][k][area] for k in OPERATIONS}
        op = max(shares, key=shares.get)
        text(fig, x0, top - 0.22, area, fontweight="bold", fontsize=BODY + 1)
        text(fig, x0 + block, top - 0.22, OPERATIONS[op].capitalize(), color=SECONDARY, fontsize=SMALL, ha="right")
        if i == 0:
            text(fig, x0, y + big + 0.04, "Top predicted", fontsize=SMALL, color=MUTED)
            text(fig, x0 + block - big, y + big + 0.04, "Synthesized", fontsize=SMALL, color=MUTED)
        for j in range(4):
            tile(fig, x0 + (j % 2) * (s + gap), y + (1 - j // 2) * (s + gap), s, C["top"][i][j])
        tile(fig, x0 + block - big, y, big, C["optimized"][i])
    return y


def main():
    b, c, d, e = (json.loads((FMRI / f"{k}.json").read_text()) for k in "bcde")
    maps, drivers = np.load(OUT / "data/c.npz"), np.load(CIRCUIT / "c.npz")
    fig = plt.figure(figsize=(W, H))

    top = H - 0.2
    heading(fig, 0, top, "a", "")  # no subtitle for panel a
    panel_a(fig, 0.46, top - 1.66, 1.9, 1.42, b)
    heading(fig, 3.0, top, "b", "Transfer to NSD")
    panel_b(fig, 3.0, top - 1.66, 4.45, 1.42, c)

    top -= 2.26
    heading(fig, 0, top, "c", "The model computes six families of features")
    low = panel_c(fig, 0, top, maps, 5.02)
    heading(fig, 5.05, top, "d", "Families by area")
    panel_d(fig, 5.52, top - 1.6, 1.96, 1.42, d)
    low = min(low, top - 2.12)

    top = low - 0.3
    heading(fig, 0, top, "e", "What each area's readout responds to")
    low = panel_e(fig, 0, top, W, drivers, e)
    print(f"lowest content at {low:.2f} in")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"combined_figure.{ext}", dpi=450 if ext == "png" else 300)
    print("wrote", OUT / "combined_figure.pdf")


if __name__ == "__main__":
    main()
