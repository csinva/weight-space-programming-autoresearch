# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib==3.10.8", "numpy==2.2.6"]
# ///
"""Render the paper fMRI figure from the bundled aggregate results.

Usage: uv run experiments/vision/figure/make_figure.py [--data DATA_DIR] [--out OUT_DIR]

Writes a 7.5 inch vector PDF/SVG, 300 dpi PNG, individual B–E panels,
and supplementary attribution heatmaps. All inputs are required and mock data
are rejected. No model evaluation or dataset downloads run here.

Adapted from neuro-autoresearch/src/figures/fmri/make_figure.py. The bundled
JSON and schematic are byte-for-byte copies of artifacts/figures/fmri inputs.
Their historical protocol provenance is retained, including source-machine paths
that are metadata only; rendering never follows those paths. These historical
results must not be used as feedback for a current model search.
"""
import argparse
import json
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "results"
# Dataset schematic for panel A; its own title is cropped because the panel heading names the dataset.
SCHEMATIC = HERE / "assets/LAION-fmri.png"
SCHEMATIC_CROP = (slice(170, 1500), slice(80, 940))  # rows, columns in pixels
OUT = HERE / "output"
MM = 1 / 25.4
# Layout coordinates below are in millimetres on a 180 mm canvas; SCALE prints them at the
# 7.5 in width of the paper's other figures, so type sizes match those figures 1:1.
SCALE = 7.5 / (180 * MM)
# Colors and type follow the paper's figures (neuro_autoresearch_latex/figures_code/palette.py
# and make_ecog_figure.py): PRIMARY marks our method, NEUTRAL marks reference models.
PRIMARY, SECONDARY, NEUTRAL = "#1F77B4", "#D6487E", "#6A747C"
INK, MUTED, DOT, AXIS, HAIR = "#202A32", "#64717B", "#C4CDD3", "#9AA5AD", "#DCE2E6"
TINT_PRIMARY = "#E3EFF8"
STYLE = {
    "width": 180 * MM * SCALE, "height": 116 * MM * SCALE,
    "font": 7.6, "label": 8, "small": 7.2, "letter": 9,
    "grey": MUTED, "light": DOT, "dark": INK,
    "highlight": PRIMARY,
}
FAMILY_COLORS = {
    "edges": "#999999", "color": "#E69F00", "luminance curvature": "#0072B2",
    "corners and contours": PRIMARY, "texture statistics": SECONDARY, "global layout": "#56B4E9",
}
BAR_GREY = DOT
FAMILY_ORDER = ["color", "luminance curvature", "corners and contours", "texture statistics", "global layout"]
FAMILY_LABELS = {
    "color": "Color", "luminance curvature": "Luminance\ncurvature",
    "corners and contours": "Corners and\ncontours", "texture statistics": "Texture\nstatistics",
    "global layout": "Global layout",
}
# Ordered from bright peaks to dark pits of the luminance image, with a neutral saddle.
SHAPE_COLORS = {
    "dome": "#A50026", "ridge": "#F46D43", "saddle": "#E2DDA2",
    "rut": "#74ADD1", "cup": "#313695",
}
POOLED, EARLY, VENTRAL = "early and ventral visual streams", "early visual stream", "ventral visual stream"
REGION_LABELS = {POOLED: "Pooled", EARLY: "Early", VENTRAL: "Ventral"}
# Panel rectangles in millimetres: left, bottom, width, height. Individual exports
# use the same rendering functions and typography, with bounds fitted to all labels.
SLOTS = {"a": (4, 60, 30, 52), "b": (38, 60, 53, 52),
         "c": (96, 60, 80, 52), "d": (4, 4, 67, 54), "e": (78, 4, 98, 54)}
SUPPLEMENT_SLOTS = {"d": (4, 4, 62, 71), "e": (73, 4, 104, 71)}
TITLES = {"a": "LAION-fMRI", "b": "Search trajectory", "c": "Transfer to NSD",
          "d": "Computational families", "e": "Inside texture statistics"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": STYLE["font"], "axes.labelsize": STYLE["label"], "axes.titlesize": STYLE["label"],
    "xtick.labelsize": STYLE["font"], "ytick.labelsize": STYLE["font"], "legend.fontsize": STYLE["small"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.labelpad": 1.5, "axes.titlepad": 3, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
})


def to_figure(fig, x, y):
    """Map layout millimetres (180 mm canvas) to figure fractions."""
    fw, fh = fig.get_size_inches() / (MM * SCALE)
    return x / fw, y / fh


def axes(fig, slot, rect, **kwargs):
    """Add axes in mm relative to a panel; coordinates describe layout, never data."""
    sx, sy, _, _ = slot
    x, y, w, h = rect
    left, bottom = to_figure(fig, sx + x, sy + y)
    width, height = to_figure(fig, w, h)
    return fig.add_axes([left, bottom, width, height], **kwargs)


def text(fig, slot, x, y, label, **kwargs):
    return fig.text(*to_figure(fig, slot[0] + x, slot[1] + y), label,
                    ha=kwargs.pop("ha", "left"), va=kwargs.pop("va", "bottom"), **kwargs)


def heading(fig, slot, key):
    text(fig, slot, 0, slot[3] - 3.5, f"{key.lower()}  {TITLES[key]}", fontsize=STYLE["letter"],
         fontweight="bold", color=INK)


def placeholder(fig, slot, label):
    ax = axes(fig, slot, (2, 11, slot[2] - 4, 29))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(HAIR)
        spine.set_linestyle((0, (2, 4)))
        spine.set_linewidth(.4)
    ax.text(0.5, 0.5, label, ha="center", va="center", color=STYLE["grey"], fontsize=STYLE["small"], transform=ax.transAxes)


def finite_values(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def limits(values, zero=False, pad=0.08):
    values = finite_values(values)
    lo, hi = (float(values.min()), float(values.max())) if values.size else (0., 1.)
    if zero:
        lo, hi = min(0., lo), max(0., hi)
    span = hi - lo or max(abs(hi), 0.1)
    return lo - pad * span, hi + pad * span


def clean_matrix(ax):
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def heatmap(ax, matrix, cmap, norm):
    """Keep the small response/attribution matrices as vector cells in PDFs."""
    rows, columns = matrix.shape
    mesh = ax.pcolormesh(np.arange(columns + 1) - 0.5, np.arange(rows + 1) - 0.5,
                         np.ma.masked_invalid(matrix), cmap=cmap, norm=norm,
                         shading="flat", edgecolors="face", linewidth=0.1, antialiased=False)
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    return mesh


def figure_key(fig, slot, x, y, handles, labels, columns=2):
    return fig.legend(handles, labels, loc="lower left", frameon=False, ncol=columns,
                      bbox_to_anchor=to_figure(fig, slot[0] + x, slot[1] + y),
                      borderaxespad=0, handlelength=1.4, handletextpad=.45,
                      columnspacing=1.3, labelspacing=.4, fontsize=STYLE["small"], labelcolor=INK)


def mean_key(fig, slot, x, y, n, color=None):
    handles = [Line2D([], [], marker="o", color=NEUTRAL, alpha=.6, lw=0, markersize=3, markeredgewidth=0),
               Line2D([], [], marker="|", color=color or STYLE["dark"], lw=0,
                      markersize=6.5, markeredgewidth=1.4)]
    return figure_key(fig, slot, x, y, handles, [f"Subjects ($n$ = {n})", "Mean"])


# ---------------------------------------------------------------- A: dataset schematic
def panel_a(fig, slot):
    if not SCHEMATIC.exists():
        placeholder(fig, slot, "Dataset overview\nArtwork reserved")
        return
    ax = axes(fig, slot, (0, 0, slot[2], slot[3] - 5.5))
    image = plt.imread(SCHEMATIC)[SCHEMATIC_CROP][..., :3].copy()
    image = image / 255. if image.dtype == np.uint8 else image
    image[(image > 0.94).all(-1)] = 1.  # the artwork's off-white background would show as a box
    ax.imshow(image, interpolation="lanczos")
    ax.set_anchor("N")
    ax.axis("off")


# ---------------------------------------------------------------- B: search trajectory
def panel_b(fig, slot, b):
    cand, best = b["candidates"], b["running_best"]
    if not cand or not best:
        placeholder(fig, slot, "No scored candidates")
        return
    ax = axes(fig, slot, (9, 11, 33, 29))
    x = np.asarray([c["index"] for c in cand])
    bx = np.asarray([r["index"] for r in best])
    ax.scatter(x, [c["pooled"] for c in cand], s=3, color=DOT, lw=0,
               zorder=1, rasterized=False)
    line_styles = {"pooled": (PRIMARY, 1.6, "-", "Pooled"),
                   "early": (NEUTRAL, 0.9, "-", "Early"),
                   "ventral": (NEUTRAL, 0.9, (0, (3, 2)), "Ventral")}
    for key, (color, width, style, _) in line_styles.items():
        ax.step(bx, [r[key] for r in best], where="post", color=color, lw=width, ls=style,
                zorder=3 if key == "pooled" else 2)
    # The first few candidates score far below the rest; clip them so the climb stays visible.
    pooled = np.asarray([c["pooled"] for c in cand])
    late = [r for r in best if r["index"] >= 0.02 * x.max()]
    values = list(pooled[pooled >= np.percentile(pooled, 2)]) + [r[k] for r in late for k in line_styles]
    ax.set_ylim(*limits(values, pad=0.06))
    ax.set_xlim(min(0, x.min()), max(x.max(), bx.max(), 1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 5, 10]))
    ax.set_xlabel("Evaluated candidates")
    ax.set_ylabel("Encoding $r$")
    # Separate endpoint labels in axes coordinates, so converging stream scores
    # remain readable. Connectors end at the unchanged data values.
    endpoints = [(k, (float(best[-1][k]) - ax.get_ylim()[0]) / np.ptp(ax.get_ylim()))
                 for k in line_styles if np.isfinite(best[-1][k])]
    endpoints.sort(key=lambda item: item[1])
    positions = np.clip([v for _, v in endpoints], 0.04, 0.96)
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + 0.11)
    if len(positions) and positions[-1] > 0.96:
        positions -= positions[-1] - 0.96
    for (key, actual), position in zip(endpoints, positions):
        color = line_styles[key][0]
        pooled_line = key == "pooled"
        ax.annotate(line_styles[key][3], xy=(1, actual), xytext=(1.04, position),
                    xycoords="axes fraction", textcoords="axes fraction", ha="left", va="center",
                    color=PRIMARY if pooled_line else MUTED, fontsize=STYLE["small"],
                    fontweight="bold" if pooled_line else "normal", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6})
    champ = next((c for c in cand if c["id"] == b["champion"]), None)
    if champ is not None:
        ax.scatter(champ["index"], champ["pooled"], s=16, color=PRIMARY, lw=0, zorder=4)
        ax.annotate(f"Searched model {champ['pooled']:.3f}", (champ["index"], champ["pooled"]), xytext=(-2, 5),
                    textcoords="offset points", ha="right", va="bottom", color=PRIMARY,
                    fontsize=STYLE["small"], fontweight="bold")
    clipped = int(np.sum(pooled < ax.get_ylim()[0]))
    note = "Gray dots: candidates"
    if clipped:
        note += f"; {clipped} below plotted range"
    text(fig, slot, 9, 0, note, fontsize=STYLE["small"], color=MUTED)



# ---------------------------------------------------------------- C: generalization to NSD
def model_label(model):
    if model["kind"] == "searched":
        return "Searched model"
    label = model["label"].replace("_", " ")
    return label.replace(" (", "\n(") if model["kind"] == "reference" else label


def panel_c(fig, slot, c):
    regions = [r for r in (POOLED, EARLY, VENTRAL) if r in c["regions"]]
    models = c["models"]
    if not models or not regions:
        placeholder(fig, slot, "No model scores")
        return
    # Every row uses increasing y coordinates; category placement does not invert
    # an axis. Baselines are ranked once by pooled performance across all columns.
    order = sorted((k for k in models if models[k]["kind"] != "reference"),
                   key=lambda k: models[k]["mean"][POOLED], reverse=True)
    refs = [k for k in models if models[k]["kind"] == "reference"]
    order += refs
    positions = {key: len(order) - 1 - i for i, key in enumerate(order)}
    if refs:
        for key in refs:
            positions[key] -= 0.45
    allv = [m["scores"][s][r] for m in models.values() for s in c["subjects"] for r in regions]
    allv += [m["mean"][r] for m in models.values() for r in regions]
    lo, hi = limits(allv, pad=0)
    lo, hi = 0.1 * np.floor(lo / 0.1), 0.1 * np.ceil(hi / 0.1)
    if lo == hi:
        lo, hi = 0., 1.
    ticks = np.array([lo, (lo + hi) / 2, hi])
    magnitude = max(abs(ticks[0]), abs(ticks[-1]))
    exponent = int(np.floor(np.log10(magnitude))) if magnitude and (magnitude < 0.01 or magnitude >= 1000) else 0
    scale = 10. ** exponent
    margin = 0.04 * (ticks[-1] - ticks[0])
    bounds = ticks[0] - margin, ticks[-1] + margin
    jit = np.linspace(-0.18, 0.18, len(c["subjects"])) if len(c["subjects"]) > 1 else [0]
    width = (59 - 4 * (len(regions) - 1)) / len(regions)
    for g, region in enumerate(regions):
        ax = axes(fig, slot, (21 + g * (width + 4), 11, width, 29))
        for name in order:
            m, y = models[name], positions[name]
            color = PRIMARY if m["kind"] == "searched" else NEUTRAL
            ax.scatter([m["scores"][s][region] for s in c["subjects"]], y + np.asarray(jit),
                       s=7, color=color, alpha=0.6, lw=0, zorder=2)
            ax.plot(m["mean"][region], y, marker="|", ms=7, mew=1.5, lw=0,
                    color=color if m["kind"] == "searched" else INK, zorder=3)
        searched = [k for k in order if models[k]["kind"] == "searched"]
        for key in searched:
            ax.axhspan(positions[key] - 0.45, positions[key] + 0.45, color=TINT_PRIMARY, lw=0, zorder=0)
        if refs:
            ax.axhline(positions[refs[0]] + 0.72, lw=0.6, color=HAIR)
        ax.set_title(REGION_LABELS[region])
        ax.set_yticks([positions[k] for k in order], [model_label(models[k]) for k in order])
        ax.tick_params(axis="y", length=0, labelleft=g == 0, pad=3)
        ax.spines["left"].set_visible(False)
        if g == 0:
            for label, key in zip(ax.get_yticklabels(), order):
                if models[key]["kind"] == "searched":
                    label.set_fontweight("bold")
                    label.set_color(PRIMARY)
                elif models[key]["kind"] == "reference":
                    label.set_color(STYLE["grey"])
        ax.set_ylim(min(positions.values()) - 0.6, max(positions.values()) + 0.6)
        ax.set_xlim(bounds)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value / scale:.2g}".replace("-", "−")))
    mean_key(fig, slot, 21, 44.5, len(c["subjects"]))
    units = rf" ($\times 10^{{{exponent}}}$)" if exponent else ""
    text(fig, slot, 50.5, 4, "Encoding $r$" + units, ha="center", fontsize=STYLE["label"])
    text(fig, slot, 50.5, 0, f"NSD · {c['n_test_images']:,} unseen test images", ha="center",
         fontsize=STYLE["small"], color=STYLE["grey"])


# ---------------------------------------------------------------- D: computational families
def family_rows(d):
    names = {k: d.get("families", {}).get(k, k) for k in d["share_beyond_edges"]}
    by_name = {v: k for k, v in names.items()}
    return [by_name[f] for f in FAMILY_ORDER]


def cell_text_color(rgba):
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return "black" if luminance > 0.179 else "white"


def percentage_norm(values, total=False):
    finite = finite_values(values)
    # Small negative Shapley shares (rounding to 0) do not warrant extending the scale.
    lo = min(0., float(finite.min())) if finite.size and finite.min() < -2 else 0.
    hi = max(100. if total else 1., float(finite.max())) if finite.size else 1.
    # Use a sequential map as specified, including negative Shapley contributions.
    return mcolors.Normalize(10 * np.floor(lo / 10), max(10, 10 * np.ceil(hi / 10)))


def annotate_matrix(ax, matrix, cmap, norm):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            label = f"{value:.0f}" if np.isfinite(value) else "–"
            if label == "-0":
                label = f"{value:.1f}"  # Do not erase a small negative contribution.
            ax.text(j, i, label, ha="center", va="center",
                    color=cell_text_color(cmap(norm(value))) if np.isfinite(value) else STYLE["grey"])


def supplement_percentage_norm(matrix):
    """Use a common percentage range, extending it to retain outlying shares."""
    finite = finite_values(matrix)
    lo = min(0., float(finite.min())) if finite.size else 0.
    hi = max(60., float(finite.max())) if finite.size else 60.
    upper = 10 * np.ceil(hi / 10)
    if lo < 0:
        return mcolors.TwoSlopeNorm(vmin=10 * np.floor(lo / 10), vcenter=0, vmax=upper)
    return mcolors.Normalize(0, upper)


def panel_d_heatmap(fig, slot, d, norm=None):
    rois, keys = d["rois"], family_rows(d)
    matrix = np.asarray([[d["share_beyond_edges"][k][r] for r in rois] for k in keys]) * 100
    norm = norm if norm is not None else supplement_percentage_norm(matrix)
    cmap = plt.get_cmap("RdBu" if isinstance(norm, mcolors.TwoSlopeNorm) else "viridis").with_extremes(bad="#eeeeee")
    ax = axes(fig, slot, (19, 15, 31, 36))
    # Reverse row positions, not axes: numeric axes keep increasing directions.
    shown = matrix[::-1]
    im = heatmap(ax, shown, cmap, norm)
    annotate_matrix(ax, shown, cmap, norm)
    ax.set_xticks(range(len(rois)), rois)
    ax.set_yticks(range(len(keys)), [FAMILY_LABELS[f] for f in FAMILY_ORDER[::-1]])
    ax.tick_params(axis="y", pad=4)
    clean_matrix(ax)
    # Edges are the baseline that every cell is measured against.
    sax = axes(fig, slot, (19, 54, 31, 5))
    sax.set_facecolor("#eeeeee")
    sax.set_xticks([])
    sax.set_yticks([0.5], ["Edges"])
    sax.set_ylim(0, 1)
    sax.text(0.5, 0.5, "Baseline", transform=sax.transAxes, ha="center", va="center", color=STYLE["grey"])
    clean_matrix(sax)
    sax.tick_params(axis="y", pad=4)
    counts = d.get("features_per_family")
    if counts:
        text(fig, slot, 61, 60.5, "Features", ha="right", color=STYLE["grey"])
        text(fig, slot, 61, 56.5, f"{counts['E']:,}", ha="right", va="center")
        for i, key in enumerate(keys):
            y = 51 - (i + 0.5) * 36 / len(keys)
            text(fig, slot, 61, y, f"{counts[key]:,}", ha="right", va="center")
    cax = axes(fig, slot, (19, 6, 31, 2))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.set_ticks(sorted(set([norm.vmin, norm.vmax, *np.arange(0, norm.vmax, 20)])))
    cb.ax.tick_params(length=1.5, pad=1.5)
    cb.set_label("Share of gain beyond edges (%)", labelpad=2)


# ---------------------------------------------------------------- E: texture operations
SUBSTAT_ORDER = ["M", "V", "J", "O", "R"]
SUBSTAT_NOTES = {"M": "texture boundaries", "V": "co-occurring orientations and colors",
                 "J": "co-occurring curvature, phase and color", "O": "structure across octaves",
                 "R": "co-occurring local geometry"}


E_ROIS = ["hV4", "FFA", "PPA"]  # texture adds little beyond edges in V1-V3 (panel D), so shares there are noisy


def panel_e_heatmap(fig, slot, e, norm=None):
    rois = [r for r in E_ROIS if r in e["rois"]]
    keys = [k for k in ("M", "V", "O", "R", "J") if k in e["share_beyond_edges"]]
    matrix = np.asarray([[e["share_beyond_edges"][k][r] for r in rois] for k in keys]) * 100
    norm = norm if norm is not None else supplement_percentage_norm(matrix)
    cmap = plt.get_cmap("RdBu" if isinstance(norm, mcolors.TwoSlopeNorm) else "viridis").with_extremes(bad="#eeeeee")
    x0, w, y0, h = 42, 42, 15, 36
    ax = axes(fig, slot, (x0, y0, w, h))
    shown = matrix[::-1]
    im = heatmap(ax, shown, cmap, norm)
    annotate_matrix(ax, shown, cmap, norm)
    ax.set_xticks(range(len(rois)), rois)
    ax.set_yticks(range(len(keys)), [E_LABELS[k] for k in keys[::-1]])
    ax.tick_params(axis="y", pad=5)
    clean_matrix(ax)
    text(fig, slot, 5, 60.5, "Attribution within the edges + texture model", color=STYLE["grey"])
    text(fig, slot, 97, 54.5, "Features", ha="right", color=STYLE["grey"])
    for i, key in enumerate(keys):
        y = y0 + h - (i + 0.5) * h / len(keys)
        n = e["substats"][key]["n_features"]
        text(fig, slot, 97, y, f"{n:,}", ha="right", va="center")
    cax = axes(fig, slot, (x0, 6, w, 2))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.set_ticks(sorted(set([norm.vmin, norm.vmax, *np.arange(0, norm.vmax, 20)])))
    cb.ax.tick_params(length=1.5, pad=1.5)
    cb.set_label("Share of texture-model gain beyond edges (%)", labelpad=2)


# ---------------------------------------------------------------- Main contribution plots

def panel_d(fig, slot, d):
    rois, keys = d["rois"], family_rows(d)
    names = d["families"]
    positions = {"V1": 0, "V2": 1, "V3": 2, "hV4": 3, "FFA": 4.6, "PPA": 5.8}
    xs = np.array([positions[r] for r in rois])
    ax = axes(fig, slot, (11, 14, 51, 29))
    all_values = []
    muted = {"color": ("#A9B3BA", ":", "o"), "luminance curvature": (NEUTRAL, "--", "s"),
             "global layout": ("#B9C2C8", "-.", "^")}
    for key in keys:
        name = names[key]
        ys = np.array([d["share_beyond_edges"][key][r] * 100 for r in rois])
        all_values.extend(ys)
        strong = name in ("corners and contours", "texture statistics")
        color, linestyle, marker = (FAMILY_COLORS[name], "-", "D" if name == "texture statistics" else "o") if strong else muted[name]
        early = np.array([r in ("V1", "V2", "V3", "hV4") for r in rois])
        ax.plot(xs[early], ys[early], color=color, ls=linestyle, lw=1.6 if strong else .8,
                zorder=3 if strong else 1)
        ax.scatter(xs, ys, color=color, marker=marker, s=16 if strong else 8,
                   linewidths=0, zorder=4 if strong else 2)
    lo, hi = limits(all_values, zero=True)
    ax.set_ylim(min(0, lo) if min(all_values) < 0 else 0, max(60, hi))
    ax.set_xlim(-.25, 6.05)
    ax.set_xticks(xs, rois)
    ax.set_yticks(np.arange(0, ax.get_ylim()[1] + .01, 20))
    ax.set_ylabel("Share of gain\nbeyond edges (%)")
    ax.spines["bottom"].set_bounds(xs.min(), xs.max())
    ax.tick_params(axis="x", length=2, pad=3)
    # Keys sit outside the axes; all text stays neutral for print accessibility.
    strong = ["corners and contours", "texture statistics"]
    # Direct labels for the two families the panel is about, as in the paper's other figures.
    # Corners and contours above its line at V1; texture statistics beside its hV4 point.
    corners = next(k for k in keys if names[k] == strong[0])
    texture = next(k for k in keys if names[k] == strong[1])
    ax.annotate("Corners and contours", (xs[0], d["share_beyond_edges"][corners][rois[0]] * 100),
                xytext=(-3, 8), textcoords="offset points", ha="left", va="bottom",
                color=FAMILY_COLORS[strong[0]], fontsize=STYLE["small"], fontweight="bold")
    ax.annotate("Texture\nstatistics", (positions["hV4"], d["share_beyond_edges"][texture]["hV4"] * 100),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center", linespacing=1.05,
                color=FAMILY_COLORS[strong[1]], fontsize=STYLE["small"], fontweight="bold")
    handles = [Line2D([], [], color=muted[name][0], ls=muted[name][1],
                      marker=muted[name][2], markersize=3, lw=.8) for name in muted]
    figure_key(fig, slot, 11, 0, handles,
               ["Color", "Luminance curvature", "Global layout"], columns=2)


E_LABELS = {"M": "Texture boundaries", "V": "Orientation/color\nco-occurrence",
            "O": "Cross-scale structure", "R": "Geometry co-occurrence",
            "J": "Joint feature\nco-occurrence"}


def panel_e(fig, slot, e):
    rois = [r for r in E_ROIS if r in e["rois"]]
    keys = [k for k in ("M", "V", "O", "R", "J") if k in e["share_beyond_edges"]]
    subjects = e.get("subjects", [])
    values = [100 * e["share_beyond_edges"][k][r] for k in keys for r in rois]
    for k in keys:
        for r in rois:
            values.extend(100 * v for v in e.get("share_beyond_edges_subjects", {}).get(k, {}).get(r, {}).values())
    low, high = limits(values, zero=True)
    low, high = min(-5, low), max(60, high)
    offsets = np.linspace(-.14, .14, len(subjects)) if len(subjects) > 1 else np.zeros(len(subjects))
    # Tie e to d: the pink family in d, split into its operations.
    text(fig, slot, 5, 45.5, "Texture statistics (pink in d),\nsplit into its operations",
         color=FAMILY_COLORS["texture statistics"], fontsize=STYLE["small"], fontweight="bold", linespacing=1.1)
    mean_key(fig, slot, 39, 0, len(subjects), FAMILY_COLORS["texture statistics"])
    for col, roi in enumerate(rois):
        ax = axes(fig, slot, (39 + 21 * col, 14, 17, 29))
        means = [100 * e["share_beyond_edges"][k][roi] for k in keys]
        for i, key in enumerate(keys):
            y = len(keys) - 1 - i
            per_subject = e.get("share_beyond_edges_subjects", {}).get(key, {}).get(roi, {})
            for subject, offset in zip(subjects, offsets):
                if subject in per_subject:
                    ax.scatter(100 * per_subject[subject], y + offset, s=7,
                               color=NEUTRAL, alpha=.6, linewidths=0, zorder=3)
            color = FAMILY_COLORS["texture statistics"]
            ax.plot(means[i], y, marker="|", ms=7, mew=1.5, color=color, zorder=4)
        ax.set_xlim(low, high)
        ax.set_ylim(-.5, len(keys) - .5)
        ax.set_xticks([v for v in range(int(np.ceil(low / 20)) * 20, int(high) + 1, 20)])
        ax.set_yticks(range(len(keys)), [E_LABELS[k] for k in keys[::-1]] if col == 0 else [])
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=HAIR, lw=.6, zorder=0)
        ax.set_title(roi)
    text(fig, slot, 68.5, 6, "Share of texture-model gain beyond edges (%)", ha="center", fontsize=STYLE["label"])


PANELS = {"b": ("b.json", panel_b), "c": ("c.json", panel_c),
          "d": ("d.json", panel_d), "e": ("e.json", panel_e)}


def load(data, name):
    path = data / name
    if not path.exists():
        return None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive)
    return json.loads(path.read_text())


def is_mock(payload):
    if payload is None:
        return False
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = json.loads(str(provenance))
    return bool(provenance.get("mock", False))


def validate_data(data):
    """Reject incomplete or synthetic inputs before creating any output files."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

    for key, payload in data.items():
        require(isinstance(payload, dict), f"Missing panel {key.upper()} data")
        require(isinstance(payload.get("provenance"), dict) and payload["provenance"],
                f"Panel {key.upper()} requires provenance")
        require(not is_mock(payload), f"Panel {key.upper()} contains mock data")
    b, c, d, e = (data[key] for key in "bcde")
    require(b["candidates"] and b["running_best"], "Panel B requires scored candidates and running best")
    require(len(b["candidates"]) == len(b["running_best"]), "Panel B requires one running-best entry per candidate")
    require(any(row["id"] == b["champion"] for row in b["candidates"]), "Panel B champion is missing")
    for row in b["candidates"] + b["running_best"]:
        require(all(finite(row[key]) for key in ("index", "pooled", "early", "ventral")),
                "Panel B contains a missing or nonfinite score")
    regions = (POOLED, EARLY, VENTRAL)
    require(c["subjects"] and c["models"] and c["n_test_images"] > 0, "Panel C is incomplete")
    require(set(regions) <= set(c["regions"]), "Panel C requires pooled, early and ventral scores")
    expected_models = {"erjpu", "pixels", "gabor", "gabor_norm", "gist", "hmax", "scattering",
                       "texture", "curvature", "alexnet_conv4"}
    require(set(c["models"]) == expected_models,
            "Panel C requires all ten paper models; missing: "
            + ", ".join(sorted(expected_models - set(c["models"]))))
    require(any(model["kind"] == "searched" for model in c["models"].values()), "Panel C requires the searched model")
    for model in c["models"].values():
        require(all(finite(model["mean"][region]) for region in regions), "Panel C has a nonfinite mean")
        require(all(finite(model["scores"][subject][region]) for subject in c["subjects"] for region in regions),
                "Panel C has a missing or nonfinite subject score")
    rois = {"V1", "V2", "V3", "hV4", "FFA", "PPA"}
    for key, payload, groups in (("D", d, family_rows(d)), ("E", e, SUBSTAT_ORDER)):
        require(set(payload["rois"]) == rois and payload["subjects"], f"Panel {key} requires six ROIs and subjects")
        for group in groups:
            for roi in rois:
                require(finite(payload["share_beyond_edges"][group][roi]), f"Panel {key} has a nonfinite share")
                require(all(finite(payload["share_beyond_edges_subjects"][group][roi][subject])
                            for subject in payload["subjects"]), f"Panel {key} has incomplete subject shares")
    require(all(d["features_per_family"][key] > 0 for key in ["E", *family_rows(d)]),
            "Panel D requires feature counts")
    require(all(e["substats"][key]["n_features"] > 0 for key in SUBSTAT_ORDER),
            "Panel E requires feature counts")


def full_figure(data):
    fig = plt.figure(figsize=(STYLE["width"], STYLE["height"]))
    heading(fig, SLOTS["a"], "a")
    panel_a(fig, SLOTS["a"])
    for key, (filename, function) in PANELS.items():
        slot = SLOTS[key]
        heading(fig, slot, key)
        if data[key] is None:
            placeholder(fig, slot, f"Awaiting {filename}")
        else:
            function(fig, slot, data[key])
    mock_panels = [key.upper() for key, payload in data.items() if is_mock(payload)]
    if mock_panels:
        note = text(fig, (0, 0), 4, 1, "MOCK DATA: " + ", ".join(mock_panels), fontsize=6.5,
                    color=STYLE["dark"], fontweight="bold")
        note.set_gid("mock-data-note")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        data = {key: load(args.data, name) for key, (name, _) in PANELS.items()}
        validate_data(data)
        if not SCHEMATIC.is_file():
            raise ValueError(f"Missing panel A schematic: {SCHEMATIC}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Incomplete figure inputs: {error}")
    (args.out / "panels").mkdir(parents=True, exist_ok=True)
    fig = full_figure(data)
    fig.savefig(args.out / "fmri_figure.pdf")
    fig.savefig(args.out / "fmri_figure.png", dpi=300)
    fig.savefig(args.out / "fmri_figure.svg")
    plt.close(fig)
    for key, (filename, function) in PANELS.items():
        _, _, w, h = SLOTS[key]
        panel = plt.figure(figsize=((w + 8) * MM * SCALE, (h + 8) * MM * SCALE))
        slot = (4, 4, w, h)
        heading(panel, slot, key)
        if data[key] is None:
            placeholder(panel, slot, f"Awaiting {filename}")
        else:
            function(panel, slot, data[key])
        if is_mock(data[key]):
            text(panel, slot, w, h - 3.4, "Mock data", ha="right",
                 fontsize=STYLE["small"], color=STYLE["grey"])
        for suffix in ("pdf", "png"):
            panel.savefig(args.out / "panels" / f"{key}.{suffix}", bbox_inches="tight",
                          pad_inches=MM, dpi=450)
        plt.close(panel)
    supplement = plt.figure(figsize=(STYLE["width"], 80 * MM * SCALE))
    shares = [100 * value for key in ("d", "e") if data[key] is not None
              for by_roi in data[key]["share_beyond_edges"].values()
              for roi, value in by_roi.items() if key == "d" or roi in E_ROIS]
    norm = supplement_percentage_norm(shares)
    for key, function in (("d", panel_d_heatmap), ("e", panel_e_heatmap)):
        slot = SUPPLEMENT_SLOTS[key]
        heading(supplement, slot, key)
        if data[key] is not None:
            function(supplement, slot, data[key], norm=norm)
        else:
            placeholder(supplement, slot, f"Awaiting {PANELS[key][0]}")
    supplement.savefig(args.out / "fmri_heatmaps.pdf")
    supplement.savefig(args.out / "fmri_heatmaps.png", dpi=450)
    plt.close(supplement)
    inputs = {f"results/{name}": args.data / name for name, _ in PANELS.values()}
    inputs["assets/LAION-fmri.png"] = SCHEMATIC
    manifest = {
        "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
        "provenance": {key: payload["provenance"] for key, payload in data.items()},
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    }
    (args.out / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out / 'fmri_figure.pdf'}")


if __name__ == "__main__":
    main()
