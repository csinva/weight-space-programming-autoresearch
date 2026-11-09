"""Figure: what the searched LAION model (erjpu) computes.

a  Intermediate maps of the model on three LAION stimuli (a photo, a radial grating, an illusory triangle):
   one representative map per computation, each column scaled over its three images.
b  The model's 9,996 features by computational family (counts from src/figures/fmri/d_families.py).
c  The single feature that best tracks each brain area, with the three LAION images it predicts most strongly.
   The best feature has the highest squared correlation with the area's voxels on the shared images, averaged
   over voxels and the five subjects (artifacts/analysis/feature_atlas/atlas.json); the panel reports the signed
   mean correlation. Images are those with the highest predicted response among sub-01's 6,204 images: the
   highest feature values when the correlation is positive, the lowest when it is negative.

Maps are computed with the model's own functions at its finest scale; the texture-boundary map repeats the
model's second-order stage on that scale for display (the model applies it one octave coarser).

Writes artifacts/figures/model/model_figure.{pdf,png,svg}; data are cached in artifacts/figures/model/data.npz.
Needs the LAION data (as for src/analysis/feature_atlas/data.py) only when the cache is missing.

Usage: python src/figures/model/make_figure.py [--refresh]
"""
import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FROZEN = REPO / ("artifacts/final/laion-visual-encoding-to-nsd-encoding/"
                 "cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633/model.py")
ATLAS = REPO / "artifacts/analysis/feature_atlas/atlas.json"
FAMILIES = REPO / "artifacts/figures/fmri/data/d.json"
FEATURES = REPO / "artifacts/figures/fmri/data/d/features_sub-01.npy"
OUT = REPO / "artifacts/figures/model"
CACHE = OUT / "data.npz"

# Paper palette (neuro_autoresearch_latex/figures_code/palette.py); family colors follow Fig. 3d.
PRIMARY, SECONDARY, NEUTRAL = "#1F77B4", "#D6487E", "#6A747C"
INK, MUTED, AXIS = "#202A32", "#64717B", "#9AA5AD"
FAMILY = {  # key: (label, color, text color on the bar)
    "E": ("Edges", NEUTRAL, "white"), "K": ("Color", "#A9B3BA", INK), "T": ("Texture statistics", SECONDARY, "white"),
    "C": ("Corners and contours", PRIMARY, "white"), "S": ("Luminance curvature", "#C4CDD3", INK),
    "X": ("Mixed", "#DCE2E6", INK), "G": ("Global layout", "#E9EDF0", INK),
}
BAR_ORDER = ["E", "K", "T", "C", "S", "X", "G"]
# Example images: indices into linspace(len(train) + 5, len(all) - 5, 6), as in the feature atlas.
EXAMPLES = (2, 4, 5)
# (header, family, map key)
COLUMNS = [("Oriented\nenergy", "E", "energy"), ("Color\nenergy", "K", "color"),
           ("Texture\nboundaries", "T", "texture"), ("Corners and\njunctions", "C", "corner"),
           ("Contour\ncurvature", "C", "curvature"), ("Bright\nlines", "C", "ridge"),
           ("Saddles", "S", "saddle")]
ROIS = ["V1", "hV4", "FFA", "PPA"]
PLAIN = {  # atlas feature name -> plain description
    "edge coherence": "Edge coherence in one region",
    "corner strength (structure tensor min)": "Density of corners",
    "line anisotropy |Lxx-Lyy|": "Horizontal, vertical lines",
}
NEGATIVE = {"edge coherence": "Mixed orientations"}  # low coherence: no single orientation

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
})


def load_model():
    module = types.ModuleType("champion")
    exec(compile(FROZEN.read_text(), str(FROZEN), "exec"), module.__dict__)
    return module


def compute():
    """Maps on three example images, plus each area's best feature and its top images."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO / "src/figures/fmri"))
    sys.path.insert(0, str(REPO / "src"))
    os.environ.update(NEURO_DATASET="laion", NEURO_STAGE="search", NEURO_METRIC="encoding",
                      NEURO_REGION="visual", NEURO_IMSIZE="128")
    import neuro_eval.datasets as nd
    from d_families import v4_selected_ids
    from neuro_eval.datasets import get_dataset, load_split, with_subject
    nd._selected_ids = v4_selected_ids
    base = get_dataset("laion", region="visual", metric="encoding")
    Xtr, Xte, *_ = load_split(with_subject(base, "sub-01"))
    X = torch.cat([Xtr, Xte]).reshape(-1, 128, 128, 3).numpy()
    examples = [int(i) for i in np.linspace(len(Xtr) + 5, len(X) - 5, 6)]
    images = X[[examples[k] for k in EXAMPLES]]

    m = load_model()
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().clamp(0, 1)
    with torch.no_grad():
        x_ret = (x + 0.04).pow(0.45)
        r, g, b = x_ret[:, 0:1], x_ret[:, 1:2], x_ret[:, 2:3]
        lum, rg, by = (r + g + b) / math.sqrt(3.0), (r - g) / math.sqrt(2.0), (r + g - 2.0 * b) / math.sqrt(6.0)
        filters = m._make_gabor_bank(x.device, torch.float32)
        chrom = filters[[16, 17, 20, 21, 24, 25, 28, 29, 32, 33]]
        norm_map, even_l, odd_l, energy_l, lum_ref = m._stage1_filter(lum, rg, by, filters, chrom)
        ori = 0.5 * (norm_map[:, 0:8] + norm_map[:, 8:16])
        curv = m._nematic_curvature_map(ori)
        phase = m._phase_congruency_map(even_l, odd_l, energy_l, lum_ref, norm_map)
        geom = m._surface_geom_map(lum, rg, by)
        # Second-order stage (Stage 2A) on the finest scale.
        sel = torch.cat([ori, norm_map[:, 20:24], norm_map[:, 24:28]], dim=1)
        sel = F.avg_pool2d(sel, 2, 2)
        sel = sel - sel.mean(dim=(2, 3), keepdim=True)
        s2 = m._make_stage2_bank(x.device, torch.float32).repeat(16, 1, 1, 1)
        conv = F.conv2d(F.pad(sel, (2, 2, 2, 2), mode="reflect"), s2, groups=16)
        conv = conv.reshape(len(x), 16, 9, sel.shape[2], sel.shape[3])
        s2_energy = torch.sqrt(conv[:, :, 0:8:2] ** 2 + conv[:, :, 1:8:2] ** 2 + 1e-6)
        texture = F.interpolate(torch.log1p(s2_energy).mean(dim=(1, 2)).unsqueeze(1), size=128, mode="bilinear")
    maps = {"energy": norm_map[:, 0:16].mean(1), "color": norm_map[:, 20:28].mean(1), "texture": texture[:, 0],
            "corner": curv[:, 5], "curvature": curv[:, 10], "ridge": phase[:, 0], "saddle": geom[:, 2]}
    maps = {k: v.numpy() for k, v in maps.items()}

    atlas = json.loads(ATLAS.read_text())
    feats = atlas["features"]
    best = [max(range(len(feats)), key=lambda j: feats[j][6][k]) for k in range(len(ROIS))]
    # Signed correlation, averaged over each area's voxels and the five subjects, as in the atlas's r².
    from d_families import V4_SUBJECTS
    signed = np.zeros(len(ROIS))
    for subject in V4_SUBJECTS:
        Xtr_s, _, _, Yte, *_ = load_split(with_subject(base, subject))
        Fs = np.load(FEATURES.parent / f"features_{subject}.npy", mmap_mode="r")[len(Xtr_s):][:, best]
        Fs, Y = np.asarray(Fs, dtype=np.float64), Yte.numpy().astype(np.float64)
        R = ((Fs - Fs.mean(0)) / Fs.std(0)).T @ ((Y - Y.mean(0)) / Y.std(0)) / (len(Fs) - 1)
        masks = np.load(FEATURES.parent / f"voxels_{subject}.npz")
        for k, roi in enumerate(ROIS):
            signed[k] += R[k, masks[f"mask_{roi}"].astype(bool)].mean() / len(V4_SUBJECTS)
    # Images with the highest predicted response: the top of the feature when it correlates positively
    # with the area, the bottom when negatively.
    A = np.load(FEATURES, mmap_mode="r")
    top = [np.argsort(-np.sign(signed[k]) * np.asarray(A[:, i], dtype=np.float64))[:4] for k, i in enumerate(best)]
    top_images = X[np.concatenate(top)].reshape(len(ROIS), 4, 128, 128, 3)
    np.savez_compressed(CACHE, images=images, top_images=top_images, best=np.array(best), signed=signed,
                        **{f"map_{k}": v for k, v in maps.items()})


def inches(fig, x, y, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def tile(ax, image, cmap=None, vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def title(fig, x, y, text):
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, text, fontsize=9, fontweight="bold", color=INK, va="bottom")


def blank(ax, color, edge=None):
    ax.set_facecolor(color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(edge is not None)
        if edge:
            spine.set_color(edge)
            spine.set_linewidth(0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recompute the cached maps and top images")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh or not CACHE.exists():
        compute()
    D = np.load(CACHE)
    atlas = json.loads(ATLAS.read_text())
    counts = json.loads(FAMILIES.read_text())["features_per_family"]
    feats = atlas["features"]

    # Drawn at the 7.5 in width and type sizes of the paper's other figures, which LaTeX scales to text width.
    W = 7.5
    gap = 0.07
    n = 1 + len(COLUMNS)
    s = (W - gap * (n - 1)) / n
    group_gap, per_area = 0.22, 3
    s_c = ((W - 3 * group_gap) / 4 - (per_area - 1) * gap) / per_area
    title_h, head_a, bar_h, key_h, label_h = 0.36, 0.42, 0.28, 0.22, 0.42
    H = (title_h + head_a + 3 * s + 2 * gap) + (title_h + 0.08 + bar_h + key_h) + (title_h + label_h + s_c) + 0.04
    fig = plt.figure(figsize=(W, H))
    W_, H_ = W, H

    # a: intermediate maps.
    y = H - 0.22
    title(fig, 0, y, "a  What each computation responds to")
    y0 = y - head_a - s
    for c, (header, fam, key) in enumerate([("Input", None, None)] + COLUMNS):
        x = c * (s + gap)
        fig.text((x + s / 2) / W_, (y0 + s + 0.08) / H_, header, ha="center", va="bottom", fontsize=8,
                 color=INK, linespacing=1.05)
        if fam:  # thin family-colored rule ties each map to its family in Fig. 3d
            blank(inches(fig, x, y0 + s + 0.035, s, 0.025), FAMILY[fam][1])
        if key:
            values = D[f"map_{key}"]
            lo, hi = np.percentile(values, 1), np.percentile(values, 99)
        for r in range(3):
            ax = inches(fig, x, y0 - r * (s + gap), s, s)
            if key:
                tile(ax, D[f"map_{key}"][r], cmap="gray", vmin=lo, vmax=hi)
            else:
                tile(ax, D["images"][r])
    y = y0 - 2 * (s + gap)

    # b: features by family, small families named in a key below the bar.
    y -= title_h
    title(fig, 0, y, "b  Features by family")
    bar_y = y - 0.08 - bar_h
    total = sum(counts[k] for k in BAR_ORDER)
    x = 0.0
    keyed = []
    for k in BAR_ORDER:
        w = W * counts[k] / total
        blank(inches(fig, x, bar_y, w, bar_h), FAMILY[k][1], edge="white")
        if w > 1.4:
            fig.text((x + w / 2) / W_, (bar_y + bar_h / 2) / H_, f"{FAMILY[k][0]} {counts[k]:,}", ha="center",
                     va="center", fontsize=8, color=FAMILY[k][2])
        else:
            keyed.append(k)
        x += w
    renderer = fig.canvas.get_renderer()
    kx, ky = W, bar_y - 0.17
    for k in reversed(keyed):  # right-aligned key, in bar order
        t = fig.text(kx / W_, ky / H_, f"{FAMILY[k][0]} {counts[k]:,}", ha="right", va="center", fontsize=7.6,
                     color=INK)
        tw = t.get_window_extent(renderer).width / fig.dpi
        sw = inches(fig, kx - tw - 0.17, ky - 0.05, 0.1, 0.1)
        blank(sw, FAMILY[k][1], edge=AXIS)
        kx -= tw + 0.17 + 0.28
    y = bar_y - key_h

    # c: best single feature per area, one group of images per area.
    y -= title_h
    title(fig, 0, y, "c  Best single feature for each area")
    group_w = per_area * s_c + (per_area - 1) * gap
    for k, roi in enumerate(ROIS):
        x0 = k * (group_w + group_gap)
        i = int(D["best"][k])
        name, family, r = feats[i][4].split(" · ")[0], feats[i][2], float(D["signed"][k])
        label = PLAIN.get(name, name) if r > 0 else NEGATIVE.get(name, "Low " + PLAIN.get(name, name).lower())
        top_line = y - 0.12
        fig.text(x0 / W_, top_line / H_, roi, fontsize=8, fontweight="bold", color=INK, va="center")
        fig.text((x0 + group_w) / W_, top_line / H_, f"$r$ = {r:.2f}".replace("-", "−"), fontsize=7.6,
                 ha="right", va="center", color=MUTED)
        highlight = family in ("C", "T")
        fig.text(x0 / W_, (top_line - 0.2) / H_, label, fontsize=7.6, va="center",
                 color=FAMILY[family][1] if highlight else INK, fontweight="bold" if highlight else "normal")
        for t in range(per_area):
            tile(inches(fig, x0 + t * (s_c + gap), y - label_h - s_c, s_c, s_c), D["top_images"][k, t])

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"model_figure.{ext}", facecolor="white")
    plt.close(fig)
    best = [(roi, feats[int(i)][4], round(float(D["signed"][k]), 3)) for k, (roi, i) in enumerate(zip(ROIS, D["best"]))]
    print("best features:", best)
    print("wrote", OUT / "model_figure.pdf")


if __name__ == "__main__":
    main()
