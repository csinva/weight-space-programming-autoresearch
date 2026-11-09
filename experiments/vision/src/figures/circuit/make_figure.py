"""Figure: how the searched LAION model (erjpu) computes the texture statistics that separate hV4, FFA and PPA.

a  The model's path from an image to three areas, drawn from its own components on one LAION image: first-stage
   kernels and energy map, then the texture operations of Fig. 3e. Link width is the operation's share of the
   texture gain beyond edges in that area (a_circuit.py).
b  Responses of three texture operations to synthetic stimuli that vary one image statistic (b_tuning.py).
c  For each area's readout, the three shared images with the highest predicted response and an image optimized
   to maximize the predicted response (c_drivers.py).

Reads artifacts/figures/circuit/data/{a,b,c}.npz; writes artifacts/figures/circuit/circuit_figure.{pdf,svg,png}.
Usage: python src/figures/circuit/make_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "artifacts/figures/circuit"
DATA = OUT / "data"

# Paper palette (neuro_autoresearch_latex/figures_code/palette.py); texture statistics are pink as in Fig. 3.
TEXTURE, NEUTRAL, INK, MUTED, AXIS, LIGHT = "#D6487E", "#6A747C", "#202A32", "#64717B", "#9AA5AD", "#C9D1D6"
LABELS = {"M": ("Texture boundaries", "oriented filters on energy maps"),
          "V": ("Orientation/color co-occurrence", "correlation between channel maps"),
          "O": ("Cross-scale structure", "correlation of fine with coarse maps"),
          "R": ("Geometry co-occurrence", ""), "J": ("Joint feature co-occurrence", "")}
W = 7.5
BODY, SMALL, TITLE = 7, 6, 9

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": BODY, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "svg.fonttype": "none",
})


class Canvas:
    """Figure addressed in inches from the bottom-left corner."""

    def __init__(self, height):
        self.h = height
        self.fig = plt.figure(figsize=(W, height))

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / W, y / self.h, w / W, h / self.h])

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", BODY)
        return self.fig.text(x / W, y / self.h, s, **kw)

    def tile(self, x, y, w, h, image, **kw):
        ax = self.axes(x, y, w, h)
        kw.setdefault("interpolation", "nearest" if image.shape[0] < 40 else "lanczos")
        ax.imshow(image, aspect="auto", **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def arrow(self, x0, y0, x1, y1):
        f = lambda x, y: (x / W, y / self.h)
        self.fig.add_artist(FancyArrowPatch(f(x0, y0), f(x1, y1), transform=self.fig.transFigure, color=AXIS,
                                            lw=0.7, arrowstyle="-|>", mutation_scale=5, shrinkA=0, shrinkB=0))

    def link(self, x0, y0, x1, y1, width, color, z):
        f = lambda x, y: (x / W, y / self.h)
        xm = (x0 + x1) / 2
        path = MPath([f(x0, y0), f(xm, y0), f(xm, y1), f(x1, y1)], [MPath.MOVETO] + [MPath.CURVE4] * 3)
        self.fig.add_artist(PathPatch(path, transform=self.fig.transFigure, fc="none", ec=color, lw=width,
                                      capstyle="butt", zorder=z))

    def title(self, y, letter, s):
        self.text(0, y, letter, fontsize=TITLE, fontweight="bold", va="bottom")
        self.text(0.2, y, s, fontsize=TITLE, fontweight="bold", va="bottom")


def panel_a(c, top, A):
    ops, areas, shares = list(A["ops"]), list(A["areas"]), A["shares"]
    big, thumb, gap = 1.0, 0.56, 0.09
    heights = [thumb if LABELS[k][1] else 0.24 for k in ops]
    total = sum(heights) + gap * (len(ops) - 1)
    mid = top - total / 2
    # Input and first stage.
    c.text(0.5, mid + big / 2 + 0.07, "Input", ha="center", va="bottom")
    c.tile(0, mid - big / 2, big, big, A["image"])
    x1 = 1.38
    c.text(x1 + big / 2, mid + big / 2 + 0.36, "Oriented and color energy", ha="center", va="bottom")
    k = 0.22
    for i, kern in enumerate(A["gabors"]):
        v = np.abs(kern).max()
        c.tile(x1 + (big - 4 * k - 3 * 0.04) / 2 + i * (k + 0.04), mid + big / 2 + 0.07, k, k, kern, cmap="gray",
               vmin=-v, vmax=v)
    c.tile(x1, mid - big / 2, big, big, A["energy"], cmap="gray")
    c.arrow(big + 0.06, mid, x1 - 0.06, mid)
    # Texture operations.
    x2, xl, xr, xa = 2.9, 3.56, 5.42, 6.42
    y, rows = top, {}
    for kkey, h in zip(ops, heights):
        yc = y - h / 2
        rows[kkey] = yc
        name, desc = LABELS[kkey]
        if kkey == "M":
            c.tile(x2, y - h, thumb, h, A["boundary"], cmap="gray")
        elif kkey == "V":
            c.tile(x2, y - h, thumb, h, A["cov"], cmap="RdBu_r", vmin=-1, vmax=1)
        elif kkey == "O":
            c.tile(x2, y - h, thumb, h, A["cross"], cmap="RdBu_r", vmin=-1, vmax=1)
        if desc:
            c.text(xl, yc + 0.03, name, va="bottom")
            c.text(xl, yc - 0.03, desc, va="top", fontsize=SMALL, color=MUTED)
            c.arrow(x1 + big + 0.06, mid + (yc - mid) * 0.35, x2 - 0.06, yc)
        else:
            c.text(xl, yc, name, va="center", color=MUTED)
        y -= h + gap
    for i, kern in enumerate(A["s2_kernels"]):
        v = np.abs(kern).max()
        c.tile(xl + i * 0.16, rows["M"] - 0.28, 0.13, 0.13, kern, cmap="gray", vmin=-v, vmax=v)
    # Links to areas: width is the share of the texture gain beyond edges.
    area_y = {a: rows[k] for a, k in zip(areas, ("M", "V", "O"))}
    for j, a in enumerate(areas):
        best = int(np.argmax(shares[:, j]))
        for i, kkey in enumerate(ops):
            top_link = i == best
            c.link(xr, rows[kkey], xa, area_y[a], 10 * shares[i, j], TEXTURE if top_link else LIGHT, 3 if top_link else 2)
        c.text(xa + 0.08, area_y[a] + 0.02, a, va="bottom", fontweight="bold", fontsize=8)
        c.text(xa + 0.08, area_y[a] - 0.03, f"{100 * shares[best, j]:.0f}% of texture gain", va="top",
               fontsize=SMALL, color=TEXTURE)
    # Colorbar for the two correlation matrices.
    c.text((xr + xa) / 2, top + 0.06, "Share of texture gain beyond edges", ha="center", va="bottom", fontsize=SMALL,
           color=MUTED)
    cax = c.axes(x2, rows["J"], thumb, 0.05)
    cax.imshow(np.linspace(-1, 1, 64)[None], cmap="RdBu_r", aspect="auto", extent=(-1, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-1, 0, 1])
    cax.tick_params(labelsize=SMALL, length=1.5, pad=1)
    for spine in cax.spines.values():
        spine.set_visible(False)
    c.text(x2 + thumb / 2, rows["J"] + 0.08, "Correlation", ha="center", va="bottom", fontsize=SMALL, color=MUTED)
    return top - total


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=SMALL, pad=1.5)


def panel_b(c, top, B):
    block, tile = 2.3, 0.56
    specs = [
        ("M", "Texture boundaries", "Envelope orientation (°)", "Modulation energy", ("Envelope 0°", "Envelope 90°")),
        ("V", "Orientation/color co-occurrence", "Patches sharing a location (%)", "Correlation, horizontal\nwith vertical energy", ("0% shared", "100% shared")),
        ("O", "Cross-scale structure", "Phase scrambling (%)", "Correlation, fine\nwith coarse", ("Intact", "Scrambled")),
    ]
    ph, eh = 0.95, 0.22
    bottom = top - 0.3 - ph - 0.1 - eh
    for i, (key, name, xlabel, ylabel, caps) in enumerate(specs):
        x0 = i * (block + 0.3)
        c.text(x0, top - 0.02, name, va="top", color=TEXTURE, fontweight="bold")
        for j in range(2):
            ty = top - 0.3 - (j + 1) * tile - j * 0.18
            c.tile(x0, ty, tile, tile, B[f"{key}_examples"][j])
            c.text(x0 + tile / 2, ty - 0.02, caps[j], ha="center", va="top", fontsize=SMALL, color=MUTED)
        xs, ys, se = B[f"{key}_x"].astype(float), B[f"{key}_y"], B[f"{key}_se"]
        if key != "M":
            xs = xs * 100
        ax = c.axes(x0 + tile + 0.62, bottom + eh + 0.1, block - tile - 0.66, ph)
        if key == "M":
            for ch in range(4):
                ax.plot(xs, ys[:, ch], color=TEXTURE, lw=1.1)
                peak = xs[int(np.argmax(ys[:-1, ch]))]
                ax.text(peak + (6 if ch == 0 else 0), ys[:, ch].max() + 0.004, f"{45 * ch}°", ha="center" if ch else "left", va="bottom",
                        fontsize=SMALL, color=TEXTURE)
            ax.set_ylim(0.13, 0.275)
            ax.set_xticks([0, 45, 90, 135, 180])
        else:
            ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
            ax.fill_between(xs, ys - se, ys + se, color=TEXTURE, alpha=0.25, lw=0)
            ax.plot(xs, ys, color=TEXTURE, lw=1.3)
            ax.set_xticks([0, 50, 100])
            ax.set_ylim((-1, 1) if key == "V" else (0, 0.4))
        ax.set_ylabel(ylabel, fontsize=SMALL, labelpad=2)
        ax.set_xticklabels([])
        style(ax)
        ex = c.axes(x0 + tile + 0.62, bottom, block - tile - 0.66, eh)
        ex.plot(xs, B[f"{key}_energy"], color=NEUTRAL, lw=1.1)
        ex.set_ylim(0, 0.6)
        ex.set_yticks([0, 0.5])
        ex.set_xticks(ax.get_xticks())
        ex.set_xlim(ax.get_xlim())
        ex.set_xlabel(xlabel, fontsize=SMALL, labelpad=1.5)
        ex.set_ylabel("Edge\nenergy", fontsize=SMALL, labelpad=2, color=NEUTRAL)
        style(ex)
    return bottom - 0.3


def panel_c(c, top, C):
    block, gap, sep = 2.3, 0.04, 0.14
    s = (block - 2 * gap - sep) / 4
    for i, area in enumerate(C["areas"]):
        x0 = i * (block + 0.3)
        c.text(x0, top - 0.02, str(area), va="top", fontweight="bold", fontsize=8)
        c.text(x0 + block, top - 0.03, f"r = {C['r'][i]:.2f}", va="top", ha="right", color=MUTED, fontstyle="italic")
        y = top - 0.34 - s
        c.text(x0, y + s + 0.03, "Highest predicted response", va="bottom", fontsize=SMALL, color=MUTED)
        c.text(x0 + block - s, y + s + 0.03, "Optimized", va="bottom", fontsize=SMALL, color=MUTED)
        for j in range(3):
            c.tile(x0 + j * (s + gap), y, s, s, C["top"][i][j])
        c.tile(x0 + block - s, y, s, s, C["optimized"][i])
    return top - 0.34 - s


def main():
    A, B, C = (np.load(DATA / f"{k}.npz") for k in "abc")
    H = 6.75
    c = Canvas(H)
    y = H - 0.2
    c.title(y, "a", "From image to area: texture operations of the searched model")
    y = panel_a(c, y - 0.3, A)
    y -= 0.36
    c.title(y, "b", "Each operation responds to one image statistic")
    y = panel_b(c, y - 0.08, B)
    y -= 0.3
    c.title(y, "c", "Images that drive each area's readout")
    y = panel_c(c, y - 0.08, C)
    print(f"lowest content at {y:.2f} in (figure height {H} in)")
    for ext in ("pdf", "svg", "png"):
        c.fig.savefig(OUT / f"circuit_figure.{ext}", dpi=450 if ext == "png" else 300)
    print("wrote", OUT / "circuit_figure.pdf")


if __name__ == "__main__":
    main()
