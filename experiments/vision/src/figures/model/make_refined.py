"""Render a compact Figure 4 from existing cached data only.

Usage: .venv/bin/python src/figures/model/make_refined.py [--iteration 2|3]
No model evaluation or manuscript modification is performed.
"""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from make_figure import CACHE, ATLAS, FAMILY, COLUMNS, INK, MUTED, inches, tile

OUT = CACHE.parent / 'refined'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iteration', type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    data = np.load(CACHE)
    atlas = json.loads(ATLAS.read_text())['features']
    # Shared examples are valid only if both the selected feature and all images match.
    assert data['best'][1] == data['best'][2]
    assert np.array_equal(data['top_images'][1], data['top_images'][2])
    W, H = 165 / 25.4, 4.65
    fig = plt.figure(figsize=(W, H))
    plt.rcParams.update({'font.size': 8})

    def text(x, y, label, **kwargs):
        fig.text(x / W, y / H, label, **kwargs)

    def rule(x, y, w, color):
        fig.add_artist(plt.Line2D([x / W, (x + w) / W], [y / H, y / H],
                                 transform=fig.transFigure, color=color, lw=1.8,
                                 solid_capstyle='butt'))

    text(0, H - .18, 'a', fontsize=10, fontweight='bold')
    text(.22, H - .18, 'Model responses', fontsize=9, fontweight='bold')
    gap = .045
    size = (W - 7 * gap) / 8
    top = H - .62 - size
    headers = [('Input', None, None)] + COLUMNS
    for c, (label, family, key) in enumerate(headers):
        x = c * (size + gap)
        text(x + size / 2, top + size + .055, label, ha='center', va='bottom',
             fontsize=7.5, linespacing=1.05)
        if family:
            rule(x, top + size + .025, size, FAMILY[family][1])
        if key:
            lo, hi = np.percentile(data[f'map_{key}'], [1, 99])
        for row in range(3):
            ax = inches(fig, x, top - row * (size + gap), size, size)
            if key:
                tile(ax, data[f'map_{key}'][row], cmap='gray', vmin=lo, vmax=hi)
            else:
                tile(ax, data['images'][row])

    y = top - 2 * (size + gap) - .30
    text(0, y, 'b', fontsize=10, fontweight='bold')
    text(.22, y, 'Features linked to visual areas', fontsize=9, fontweight='bold')
    group_gap = .20
    group_w = (W - 2 * group_gap) / 3
    image_gap = .035
    thumb = (group_w - 2 * image_gap) / 3
    groups = [([0], 'Mixed edge directions'), ([1, 2], 'Corner strength'), ([3], 'Horizontal / vertical lines')]
    rois = ['V1', 'hV4', 'FFA', 'PPA']
    for g, (indices, label) in enumerate(groups):
        x = g * (group_w + group_gap)
        k = indices[0]
        family = atlas[int(data['best'][k])][2]
        text(x, y - .24, ' / '.join(rois[i] for i in indices),
             fontsize=8, fontweight='bold', va='center')
        rs = ' / '.join(f'{data["signed"][i]:.2f}'.replace('-', '−') for i in indices)
        if args.iteration == 2:
            text(x + group_w, y - .24, '$r$ = ' + rs, fontsize=7.5,
                 color=MUTED, va='center', ha='right')
        text(x, y - .44, label, fontsize=8, va='center')
        rule(x, y - .56, group_w, FAMILY[family][1])
        for j in range(3):
            tile(inches(fig, x + j * (thumb + image_gap), y - .60 - thumb, thumb, thumb),
                 data['top_images'][k, j])

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'figure4_v{args.iteration}.{ext}', facecolor='white', dpi=400)
    plt.close(fig)
    print(OUT / f'figure4_v{args.iteration}.png')


if __name__ == '__main__':
    main()
