"""Illustrative panels for the erjpu architecture diagram.

A minimal re-implementation of each stage, applied to one sample photo
(skimage 'coffee'); it shows what each kind of computation does, not the
model's exact parameters.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import fftconvolve
from skimage import data, transform

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ORANGE, INK, MUTED = '#C2410C', '#1F1F1D', '#6B6B66'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': MUTED})

img = data.coffee()[:, 33:567].astype(float) / 255  # 400 x 534, 4:3
img = transform.resize(img, (300, 400), anti_aliasing=True)
comp = img ** 0.5
L = comp.mean(2)
RG = comp[..., 0] - comp[..., 1]
BY = comp[..., 2] - (comp[..., 0] + comp[..., 1]) / 2


def gabor(theta, wl, phase=0.0, size=None):
    sigma = 0.56 * wl
    size = size or int(np.ceil(3 * sigma)) * 2 + 1
    r = np.arange(size) - size // 2
    x, y = np.meshgrid(r, r)
    xr = x * np.cos(theta) + y * np.sin(theta)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / wl + phase)
    g -= g.mean() * (phase == 0)
    return g / np.abs(g).sum()


def dog(s1, s2, size=31):
    r = np.arange(size) - size // 2
    x, y = np.meshgrid(r, r)
    d = np.exp(-(x ** 2 + y ** 2) / (2 * s1 ** 2)) / s1 ** 2 - np.exp(-(x ** 2 + y ** 2) / (2 * s2 ** 2)) / s2 ** 2
    return d / np.abs(d).max()


THETAS = np.arange(8) * np.pi / 8
def conv(a, k):
    p = k.shape[0] // 2
    return fftconvolve(np.pad(a, p, mode='reflect'), k, mode='same')[p:-p, p:-p]


def energy(a, theta, wl):
    e, o = conv(a, gabor(theta, wl)), conv(a, gabor(theta, wl, np.pi / 2))
    return e, o, np.sqrt(e ** 2 + o ** 2)


E = np.stack([energy(L, t, 8)[2] for t in THETAS])
pool = gaussian_filter(E.sum(0), 12)
N = E / (0.02 + pool)


def new(w=4, h=3):
    return plt.figure(figsize=(w, h), dpi=152)


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=152, facecolor='white')
    plt.close(fig)


def show(ax, a, cmap='gray', vmin=None, vmax=None, title=None):
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=4)


def pct(a, lo=1, hi=99):
    return np.percentile(a, lo), np.percentile(a, hi)


def sym(a):
    m = np.percentile(np.abs(a), 99)
    return -m, m


# 1 input
fig = new(); ax = fig.add_axes([0, 0, 1, 1]); show(ax, img); save(fig, 'p01_input')

# 2 compression curve
fig = new(); ax = fig.add_axes([.16, .17, .78, .76])
x = np.linspace(0, 1, 200)
ax.plot(x, x, color=MUTED, lw=1.5, ls='--', label='linear')
ax.plot(x, x ** 0.5, color=ORANGE, lw=3.5, label='compressed')
ax.set_xlabel('pixel intensity'); ax.set_ylabel('output')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.text(.55, .45, 'intensity$^{\\gamma}$, $\\gamma<1$', color=ORANGE, fontsize=13, fontweight='bold')
save(fig, 'p02_compression')

# 3 opponent channels
rg_cmap = LinearSegmentedColormap.from_list('rg', ['#1a9850', '#f7f7f7', '#d73027'])
by_cmap = LinearSegmentedColormap.from_list('by', ['#d4a017', '#f7f7f7', '#2c5fbf'])
fig = new(4, 3)
for i, (a, cm, t) in enumerate([(L, 'gray', 'L (luminance)'), (RG, rg_cmap, 'red − green'), (BY, by_cmap, 'blue − yellow')]):
    ax = fig.add_axes([.02, .69 - i * .33, .62, .29])
    show(ax, a, cm, *(pct(a) if i == 0 else sym(a)))
    fig.text(.67, .835 - i * .33, t, fontsize=12, color=INK, va='center')
save(fig, 'p03_opponent')

# 4 filter bank
fig = new(4, 3)
def tile(x, y, k, cm, w=.105):
    ax = fig.add_axes([x, y, w, w * 4 / 3]); show(ax, k, cm, *sym(k))
for j, t in enumerate(THETAS):
    for i, wl in enumerate((5, 9)):
        tile(.03 + j * .118, .78 - i * .17, gabor(t, wl, size=31), 'gray')
fig.text(.5, .955, 'luminance Gabors: 8 orientations × 2 scales', ha='center', fontsize=10, color=INK)
for j, (k, cm) in enumerate([(gabor(np.pi / 4, 9, np.pi / 2, 31), 'gray'), (dog(2.5, 6), 'gray'), (-dog(2.5, 6), 'gray')]):
    tile(.03 + j * .118, .32, k, cm)
fig.text(.035 + 1.5 * .118, .27, 'odd, on and off blobs', ha='center', fontsize=8.5, color=INK, va='top')
for j, t in enumerate(THETAS[::2]):
    tile(.42 + j * .118, .40, gabor(t, 9, size=31), rg_cmap)
    tile(.42 + j * .118, .14, gabor(t, 9, size=31), by_cmap)
fig.text(.42 + 1.5 * .118 + .05, .355, 'red–green Gabors', ha='center', fontsize=8.5, color=INK, va='center')
fig.text(.42 + 1.5 * .118 + .05, .095, 'blue–yellow Gabors', ha='center', fontsize=8.5, color=INK, va='center')
for j, cm in enumerate((rg_cmap, by_cmap)):
    tile(.03 + j * .118, .06, dog(2.5, 6), cm)
fig.text(.03 + .5 * .118 + .05, .03, 'color blobs', ha='center', fontsize=8.5, color=INK, va='center')
save(fig, 'p04_filters')

# 5 energy
e, o, en = energy(L, THETAS[2], 8)
fig = new(4, 3)
for i, (a, t, cm) in enumerate([(e, 'even', 'gray'), (o, 'odd', 'gray')]):
    ax = fig.add_axes([.02 + i * .49, .53, .47, .4]); show(ax, a, cm, *sym(a), title=t)
ax = fig.add_axes([.255, .02, .47, .4]); show(ax, en, 'magma', *pct(en))
fig.text(.5, .455, '√(even² + odd²)', ha='center', fontsize=13, color=ORANGE, fontweight='bold')
save(fig, 'p05_energy')

# 6 divisive normalization
fig = new(4, 3)
before, after = E.sum(0), N.sum(0)
ax = fig.add_axes([.02, .2, .47, .6]); show(ax, before, 'magma', *pct(before), title='energy')
ax = fig.add_axes([.51, .2, .47, .6]); show(ax, after, 'magma', *pct(after), title='÷ local energy')
fig.text(.5, .07, 'weak edges in dark regions are boosted', ha='center', fontsize=10, color=MUTED)
save(fig, 'p06_norm')

# 7 V1 maps stack
fig = new(4, 3)
for i, t in enumerate([0, 2, 4, 6]):
    ax = fig.add_axes([.06 + i * .12, .44 - i * .13, .52, .52]); show(ax, N[t], 'magma', *pct(N[t]))
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('white'); s.set_linewidth(1.5)
save(fig, 'p07_v1maps')

# 8 foveated grid pooling
fig = new(4, 3)
m = N.sum(0)
ax = fig.add_axes([.02, .1, .47, .8]); show(ax, m, 'magma', *pct(m))
u = np.linspace(-1, 1, 10)
edges = np.sign(u) * np.abs(u) ** 1.7
xs, ys = (edges + 1) / 2 * 399, (edges + 1) / 2 * 299
for x in xs:
    ax.axvline(x, color='white', lw=.8)
for y in ys:
    ax.axhline(y, color='white', lw=.8)
cells = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        y0, y1, x0, x1 = int(ys[i]), max(int(ys[i + 1]), int(ys[i]) + 1), int(xs[j]), max(int(xs[j + 1]), int(xs[j]) + 1)
        cells[i, j] = m[y0:y1, x0:x1].mean()
ax2 = fig.add_axes([.51, .1, .47, .8])
ax2.pcolormesh(xs, ys[::-1], cells[::-1], cmap='magma', vmin=pct(m)[0], vmax=pct(m)[1])
ax2.set_aspect('equal'); ax2.set_xticks([]); ax2.set_yticks([])
for s in ax2.spines.values():
    s.set_visible(False)
ax.set_title('finer cells at the centre', fontsize=10, pad=3)
ax2.set_title('mean per cell', fontsize=10, pad=3)
save(fig, 'p08_grid')

# 9 texture boundaries: filter, rectify, filter
coarse = zoom(N[2], .5)
_, _, b = energy(coarse, THETAS[2], 18)
fig = new(4, 3)
ax = fig.add_axes([.02, .2, .47, .6]); show(ax, coarse, 'magma', *pct(coarse), title='V1 energy')
ax = fig.add_axes([.51, .2, .47, .6]); show(ax, b, 'magma', *pct(b), title='2nd Gabor energy')
fig.text(.5, .07, 'responds where texture changes', ha='center', fontsize=10, color=MUTED)
save(fig, 'p09_texture')

# 10 orientation field and curvature
vx = (N * np.cos(2 * THETAS)[:, None, None]).sum(0)
vy = (N * np.sin(2 * THETAS)[:, None, None]).sum(0)
vx, vy = gaussian_filter(vx, 3), gaussian_filter(vy, 3)
mag = np.hypot(vx, vy)
ang = (np.arctan2(vy, vx) + np.pi) / (2 * np.pi)
import matplotlib.colors as mc
rgb = mc.hsv_to_rgb(np.dstack([ang, np.ones_like(ang) * .85, np.clip(mag / np.percentile(mag, 98), 0, 1)]))
gyx, gxx = np.gradient(vx); gyy, gxy = np.gradient(vy)
curv = np.hypot(gxx + gyy, gyx - gxy) * (mag > np.percentile(mag, 40))
fig = new(4, 3)
ax = fig.add_axes([.02, .2, .47, .6]); show(ax, rgb, title='orientation (hue)')
ax = fig.add_axes([.51, .2, .47, .6]); show(ax, curv, 'magma', *pct(curv, 2, 98), title='curvature')
fig.text(.5, .07, 'where contours bend or meet', ha='center', fontsize=10, color=MUTED)
save(fig, 'p10_orientation')

# 11 edge phase: 1D profiles
xx = np.linspace(-1, 1, 300)
profiles = [('bright line', np.exp(-xx ** 2 / .004)), ('dark crease', -np.exp(-xx ** 2 / .004)), ('step edge', np.tanh(xx / .03))]
k = np.exp(-xx ** 2 / .02)
ke, ko = k * np.cos(2 * np.pi * xx / .5), k * np.sin(2 * np.pi * xx / .5)
ke -= ke.mean()
fig = new(4, 3)
for i, (t, p) in enumerate(profiles):
    ax = fig.add_axes([.04 + i * .325, .52, .29, .34])
    ax.plot(xx, p, color=INK, lw=2); ax.set_title(t, fontsize=10, pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-1.3, 1.3)
    for s in ax.spines.values():
        s.set_visible(False)
    ev, od = np.convolve(p, ke, 'same'), np.convolve(p, ko, 'same')
    ax = fig.add_axes([.04 + i * .325, .1, .29, .34])
    sc = max(np.abs(ev).max(), np.abs(od).max())
    ax.bar([0, 1], [ev[150] / sc, np.abs(od).max() / sc * np.sign(od[np.abs(od).argmax()])], color=[ORANGE, MUTED], width=.6)
    ax.axhline(0, color=MUTED, lw=.8); ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['even', 'odd'], fontsize=9); ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
save(fig, 'p11_phase')

# 12 surface geometry: five shapes + shape-index tuning
r = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(r, r)
shapes = [('cup', X ** 2 + Y ** 2), ('rut', X ** 2), ('saddle', X ** 2 - Y ** 2), ('ridge', -X ** 2), ('dome', -(X ** 2 + Y ** 2))]
fig = new(4, 3)
for i, (t, z) in enumerate(shapes):
    ax = fig.add_axes([.0 + i * .2, .54, .2, .44], projection='3d')
    m_ = X ** 2 + Y ** 2 <= 1.05
    zz = np.where(m_, z, np.nan)
    ax.plot_surface(X, Y, zz, rstride=4, cstride=4, cmap='Greys_r', vmin=-1.6, vmax=1.4, linewidth=.25, edgecolor='#555', antialiased=True)
    ax.view_init(elev=28, azim=-60); ax.set_box_aspect((1, 1, .8), zoom=1.3); ax.set_axis_off(); ax.set_zlim(-1.1, 1.1)
    fig.text(.1 + i * .2, .5, t, ha='center', fontsize=11, color=INK)
ax = fig.add_axes([.06, .14, .9, .27])
si = np.linspace(-1, 1, 300)
for c in (-1, -.5, 0, .5, 1):
    ax.plot(si, np.exp(-(si - c) ** 2 / (2 * .15 ** 2)), color=ORANGE, lw=2.2)
ax.set_yticks([]); ax.set_xticks([-1, 0, 1]); ax.set_xlim(-1.08, 1.08)
ax.set_xlabel('shape index', fontsize=10, labelpad=1)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
save(fig, 'p12_geometry')

# 13 co-occurrence
a, bb = N[0], gaussian_filter(np.abs(conv(RG, gabor(0, 8))), 2)
fig = new(4, 3)
for i, (m_, t) in enumerate([(a, 'edge energy'), (bb, 'color edge'), (a * bb, 'product')]):
    ax = fig.add_axes([.02 + i * .335, .3, .3, .45]); show(ax, m_, 'magma', *pct(m_), title=t)
fig.text(.335, .52, '×', fontsize=20, ha='center', va='center', color=ORANGE, fontweight='bold')
fig.text(.67, .52, '=', fontsize=20, ha='center', va='center', color=INK)
fig.text(.5, .12, 'then average over space', ha='center', fontsize=10, color=MUTED)
save(fig, 'p13_cooccur')

# 14 global layout
small = transform.resize(L, (24, 32), anti_aliasing=True)
fig = new(4, 3)
ax = fig.add_axes([.02, .2, .47, .6]); show(ax, small, 'gray', title='32-pixel image')
ax.axvline(15.5, color=ORANGE, lw=2, ls='--')
ax.add_patch(plt.Circle((15.5, 11.5), 7, fill=False, color='#2c5fbf', lw=2))
diff = np.abs(small - small[:, ::-1])
ax = fig.add_axes([.51, .2, .47, .6]); show(ax, diff, 'magma', title='|left − mirror|')
fig.text(.5, .07, 'symmetry, centre versus periphery', ha='center', fontsize=10, color=MUTED)
save(fig, 'p14_global')

# 15 raw color 6x6
mos = transform.resize(comp, (6, 8), anti_aliasing=True, order=0)
cells = np.stack([[comp[i * 50:(i + 1) * 50, j * 67:(j + 1) * 67].reshape(-1, 3).mean(0) for j in range(6)] for i in range(6)])
fig = new(4, 3)
ax = fig.add_axes([.02, .15, .47, .7]); show(ax, img, title='image')
ax = fig.add_axes([.51, .15, .47, .7]); ax.imshow(cells, interpolation='nearest', aspect=.75)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title('6×6 mean color', fontsize=11, pad=4)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, 'p15_rawcolor')
print('ok', sorted(p.name for p in OUT.glob('*.png')))
