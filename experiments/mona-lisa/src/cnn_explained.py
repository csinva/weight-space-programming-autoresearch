"""A two-part explainer figure: (1) how the network is built, (2) how all its weights are
assembled into the picture.  usage: python src/cnn_explained.py [cnn_small_weights.npz]"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
sys.path.insert(0, os.path.dirname(__file__))
from mnist import load

root = os.path.join(os.path.dirname(__file__), '..')
path = sys.argv[1] if len(sys.argv) > 1 else 'cnn_small_weights.npz'
if not os.path.isabs(path): path = os.path.join(root, path)
Z = np.load(path); meta = json.load(open(path.replace('.npz', '.json')))
order = meta['order']; C, R = meta['cols'], meta['rows']; TH, TW = meta['tile']; T = TH*TW
PER = meta['per_class']; NP = 10*PER; n1, k1 = meta['n_layer1'], meta['k1']
small_cfg = meta.get('config') == 'small'
ACC = '96.17%' if small_cfg else '97.48%'
flat = np.concatenate([Z[nm].ravel() for nm in order]); lo, hi = flat.min(), flat.max()
pic = flat.reshape(R, C, TH, TW).transpose(0, 2, 1, 3).reshape(R*TH, C*TW); H, W = pic.shape
sizes = [Z[nm].size for nm in order]; starts = np.cumsum([0] + sizes[:-1])
first_tile = lambda L: starts[L]//T; last_tile = lambda L: (starts[L] + sizes[L] - 1)//T
gray = lambda a: np.clip((a - lo)/(hi - lo), 0, 1)
tile = lambda t: flat[t*T:(t+1)*T].reshape(TH, TW)
t3 = first_tile(2)
COL = {1: '#d1495b', 2: '#e07a1f', 3: '#3366cc', 4: '#2e9d57'}
def layer_of_tile(t):
    L = int(np.searchsorted(starts, t*T, side='right')) - 1
    return 1 if L == 0 else 2 if L == 1 else 3 if L == 2 else 4

TS = float(os.environ.get('TEXT_SCALE', 1.4))           # text size multiplier (layout unchanged)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(20, 15.5), dpi=120, facecolor='white')
def ax_at(r, **kw):
    a = fig.add_axes(r, **kw); a.set_xticks([]); a.set_yticks([]); return a
def arrow(p0, p1, col='#555', lw=1.8, rad=0.0, style='-|>', ms=16, txt=None, txy=None, ha='center'):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style, mutation_scale=ms,
                                   color=col, lw=lw, connectionstyle=f'arc3,rad={rad}'))
    if txt: fig.text(*txy, txt, ha=ha, va='center', fontsize=TS*10.5, color='#333')
def frame(a, col, lw=2.5):
    for s in a.spines.values(): s.set_edgecolor(col); s.set_linewidth(lw)

fig.text(0.5, 0.962, f'weights = Mona Lisa     {ACC} MNIST test accuracy     no training', ha='center', fontsize=TS*21, weight='bold')

# ====================================================================== PART 1
fig.text(0.02, 0.935, '1   Build', fontsize=TS*17, weight='bold', color='#222')
fig.add_artist(plt.Line2D([0.02, 0.98], [0.925, 0.925], color='#ccc', lw=1, transform=fig.transFigure))

# layer 1
for k in (3, 2, 1):
    b_ = ax_at([0.03 + 0.007*k, 0.66 + 0.009*k, 0.12, 0.155]); b_.imshow(gray(Z['conv1'][k, 0]), cmap='Greys_r', vmin=0, vmax=1)
    frame(b_, '#bbb', 1.5)
a1 = ax_at([0.03, 0.66, 0.12, 0.155])
a1.imshow(Z['conv1'][0, 0], cmap='Greys_r')
for (i, j), v in np.ndenumerate(Z['conv1'][0, 0]):
    a1.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9, color='#d1495b')
frame(a1, COL[1])
fig.text(0.09, 0.885, 'Layer 1', ha='center', fontsize=TS*14, weight='bold', color=COL[1])
fig.text(0.09, 0.862, f'{n1} filters (1 used)', ha='center', fontsize=TS*12, color='#333')

# layer 2
a2 = ax_at([0.205, 0.715, 0.09, 0.045])
a2.imshow(np.array([[1.0, 0.0]]), cmap='Greys_r', vmin=-0.3, vmax=1.3)
a2.text(0, 0, '+1', ha='center', va='center', fontsize=TS*15, weight='bold', color='#222')
a2.text(1, 0, '-1', ha='center', va='center', fontsize=TS*15, weight='bold', color='white')
frame(a2, COL[2])
fig.text(0.25, 0.885, 'Layer 2', ha='center', fontsize=TS*14, weight='bold', color=COL[2])
fig.text(0.25, 0.862, '1 filter', ha='center', fontsize=TS*12, color='#333')
arrow((0.176, 0.74), (0.198, 0.74))

# layer 3: one memory filter, its weights first, how they are made second
Xtr, ytr, _, _ = load()
idx = np.concatenate([np.where(ytr == c)[0][:PER] for c in range(10)])
fi = 3*PER + 9                                                  # an example filter (class 3)
f = Z['conv3_memory'][fi, 0]
P = np.repeat(0.5*(f[:, 0::2] + f[:, 1::2]), 2, axis=1)         # the painting part (equal pairs)
rip = f - P                                                     # the ripple: +e*c, -e*c on each pair
c = 0.5*(f[:, 0::2] - f[:, 1::2])
fig.text(0.47, 0.885, 'Layer 3', ha='center', fontsize=TS*14, weight='bold', color=COL[3])
fig.text(0.47, 0.862, f'{NP:,} filters', ha='center', fontsize=TS*12, color='#333')
bw = 0.20; bh = bw*(20/15.5)*TH/TW
for k in (3, 2, 1):
    b_ = ax_at([0.37 + 0.007*k, 0.672 + 0.009*k, bw, bh]); b_.imshow(gray(Z['conv3_memory'][fi + 97*k, 0]), cmap='Greys_r', vmin=0, vmax=1)
    frame(b_, '#9db4e0', 1.5)
aw = ax_at([0.37, 0.672, bw, bh]); aw.imshow(gray(f), cmap='Greys_r', vmin=0, vmax=1); frame(aw, COL[3], 4)
fig.text(0.47, 0.664, 'weights', ha='center', va='top', fontsize=TS*12, color=COL[3], weight='bold')
arrow((0.302, 0.74), (0.365, 0.74))

# layers 4-16: an example weight first, the halving schematic second
fig.text(0.775, 0.885, 'Layers 4-16', ha='center', fontsize=TS*14, weight='bold', color=COL[4])
fig.text(0.775, 0.862, '3 filters each', ha='center', fontsize=TS*12, color='#333')
w4 = Z['conv4_max'][:3, 0, 0]                                    # the 3 live 1x2 kernels of layer 4
kx, ky, kw_, kh_ = 0.735, 0.672, 0.075, 0.13
ak = ax_at([kx, ky, kw_, kh_])
ak.imshow(gray(w4), cmap='Greys_r', vmin=0, vmax=1)
for (i, j), v in np.ndenumerate(w4):
    ak.text(j, i, f'{v:+.0f}', ha='center', va='center', fontsize=TS*17, weight='bold', color='#111' if gray(v) > 0.5 else 'white')
frame(ak, COL[4], 3)
for j, lab in enumerate(['a', 'b']):
    fig.text(kx + kw_*(j + 0.5)/2, ky + kh_ + 0.008, lab, ha='center', va='bottom', fontsize=TS*14, weight='bold', color='#333')
for i, lab in enumerate(['f1', 'f2', 'f3']):
    fig.text(kx - 0.008, ky + kh_*(2.5 - i)/3, lab, ha='right', va='center', fontsize=TS*12, color='#777')
for i, lab in enumerate(['a+b', 'a-b', 'b-a']):
    fig.text(kx + kw_ + 0.008, ky + kh_*(2.5 - i)/3, lab, va='center', fontsize=TS*13, color='#333')
fig.text(kx + kw_/2, 0.664, 'weights', ha='center', va='top', fontsize=TS*12, color=COL[4], weight='bold')
arrow((0.597, 0.74), (0.705, 0.74))

# what each part computes: one row of boxes, each under its layer
from matplotlib.patches import FancyBboxPatch
def role(xc, w, col, head, sub):
    y0_, h_ = 0.574, 0.058
    fig.add_artist(FancyBboxPatch((xc - w/2, y0_), w, h_, boxstyle='round,pad=0.004,rounding_size=0.008',
                                  transform=fig.transFigure, fc=col, ec=col, alpha=0.13, lw=0))
    fig.add_artist(FancyBboxPatch((xc - w/2, y0_), w, h_, boxstyle='round,pad=0.004,rounding_size=0.008',
                                  transform=fig.transFigure, fc='none', ec=col, lw=2))
    fig.text(xc, y0_ + 0.038, head, ha='center', va='center', fontsize=TS*14, weight='bold', color=col)
    fig.text(xc, y0_ + 0.016, sub, ha='center', va='center', fontsize=TS*10.5, color='#333')
GOLD = '#b8860b'
role(0.09, 0.13, COL[1], 'blur + shrink', '6x6 conv, stride 2')
role(0.25, 0.14, COL[2], 'pairs (1+y, 1-y)', '1x2 transposed conv')
role(0.47, 0.245, COL[3], 'similarity', '14x28 conv, 1 filter per stored digit')
role(0.765, 0.175, COL[4], 'best per class', '1x2 convs, stride 2: max(a, b)')
role(0.932, 0.115, GOLD, 'argmax', 'nearest digit\'s class')
for x0_, x1_ in [(0.158, 0.177), (0.323, 0.345), (0.595, 0.672), (0.857, 0.870)]:
    arrow((x0_, 0.603), (x1_, 0.603), col='#555', lw=1.6, ms=14)

# ====================================================================== PART 2 (visual, few words)
fig.text(0.02, 0.530, '2   Assemble', fontsize=TS*17, weight='bold', color='#222')
fig.add_artist(plt.Line2D([0.02, 0.98], [0.52, 0.52], color='#ccc', lw=1, transform=fig.transFigure))
BIG = dict(arrowstyle='simple,head_width=1.6,head_length=1.4,tail_width=0.55', color='#bbb', lw=0)
def big_arrow(x0, x1, y):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, mutation_scale=22, **BIG))
ph = 0.46; pw = ph*(15.5/20)*(W/H); yb = 0.03

# --- A: the layers, as objects
axA = fig.add_axes([0.02, yb, 0.17, ph]); axA.axis('off'); axA.set_xlim(0, 100); axA.set_ylim(250, 0)
def put(a, img, x, y, w, col, lw=2.5, cmap='Greys_r'):
    h = w*img.shape[0]/img.shape[1]
    a.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=(x, x + w, y + h, y), interpolation='nearest', zorder=3)
    a.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=lw, zorder=4)); return h
g1 = gray(Z['conv1'][:, 0].reshape(14, 14, k1, k1).transpose(0, 2, 1, 3).reshape(14*k1, 14*k1)) if small_cfg else \
     gray(Z['conv1'][:784, 0].reshape(28, 28, k1, k1).transpose(0, 2, 1, 3).reshape(28*k1, 28*k1))
put(axA, g1, 30, 2, 40, COL[1])
put(axA, gray(Z['convT2_pairs'].ravel()[:TH*TW].reshape(TH, TW)), 22, 52, 56, COL[2])
deck = [t3 + 1500*k + 37 for k in range(6)]
for k, t in enumerate(deck):
    put(axA, gray(tile(t)), 10 + 5*k, 92 + 9*k, 60, COL[3], lw=2)
axA.text(48, 186, 'x 10,240' if small_cfg else 'x 40,960', fontsize=TS*13, color=COL[3], weight='bold', ha='center', va='center')
for k in range(3):
    put(axA, gray(tile(first_tile(3) + k)), 40 + 21*k, 212, 18, COL[4], lw=2)
for txt, y, col in [('1', 22, COL[1]), ('2', 60, COL[2]), ('3', 120, COL[3]), ('4-16', 218, COL[4])]:
    axA.text(0, y, txt, fontsize=TS*16, weight='bold', color=col, va='center')
big_arrow(0.195, 0.245, yb + ph/2)

# --- B: the picture mid-assembly, filling left to right, top to bottom
xB = 0.255
axB = ax_at([xB, yb, pw, ph])
rows_done = 64; frontier = 23                                    # tiles placed in the current row
part = np.full_like(pic, np.nan)
part[:rows_done*TH] = pic[:rows_done*TH]
part[rows_done*TH:(rows_done+1)*TH, :frontier*TW] = pic[rows_done*TH:(rows_done+1)*TH, :frontier*TW]
canvas = np.where(np.isnan(part), 0.93, gray(np.nan_to_num(part)))
axB.imshow(canvas, cmap='Greys_r', vmin=0, vmax=1, interpolation='lanczos')
for r_ in range(rows_done, R + 1, 4): axB.axhline(r_*TH - 0.5, color='#d5d5d5', lw=0.6)
for c_ in range(0, C + 1, 4): axB.plot([c_*TW - 0.5]*2, [rows_done*TH, H], color='#d5d5d5', lw=0.6)
for s_ in axB.spines.values(): s_.set_edgecolor('#999')
# tiles flying in at the frontier
for k in range(4):
    t = rows_done*C + frontier + k
    x_, y_ = (frontier + 2)*TW + 150*k, rows_done*TH + 40 + 70*k*k
    axB.imshow(gray(tile(t)), cmap='Greys_r', vmin=0, vmax=1, extent=(x_, x_ + 5*TW, y_ + 5*TH, y_), zorder=5)
    axB.add_patch(Rectangle((x_, y_), 5*TW, 5*TH, fill=False, ec=COL[3], lw=2, zorder=6))
axB.set_xlim(-0.5, W - 0.5); axB.set_ylim(H - 0.5, -0.5)
# snake arrows showing the order
for r_ in [8, 32]:
    y_ = r_*TH*1.0
    axB.add_patch(FancyArrowPatch((80, y_), (W - 80, y_), arrowstyle='-|>', mutation_scale=18, color='#f2b134', lw=2.4, zorder=7))
    axB.add_patch(FancyArrowPatch((W - 80, y_ + 30), (80, y_ + 16*TH*0.95), arrowstyle='-|>', mutation_scale=14,
                                  color='#f2b134', lw=1.4, ls=(0, (4, 3)), zorder=7))
# magnified top-left corner: the first tiles, coloured by layer
axZ = ax_at([xB + 0.012, yb + 0.03, pw - 0.024, 0.06])
nz = 24; zc = np.ones((2*TH + 3, 12*(TW + 3)))
for k in range(nz):
    r_, c_ = divmod(k, 12)
    zc[r_*(TH + 3):r_*(TH + 3)+TH, c_*(TW + 3):c_*(TW + 3)+TW] = gray(tile(k))
axZ.imshow(zc, cmap='Greys_r', vmin=0, vmax=1)
for k in range(nz):
    r_, c_ = divmod(k, 12)
    axZ.add_patch(Rectangle((c_*(TW + 3) - 0.5, r_*(TH + 3) - 0.5), TW, TH, fill=False, ec=COL[layer_of_tile(k)], lw=1.8))
for s_ in axZ.spines.values(): s_.set_edgecolor('#666'); s_.set_linewidth(1.2)
axB.add_patch(Rectangle((-0.5, -0.5), 12*TW, 2*TH, fill=False, ec='#666', lw=1.6, zorder=8))
fig.add_artist(ConnectionPatch((12*TW, 2*TH), (zc.shape[1] - 1, 0), 'data', 'data', axesA=axB, axesB=axZ, arrowstyle='-|>', mutation_scale=16, color='#666', lw=1.4, ls=(0, (4, 3))))
big_arrow(xB + pw + 0.008, xB + pw + 0.058, yb + ph/2)

# --- C: the finished picture, with arrows from the painting to the weight tiles
xC = xB + pw + 0.068
axP = ax_at([xC, yb, pw, ph])
axP.imshow(gray(pic), cmap='Greys_r', vmin=0, vmax=1, interpolation='lanczos')
for s_ in axP.spines.values(): s_.set_edgecolor('#888')
xa = (last_tile(0) + 1) % C*TW
axP.add_patch(Rectangle((0, 0), xa, TH*3, fill=False, ec=COL[1], lw=3))
axP.add_patch(Rectangle((xa, 0), TW, TH*3, fill=False, ec=COL[2], lw=3))
tt = first_tile(3); ty = (tt//C)*TH; tx = (tt % C)*TW
axP.add_patch(Rectangle((tx, ty - TH), W - tx, H - ty + TH, fill=False, ec=COL[4], lw=3))
picks = [(int(0.205*R), int(0.40*C)), (int(0.33*R), int(0.86*C)), (int(0.86*R), int(0.42*C))]
xE = xC + pw + 0.035
for k, (r_, c_) in enumerate(picks):
    t = r_*C + c_; ya = yb + ph - 0.11 - k*0.16
    axE = ax_at([xE, ya, 0.13, 0.065]); axE.imshow(gray(tile(t)), cmap='Greys_r', vmin=0, vmax=1); frame(axE, COL[3], 3)
    axP.add_patch(Rectangle((c_*TW - 25, r_*TH - 25), TW + 50, TH + 50, fill=False, ec=COL[3], lw=2))
    fig.add_artist(ConnectionPatch(((c_ + 1)*TW + 25, r_*TH + TH/2), (0, TH/2), 'data', 'data', axesA=axP, axesB=axE,
                                   arrowstyle='-|>', mutation_scale=18, color=COL[3], lw=2))
out = os.path.join(root, 'figures', ('cnn_small_' if small_cfg else 'cnn_') + 'explained.png')
plt.savefig(out, dpi=120, facecolor='white'); print('saved', out)
