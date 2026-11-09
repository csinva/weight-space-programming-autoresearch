"""One annotated figure: the whole weight picture, with every layer's part pulled out and labelled.
usage: python src/cnn_overview.py [cnn_weights.npz | cnn_small_weights.npz]"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from PIL import Image

root = os.path.join(os.path.dirname(__file__), '..')
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, 'cnn_weights.npz')
if not os.path.isabs(path): path = os.path.join(root, path)
Z = np.load(path); meta = json.load(open(path.replace('.npz', '.json')))
small_cfg = meta.get('config') == 'small'
ACC = {'big': '97.48%', 'small': '96.17%'}[meta.get('config', 'big')]
order = meta['order']; C, R = meta['cols'], meta['rows']; TH, TW = meta['tile']; T = TH*TW
n, k1, PER = meta['n_layer1'], meta['k1'], meta['per_class']; NP = 10*PER
flat = np.concatenate([Z[nm].ravel() for nm in order])
lo, hi = flat.min(), flat.max()
pic = flat.reshape(R, C, TH, TW).transpose(0, 2, 1, 3).reshape(R*TH, C*TW)
H, W = pic.shape
sizes = [Z[nm].size for nm in order]; starts = np.cumsum([0] + sizes[:-1])
tiles_of = lambda L: (starts[L]//T, (starts[L] + sizes[L] - 1)//T)
def tile_xy(t): return (t % C)*TW, (t//C)*TH
def gray(a): return np.clip((a - lo)/(hi - lo), 0, 1)
u = H/9576                                                     # annotation scale (1 for the big picture)

cols = {1: '#e6194b', 2: '#f58231', 3: '#4363d8', 'tree': '#3cb44b', 15: '#911eb4', 16: '#f032e6'}
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(19, 13), dpi=130, facecolor='white')
gs = fig.add_gridspec(6, 5, width_ratios=[1, 1, 3.3, 1, 1], hspace=0.75, wspace=0.12,
                      left=0.015, right=0.985, top=0.93, bottom=0.02)
fig.suptitle(f'All {flat.size:,} weights of the 16-layer convnet, in layer order, as {TH}x{TW} tiles '
             f'({ACC} MNIST test accuracy, no training)', fontsize=14, weight='bold')

# ------------------------------------------------------------------ centre: the whole picture
axM = fig.add_subplot(gs[:, 2])
f = 6 if W > 3000 else 1
shown = np.asarray(Image.fromarray((gray(pic)*255).astype(np.uint8)).resize((W//f, H//f), Image.LANCZOS))
axM.imshow(shown, cmap='gray', vmin=0, vmax=255, extent=(0, W, H, 0))
axM.set_xticks([]); axM.set_yticks([])
axM.set_title(f'the whole picture: {C} x {R} tiles = {W} x {H} weights', fontsize=10)
hb = 90*u
a1, b1 = tiles_of(0); a2, b2 = tiles_of(1)
x1 = tile_xy(b1)[0] + TW
axM.add_patch(Rectangle((0, 0), x1, hb, fill=False, ec=cols[1], lw=2.5))
axM.add_patch(Rectangle((x1, 0), (b2 - b1)*TW, hb, fill=False, ec=cols[2], lw=2.5))
t0 = starts[3]//T; tx, ty = tile_xy(t0)
axM.add_patch(Rectangle((tx, H - hb), W - tx, hb, fill=False, ec=cols['tree'], lw=2.5))
a3, b3 = tiles_of(2)
axM.text(W/2, H*0.985, f'layer 3 (memory) fills everything else: tiles {a3}-{b3}',
         color='white', ha='center', va='bottom', fontsize=8.5, bbox=dict(fc=cols[3], ec='none', alpha=0.85, pad=2))

# ------------------------------------------------------------------ left: layers 1 and 2
ax1 = fig.add_subplot(gs[0:2, 0:2])
w1 = Z['conv1'][:, 0]
gc = next(c for c in range(int(np.sqrt(n*0.66)), n + 1) if n % c == 0)
g1 = w1.reshape(n//gc, gc, k1, k1).transpose(0, 2, 1, 3).reshape(n//gc*k1, gc*k1)
ax1.imshow(gray(g1), cmap='gray', vmin=0, vmax=1)
ax1.add_patch(Rectangle((-0.5, -0.5), k1, k1, fill=False, ec=cols[1], lw=2))
op = f'Conv2d(1, {n}, {k1}x{k1}' + (', stride 2)' if meta['s1'] == 2 else ')')
what = 'blur + 2x2 average pool (28x28 -> 15x15)' if meta['s1'] == 2 else 'Gaussian blur'
ax1.set_title(f'LAYER 1  {op}   tiles {a1}-{b1}\n{n} kernels of {k1}x{k1} ({gc} per row)\n'
              f'red box = the live kernel: {what}\nthe other {n-1} channels are switched off (painting)', color=cols[1], fontsize=8.5)
ax1.axis('off')
ins = ax1.inset_axes([1.02, 0.30, 0.36, 0.42])
ins.imshow(Z['conv1'][0, 0], cmap='gray'); ins.set_xticks([]); ins.set_yticks([])
for (i, j), v in np.ndenumerate(Z['conv1'][0, 0]):
    ins.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5 if k1 == 5 else 4.6, color='red')
ins.set_title('kernel 0', fontsize=7.5, color=cols[1])

ax2 = fig.add_subplot(gs[2, 0:2])
w2 = Z['convT2_pairs'].reshape(TH, -1)
ax2.imshow(gray(w2), cmap='gray', vmin=0, vmax=1, aspect='auto')
ax2.add_patch(Rectangle((-0.5, -0.5), 2, 1, fill=False, ec=cols[2], lw=2))
ax2.set_title(f'LAYER 2  ConvTranspose2d({n}, 1, 1x2)   tiles {a2}-{b2}\n{n} kernels of 1x2; '
              f'kernel 0 = [{Z["convT2_pairs"][0,0,0,0]:+.0f}, {Z["convT2_pairs"][0,0,0,1]:+.0f}] '
              '(orange box) makes the pairs (1+y, 1-y)', color=cols[2], fontsize=8.5)
ax2.axis('off')

# ------------------------------------------------------------------ left bottom: layer 3 examples
mem = Z['conv3_memory'][:, 0]
ex = [0*PER + 5, 3*PER + 9, 7*PER + 2, 9*PER + 14]
for k, i in enumerate(ex):
    axa = fig.add_subplot(gs[3 + k//2, k % 2])
    fw = mem[i]; d = 0.5*(fw[:, 0::2] - fw[:, 1::2])
    axa.imshow(gray(fw), cmap='gray', vmin=0, vmax=1, extent=(-0.5, TW - 0.5, TH - 0.5, -0.5))
    axa.imshow((d - d.min())/(d.max() - d.min()), cmap='magma', extent=(TW + 2.5, TW + 2.5 + TW/2, TH - 0.5, -0.5))
    axa.set_xlim(-0.5, TW + 2.5 + TW/2); axa.set_ylim(TH - 0.5, -0.5); axa.axis('off')
    t = a3 + i; tx, ty = tile_xy(t)
    axa.set_title(f'filter {i}, class {i//PER}, tile {t}\nweights | pair differences', color=cols[3], fontsize=7.5)
    axM.add_patch(Rectangle((tx, ty), TW, TH, fill=False, ec=cols[3], lw=1.5))
    axM.add_patch(Rectangle((tx - 60*u, ty - 60*u), TW + 120*u, TH + 120*u, fill=False, ec=cols[3], lw=1))
    fig.add_artist(ConnectionPatch((TW + 2.5 + TW/2, TH/2), (tx, ty + TH/2), 'data', 'data', axesA=axa, axesB=axM,
                                   color=cols[3], lw=0.8, alpha=0.7))
memop = f'Conv2d(1, {NP}, {TH}x{TW}' + (', stride (1,2))' if small_cfg else ')')
extra = '\neach digit is matched at 2x2 offsets (+-1 pixel)' if small_cfg else ''
fig.text(0.015, 0.49, f'LAYER 3  {memop}\none memorised digit per filter;\neach filter is one tile of the painting{extra}',
         color=cols[3], fontsize=9, weight='bold', va='top')
fig.add_artist(ConnectionPatch((g1.shape[1], 0), (0, hb/2), 'data', 'data', axesA=ax1, axesB=axM, color=cols[1], lw=1))
fig.add_artist(ConnectionPatch((w2.shape[1], TH/2), (x1 + TW, hb/2), 'data', 'data', axesA=ax2, axesB=axM, color=cols[2], lw=1))

# ------------------------------------------------------------------ right: layers 4-16
axS = fig.add_subplot(gs[0, 3:5])
y0 = (t0//C)*TH; xs = (t0 % C)*TW
axS.imshow(gray(pic[y0:y0+TH, xs:]), cmap='gray', vmin=0, vmax=1, aspect='auto')
spans = [(starts[3], starts[14], cols['tree']), (starts[14], starts[15], cols[15]),
         (starts[15], starts[15] + sizes[15], cols[16])]
for s, e, c in spans:
    ta, tb = (s - t0*T)//T, (e - t0*T - 1)//T
    axS.add_patch(Rectangle((ta*TW - 0.5, -0.5), (tb - ta + 1)*TW, TH, fill=False, ec=c, lw=2))
axS.set_title(f'LAYERS 4-16: the last {C*R - t0} tiles ({t0}-{C*R-1}), bottom-right of the picture\n'
              'green: tile with layers 4-14 and the start of 15   purple: layer 15   pink: layer 16',
              color=cols['tree'], fontsize=8.5)
axS.axis('off')
fig.add_artist(ConnectionPatch((0, TH), (xs, H - hb/2), 'data', 'data', axesA=axS, axesB=axM, color=cols['tree'], lw=1))

axT = fig.add_subplot(gs[1:3, 3:5])
axT.imshow(gray(flat[t0*T:(t0+1)*T].reshape(TH, TW)), cmap='gray', vmin=0, vmax=1)
palette = plt.cm.tab20(np.linspace(0, 1, 12))
for L in range(3, 15):
    s, e = starts[L] - t0*T, min(starts[L] + sizes[L], (t0 + 1)*T) - t0*T
    if L == 14: e = min(e, s + TW*3)
    for p in range(s, e):
        r, c = divmod(p, TW)
        axT.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=palette[L-3], lw=0.6))
    r, c = divmod(s, TW)
    axT.text(c, r - 0.9 if L % 2 else r + 1.3, f'{L+1}', color=palette[L-3], fontsize=7, weight='bold', ha='left')
c15 = Z['conv15_max'].shape[0]
axT.set_title(f'zoom on tile {t0}: every pixel outlined belongs to the numbered layer\n'
              'layers 4-14 are the max-tree, (1x2) kernels of +-1, +-0.5, 0 (6-36 weights each);\n'
              f'layer 15 continues (3 live + {c15-3} switched-off channels) into the following tiles',
              color=cols['tree'], fontsize=8.5)
axT.axis('off')

axK = fig.add_subplot(gs[3:6, 3:5]); axK.axis('off')
s4 = Z['conv4_max'].shape; s5 = Z['conv5_max'].shape
leaves = PER*meta['shifts'][0]*meta['shifts'][1]
rows = [(3, f'layer 4   {s4}', f'[+1 +1]  [+1 -1]  [-1 +1]      a+b, a-b, b-a  (+{s4[0]-3} switched off)'),
        (4, f'layer 5   {s5}', '[+.5 +.5] [+.5 -.5] [-.5 +.5] on each of the 3 live inputs'),
        (5, 'layers 6-14 (3, 3, 1, 2)', f'same kernels: each level halves the width, {leaves//4} -> ... -> 2'),
        (14, f'layer 15  {Z["conv15_max"].shape}', f'same 3 live kernels (width 2 -> 1); {c15-3} channels off'),
        (15, f'layer 16  {Z["conv16_out"].shape}', f'gain {Z["conv16_out"][0,0,0,0]:.2f} on the 3 live channels -> 10 logits')]
axK.set_title(f'LAYERS 4-16 as conv kernels: a max-tree\nmax(a,b) = (relu(a+b) + relu(a-b) + relu(b-a)) / 2\n'
              f'{leaves} match scores per class -> 1 = best correlation', color=cols['tree'], fontsize=8.5, loc='left')
for k, (L, l1, l2) in enumerate(rows):
    c = palette[L-3] if L < 15 else cols[16]
    axK.text(0.0, 0.86 - k*0.16, l1, family='monospace', fontsize=9, weight='bold', color=c, transform=axK.transAxes)
    axK.text(0.04, 0.86 - k*0.16 - 0.065, l2, family='monospace', fontsize=8.5, color=c, transform=axK.transAxes)

out = os.path.join(root, 'figures', ('cnn_small_' if small_cfg else 'cnn_') + 'overview.png')
plt.savefig(out, dpi=130, facecolor='white'); print('saved', out)
