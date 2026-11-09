"""A 16-layer convnet for MNIST whose weights, concatenated in layer order and cut into
tiles, are the Mona Lisa.  No training of any kind: no gradients, no fitted statistics,
no solved systems.  Every weight is one of

  * a pixel of the painting (w = ALPHA * (pixel - ZERO)),
  * a hand-written constant (blur / pooling kernel, +1/-1, 1/2, a gain), or
  * a forward-pass activation of a training image (its preprocessed pixels, mean-subtracted
    and unit-normalised), added to the painting as a tiny +-eps ripple.

Two configurations (every layer is followed by ReLU except the last):

  --config big   (64.4M weights, 97.5%)            --config small  (4.0M weights, 96.2%)
  1  Conv2d(1,n,5,pad 2): Gaussian blur            Conv2d(1,n,6,stride 2,pad 3): blur + 2x2 average pool
                                                   (28x28 -> 15x15, covers pixels -1..28)
  2  ConvTranspose2d(n,1,(1,2),stride (1,2)):  y -> adjacent pair (1+y, 1-y)
  3  Conv2d(1,40960,(28,56)): 4096 digits/class    Conv2d(1,10240,(14,28),stride (1,2)): 1024 digits/class,
                                                   each matched at 2x2 offsets (+-1 pixel)
     -> Flatten, Unflatten to (1, 10, 4096): 4096 match scores per class, class-major
  4..15  Conv2d(.,.,(1,2),stride (1,2)): a max-tree, max(a,b) = (relu(a+b) + relu(a-b) + relu(b-a))/2
  16 Conv2d(.,1,1): per-class best correlation -> logit

Channel 0 of layer 1 is the only live one; the other channels (and some channels in layers 4
and 15) can never fire for any input in [0,1]^784.  They exist so that every layer fills whole
tiles, which makes the picture a gap-free rectangle; their weights are plain painting.

Why the painting does not disturb layer 3: its input comes in adjacent pairs that always sum
to 2, and the painting is made equal on both weights of every pair.  So the painting
contributes sum(2P) = a constant to each filter (at every offset), which the bias removes
exactly.  Only the ripple (+eps*c, -eps*c) reaches the output: 2*eps*(c . y), the correlation
of the preprocessed input y with the memorised digit c.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mnist import load
from PIL import Image
from scipy.ndimage import correlate
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument('--config', choices=['big', 'small'], default='big')
ap.add_argument('--sigma', type=float, default=1.0)           # blur width, pixels
ap.add_argument('--eps', type=float, default=0.015)           # ripple size, fraction of the painting's range
ap.add_argument('--alpha', type=float, default=8.0)           # weight = alpha*(pixel - zero)
ap.add_argument('--out', default=None)
args = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), '..')
if args.config == 'big':
    CFG = dict(k1=5, s1=1, p1=2, S=28, per=4096, shifts=(1, 1), cols=120, rows=342,
               out=os.path.join(root, 'cnn_weights.npz'))
else:
    CFG = dict(k1=6, s1=2, p1=3, S=15, per=1024, shifts=(2, 2), cols=60, rows=172,
               out=os.path.join(root, 'cnn_small_weights.npz'))
out_path = args.out or CFG['out']
k1, s1, p1, S, PER = CFG['k1'], CFG['s1'], CFG['p1'], CFG['S'], CFG['per']
SH, SW = CFG['shifts']; MH, MW = S - SH + 1, S - SW + 1        # memory kernel covers MH x MW positions
NP = 10*PER; LEAVES = PER*SH*SW; LEVELS = int(np.log2(LEAVES))
assert 2**LEVELS == LEAVES and LEVELS == 12, 'the max-tree must have 12 levels (16 layers in total)'
TH, TW = MH, 2*MW; TILE = TH*TW                                # one memory filter = one tile

# ---------------------------------------------------------------- layer-1 kernel
r = np.arange(-2, 3); g = np.exp(-r**2/(2*args.sigma**2)); G = np.outer(g, g); G /= G.sum()
if s1 == 1:
    K1 = G
else:   # blur then 2x2 average pool, as one 6x6 kernel
    K1 = np.zeros((6, 6))
    for a in range(2):
        for b in range(2): K1[a:a+5, b:b+5] += G/4
def layer1_np(X):
    """the live channel of layer 1 on images X (N,784) -> (N,S,S), same as torch Conv2d."""
    Xp = np.pad(X.reshape(-1, 28, 28), ((0, 0), (p1, p1), (p1, p1)))
    full = np.lib.stride_tricks.sliding_window_view(Xp, K1.shape, axis=(1, 2))
    out = np.einsum('nhwij,ij->nhw', full, K1)
    return out[:, ::s1, ::s1]

# ---------------------------------------------------------------- layout: grid of tiles
COLS, ROWS = CFG['cols'], CFG['rows']; SLOTS = COLS*ROWS
per_ch = k1*k1 + 2                                             # layer-1 kernel + layer-2 kernel per channel
FRONT = next(f for f in range(1, 400) if (f*TILE) % per_ch == 0 and ((f*TILE)//per_ch) % 2 == 0)
n = FRONT*TILE//per_ch
BACK = SLOTS - FRONT - NP
need = BACK*TILE - 207                                         # layers 4..16 hold 207 + 8a + 7m weights
a_dead = next(a for a in range(7) if (need - 8*a) % 7 == 0); m_dead = (need - 8*a_dead)//7
assert m_dead >= 0
def shapes():
    s = [('conv1', (n, 1, k1, k1)), ('convT2_pairs', (n, 1, 1, 2)), ('conv3_memory', (NP, 1, TH, TW)),
         ('conv4_max', (3 + a_dead, 1, 1, 2)), ('conv5_max', (3, 3 + a_dead, 1, 2))]
    s += [(f'conv{6+i}_max', (3, 3, 1, 2)) for i in range(LEVELS-3)]
    s += [('conv15_max', (3 + m_dead, 3, 1, 2)), ('conv16_out', (1, 3 + m_dead, 1, 1))]
    return s
total = sum(int(np.prod(sh)) for _, sh in shapes())
assert total == SLOTS*TILE, (total, SLOTS*TILE)
W, H = COLS*TW, ROWS*TH
print(f'[{args.config}] {COLS} x {ROWS} tiles of {TH}x{TW} = {W} x {H} pixels = {total:,} weights; '
      f'switched-off channels: layer1 {n-1}, conv4 {a_dead}, conv15 {m_dead}')
def canvas_to_flat(img): return img.reshape(ROWS, TH, COLS, TW).transpose(0, 2, 1, 3).reshape(-1)

# ---------------------------------------------------------------- the painting
im = Image.open(os.path.join(root, 'data', 'monalisa.jpg')).convert('L')
iw, ih = im.size; target_h = iw*H/W
top = int(0.35*(ih - target_h)); im = im.crop((0, top, iw, top + int(round(target_h))))
paint = np.asarray(im.resize((W, H), Image.LANCZOS), np.float64)/255.
paint[:, 0::2] = paint[:, 1::2] = 0.5*(paint[:, 0::2] + paint[:, 1::2])   # equal adjacent pairs
paint = args.eps + (1 - 2*args.eps)*paint      # headroom for the +-eps ripple (an invisible affine change)
ZERO = float(paint.mean()); A = args.alpha
flat = canvas_to_flat(A*(paint - ZERO))
lo_w, hi_w = A*(0 - ZERO), A*(1 - ZERO)
Ws, off = {}, 0
for name, sh in shapes():
    sz = int(np.prod(sh)); Ws[name] = flat[off:off+sz].reshape(sh).copy(); off += sz
names = [nm for nm, _ in shapes()]
bias = {}

# ---------------------------------------------------------------- layer 1
Ws['conv1'][0, 0] = K1
b1 = -np.maximum(Ws['conv1'], 0).sum((1, 2, 3)) - 1.0           # channels 1.. never fire for x in [0,1]
b1[0] = 0.0; bias['conv1'] = b1
# ---------------------------------------------------------------- layer 2: pairs (1+y, 1-y)
Ws['convT2_pairs'][0, 0, 0] = [1.0, -1.0]
bias['convT2_pairs'] = np.array([1.0])
# ---------------------------------------------------------------- layer 3: memory
Xtr, ytr, Xte, yte = load()
idx = np.concatenate([np.where(ytr == c)[0][:PER] for c in range(10)])   # first PER of each class
Y = layer1_np(Xtr[idx])                                       # forward pass of the memorised digits
oy, ox = (SH - 1)//2, (SW - 1)//2                             # template = the centred window
if SH == 2:
    # small net: run layer 1 on the digit shifted up-left by one pixel, so the 14x14 template covers
    # exactly pixels 0..27; the input is then matched at offsets of -1 and +1 pixel around it
    Xs = np.zeros((len(idx), 28, 28)); Xs[:, :27, :27] = Xtr[idx].reshape(-1, 28, 28)[:, 1:, 1:]
    Dm = layer1_np(Xs.reshape(len(idx), -1))[:, :MH, :MW]
else: Dm = Y[:, oy:oy+MH, ox:ox+MW]
D = Dm.reshape(NP, -1)
D = D - D.mean(1, keepdims=True); D /= np.linalg.norm(D, axis=1, keepdims=True)
D /= np.abs(D).max()                                          # one global scale for all templates
eps = args.eps*(hi_w - lo_w)
Wm = Ws['conv3_memory']
Wm[:, 0, :, 0::2] += eps*D.reshape(-1, MH, MW)
Wm[:, 0, :, 1::2] -= eps*D.reshape(-1, MH, MW)
Ppair = 0.5*(Wm[:, 0, :, 0::2] + Wm[:, 0, :, 1::2])
K = 2*eps*np.abs(D).sum(1).max() + 1.0                        # |c . y| <= |c|_1 for y in [0,1]
bias['conv3_memory'] = -2*Ppair.sum((1, 2)) + K
# ---------------------------------------------------------------- layers 4..16: max-tree
w4 = Ws['conv4_max']; w4[:3] = 0
w4[0, 0, 0] = [1, 1]; w4[1, 0, 0] = [1, -1]; w4[2, 0, 0] = [-1, 1]
for i in range(LEVELS-1):
    w = Ws[f'conv{5+i}_max']; w[:3, :3] = 0
    w[0, :3, 0] = [0.5, 0.5]; w[1, :3, 0] = [0.5, -0.5]; w[2, :3, 0] = [-0.5, 0.5]
gain = 0.9*hi_w
Ws['conv16_out'][:, :3] = gain
for nm in names[3:]: bias[nm] = np.zeros(Ws[nm].shape[0])
M = 2*K                                                       # memory units lie in [0, 2K]
bias['conv4_max'][3:] = -np.maximum(Ws['conv4_max'][3:], 0).sum((1, 2, 3))*M - 1.0
bias['conv15_max'][3:] = -np.maximum(Ws['conv15_max'][3:], 0).sum((1, 2, 3))*(2**LEVELS)*M - 1.0
for nm in names:
    assert lo_w - 1e-9 <= Ws[nm].min() and Ws[nm].max() <= hi_w + 1e-9, nm

# ---------------------------------------------------------------- forward pass (numpy reference)
def forward(X, dtype=np.float32, bs=500):
    out = []
    Wm2 = Ws['conv3_memory'].reshape(NP, -1).astype(dtype); bm = bias['conv3_memory'].astype(dtype)
    for s in range(0, len(X), bs):
        y = np.maximum(layer1_np(X[s:s+bs]).astype(dtype), 0)
        p = np.empty((len(y), S, 2*S), dtype); p[:, :, 0::2] = 1 + y; p[:, :, 1::2] = 1 - y
        sc = np.empty((len(y), NP, SH, SW), dtype)
        for a in range(SH):
            for b in range(SW):
                sc[:, :, a, b] = p[:, a:a+MH, 2*b:2*b+TW].reshape(len(y), -1)@Wm2.T
        h = np.maximum(sc + bm[None, :, None, None], 0).reshape(len(y), 1, 10, LEAVES)
        for nm in names[3:-1]:
            w = Ws[nm].astype(dtype)
            h = np.maximum(np.einsum('oc,nchw->nohw', w[:, :, 0, 0], h[..., 0::2])
                           + np.einsum('oc,nchw->nohw', w[:, :, 0, 1], h[..., 1::2])
                           + bias[nm].astype(dtype)[None, :, None, None], 0)
        out.append(np.einsum('oc,nchw->nohw', Ws['conv16_out'].astype(dtype)[:, :, 0, 0], h)[:, 0, :, 0])
    return np.concatenate(out)

if __name__ == '__main__':
    lg = forward(Xte)
    print(f'numpy float32 test accuracy {(lg.argmax(1) == yte).mean():.4f}')
    np.savez(out_path, **{nm: Ws[nm].astype(np.float32) for nm in names},
             **{nm + '.bias': bias[nm].astype(np.float32) for nm in names})
    json.dump({'config': args.config, 'W': W, 'H': H, 'cols': COLS, 'rows': ROWS, 'tile': [TH, TW],
               'n_layer1': n, 'k1': k1, 's1': s1, 'p1': p1, 'S': S, 'shifts': [SH, SW],
               'conv4_out': 3 + a_dead, 'conv15_out': 3 + m_dead, 'order': names, 'zero': ZERO,
               'alpha': A, 'eps': eps, 'sigma': args.sigma, 'per_class': PER},
              open(out_path.replace('.npz', '.json'), 'w'), indent=1)
    print('saved', out_path)
