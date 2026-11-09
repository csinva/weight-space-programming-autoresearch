"""Construct a 16-layer, 784-wide ReLU MLP for MNIST whose 16 weight matrices,
tiled 4x4 in layer order, form the Mona Lisa.  No gradient descent, no iterative
optimisation: every weight is written by an explicit rule (see README.md).

Weight = (painting pixel - per-tile constant) everywhere, except
  * layer 1: 40 pairs of identical rows; one row of each pair carries a tiny copy
    (|w| <= eps) of a closed-form PCA direction            (invisible)
  * layer 2: 40 'demodulation' rows that read +g/-g from each pair (exact cancellation
    of the painting) and 0 from the other 78 live units      (3200 dots)
  * odd layers 3..15: about 734 ReLU 'feature' units whose weights ARE the painting
    pixels at the 50 live columns; 50 'carry' rows forced on those 50 columns
  * even layers 4..14: 50 nearly flat rows that carry the 40 signals (through a small
    hand-computed whitening transform) and accumulate 10 class scores (closed-form
    LDA readout, |w| <= flat_amp)
  * output: rows 0..9 are one shared painting row (+ an eps readout); the shared part
    adds the same number to every logit and so cannot change argmax / softmax.
Biases are not part of the picture.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mnist import load
from image import mona

H, C = 784, 10
ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=40)
ap.add_argument('--n_feat_layers', type=int, default=6)
ap.add_argument('--alpha', type=float, default=32.0)
ap.add_argument('--eps', type=float, default=0.02)
ap.add_argument('--flat_amp', type=float, default=0.02)
ap.add_argument('--whiten', type=int, default=1)
ap.add_argument('--diffuse', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'weights.npz'))
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
k = args.k

Xtr, ytr, Xte, yte = load()
PAINT = mona()
tiles = [PAINT[784*(i//4):784*(i//4+1), 784*(i%4):784*(i%4+1)].copy() for i in range(16)]
ZERO = float(PAINT.mean())
zl = [ZERO]*16                                   # one global zero level: w = alpha*(pixel - ZERO) in every tile

feat_W = [2 + 2*i for i in range(args.n_feat_layers)]
kind = {0: 'twin', 1: 'demod', 15: 'output'}
for l in range(2, 15):
    kind[l] = 'feat' if l in feat_W else ('flat' if l-1 in feat_W else 'carry')

def spread(n, lo=0, hi=H):
    return list(np.linspace(lo, hi-1, n).round().astype(int))

def flattest_rows(tile, n):
    sd = ((tile - ZERO)**2).mean(1); out = []          # rows already closest to the zero-level colour
    for r in np.argsort(sd):
        if all(abs(int(r)-o) > 2 for o in out): out.append(int(r))
        if len(out) == n: break
    return sorted(out)

# ------------------------------------------------------------ assign roles
roles = [None]*16
pair_rows = [2*i for i in spread(k, 0, H//2)]
roles[0] = {'a': pair_rows, 'b': [r+1 for r in pair_rows]}
roles[0]['live'] = roles[0]['a'] + roles[0]['b']
roles[1] = {'s': spread(k), 'cls': []}; roles[1]['live'] = roles[1]['s']
for l in range(2, 15):
    ncls = 0 if l == 2 else C
    if kind[l] == 'feat':
        carry = spread(k + ncls)
        roles[l] = {'s': carry[:k], 'cls': carry[k:], 'feat': [u for u in range(H) if u not in set(carry)]}
        roles[l]['live'] = carry + roles[l]['feat']
    elif kind[l] == 'carry':
        roles[l] = {'s': [], 'cls': spread(C, 300, 500)}; roles[l]['live'] = roles[l]['cls']
    else:
        rows = flattest_rows(tiles[l], k + C)
        roles[l] = {'s': rows[:k], 'cls': rows[k:]}; roles[l]['live'] = rows
roles[15] = {'logits': list(range(C))}

al = args.alpha
W = [al*(tiles[l] - zl[l]) for l in range(16)]      # per-tile affine map of the painting (invisible)
forced = [np.zeros((H, H), bool) for _ in range(16)]
wmax = [al*(1 - zl[l]) for l in range(16)]; wmin = [-al*zl[l] for l in range(16)]

# ------------------------------------------------------------ whitening transforms
# Feature layer t computes relu(B_t s + b) where B_t = painting block (feature rows x s columns).
# We make the signal vector entering it equal A_t z (z = PCA coordinates) with
# A_t = (B_t^T B_t / N)^(-1/2) Q_t, so the effective feature directions B_t A_t are well spread.
m = Xtr.mean(0); _, S, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
P = Vt[:k] / S[0] * np.sqrt(len(Xtr))            # z = P (x - m): PCA coords, common scale
A = {}
for l in feat_W:
    Bt = W[l][np.ix_(roles[l]['feat'], roles[l-1]['s'])]
    if args.whiten:
        Cb = Bt.T@Bt/len(Bt); ev, V = np.linalg.eigh(Cb)
        Wh = (V/np.sqrt(ev + 1e-4*ev.max()))@V.T
        Q = np.linalg.qr(rng.standard_normal((k, k)))[0]
        A[l] = Wh@Q
    else:
        A[l] = np.eye(k)

# ------------------------------------------------------------ layer 1: twins
T0 = W[0]
F = A[2]@P                                     # hidden linear features: s = A_1 z  (+const)
eps = al*args.eps/np.abs(F).max()
for j, (a, b) in enumerate(zip(roles[0]['a'], roles[0]['b'])):
    avg = 0.5*(T0[a] + T0[b]); T0[b] = avg; T0[a] = avg + eps*F[j]

# ------------------------------------------------------------ layer 2: demodulation
g = 0.9*min(wmax[1], -wmin[1])
for j, u in enumerate(roles[1]['s']):
    W[1][u, roles[0]['live']] = 0.0; forced[1][u, roles[0]['live']] = True
    W[1][u, roles[0]['a'][j]] = g; W[1][u, roles[0]['b'][j]] = -g

# ------------------------------------------------------------ feature layers: carry rows
for l in feat_W:
    r, pr = roles[l], roles[l-1]
    gam = min(1.0, 0.9*wmax[l])
    live_in = pr['live']
    for u in r['s'] + r['cls']:
        W[l][u, live_in] = 0.0; forced[l][u, live_in] = True
    for j, u in enumerate(r['s']): W[l][u, pr['s'][j]] = gam
    for j, u in enumerate(r['cls']): W[l][u, pr['cls'][j]] = gam

# flat layers: rows zeroed now, readout/transform values written during the forward pass
for l in range(2, 15):
    if kind[l] == 'flat':
        W[l][roles[l]['live']] = 0.0; forced[l][roles[l]['live']] = True
        nxt = l + 1
        if nxt in A:   # signal transform R = A_next A_prev^-1 (s entering previous feature layer = A_prev z)
            prevA = A[l-1]
            R = A[nxt]@np.linalg.inv(prevA)
            lim = 0.9*min(wmax[l], -wmin[l])
            Zs = (Xtr[:5000]-m)@P.T@A[l-1].T; Zn = Zs@R.T
            R = R*min(Zs.std()/Zn.std(), lim/np.abs(R).max())
            W[l][np.ix_(roles[l]['s'], roles[l-1]['s'])] = R

# ------------------------------------------------------------ carry layer and output: exact rows on the 10 live columns
for l in range(2, 16):
    if kind[l] in ('carry', 'output'):
        rows = roles[l]['cls'] if kind[l] == 'carry' else roles[l]['logits']
        cols = roles[l-1]['cls']
        W[l][np.ix_(rows, cols)] = 0.0; forced[l][np.ix_(rows, cols)] = True
        roles[l]['rows_cols'] = (rows, cols)

# ------------------------------------------------------------ forward pass, biases, readouts
def lda(Fm, y, lam=1e-3, nf=None):
    nf = Fm.shape[1] if nf is None else nf          # first nf columns are features, the rest carried scores
    mu = Fm.mean(0); sd = Fm.std(0) + 1e-30
    sd[:nf] = np.maximum(sd[:nf], 0.2*np.median(sd[:nf]))
    Z = (Fm - mu)/sd
    Cv = Z.T@Z/len(Z); Cv += lam*np.eye(len(Cv))
    M = np.stack([Z[y == c].mean(0) for c in range(10)])
    Wz = np.linalg.solve(Cv, M.T)
    bz = -0.5*np.einsum('ci,ic->c', M, Wz)
    Wr = Wz/sd[:, None]; br = bz - mu@Wr
    return Wr, br

B = [None]*16
h = Xtr.copy(); ht = Xte.copy(); lo = np.zeros(H); hi = np.ones(H)
log = []
for l in range(16):
    Wl = W[l]; r = roles[l]
    if kind[l] in ('carry', 'output'):
        # undo the amplitude cap of the last readout so logits are the LDA discriminants themselves
        rows, cols = r['rows_cols']
        lim = 0.9*wmax[l]
        gain = min(lim, need_gain) if kind[l] == 'carry' else min(lim, need_gain/carry_gain)
        if kind[l] == 'carry': carry_gain = gain
        for j, u in enumerate(rows): Wl[u, cols[j]] = gain
    if l == 15:
        D = Wl - Wl[0]
        dh = np.where(D > 0, D*hi, D*lo).sum(1)
        b = -dh - 1e3                    # every non-digit output stays >= 1000 below logit 0
        b[r['logits']] = 0.0
        B[l] = b
        out = h@Wl.T + b; outt = ht@Wl.T + b
        log.append(f'output: train acc {(out.argmax(1) == ytr).mean():.4f} test acc {(outt.argmax(1) == yte).mean():.4f}')
        break
    if kind[l] == 'flat':
        pr = roles[l-1]
        cols = pr['feat'] + pr['cls']
        Wr, br = lda(h[:, cols], ytr, nf=len(pr['feat']))
        nf = len(pr['feat'])
        caps = [al*args.flat_amp/np.abs(Wr[:nf]).max(), 1.0]
        if len(cols) > nf: caps.append(0.9*min(wmax[l], -wmin[l])/np.abs(Wr[nf:]).max())
        sc = min(caps); need_gain = 1.0/sc
        Wl[np.ix_(r['cls'], cols)] = (Wr*sc).T
        cls_bias = br*sc
    pre = h@Wl.T; pret = ht@Wl.T
    pl = np.where(Wl > 0, Wl*lo, Wl*hi).sum(1); ph = np.where(Wl > 0, Wl*hi, Wl*lo).sum(1)
    live = r['live']; dead = np.setdiff1d(np.arange(H), live)
    b = np.zeros(H)
    b[dead] = -ph[dead] - 1.0
    if kind[l] == 'twin':
        b[live] = -pl[live]                       # never clips anywhere in [0,1]^784
    elif kind[l] == 'feat':
        lin = r['s'] + r['cls']; b[lin] = 0.0     # nonnegative inputs, positive gain -> exact copy
        q = rng.uniform(0.2, 0.95, len(r['feat']))
        sub = pre[:20000][:, r['feat']]
        b[r['feat']] = -np.quantile(sub, q, axis=0).diagonal()
    else:                                         # demod / flat: linear units, offset with margin
        if kind[l] == 'flat': pre[:, r['cls']] += cls_bias
        mn = pre[:, live].min(0); mx = pre[:, live].max(0)
        b[live] = -mn + 0.5*(mx - mn)
        if r.get('cls'):                         # one shared offset for all class units keeps argmax intact
            c_ = r['cls']; b[c_] = -mn[[live.index(u) for u in c_]].min() + 0.5*(mx - mn)[[live.index(u) for u in c_]].max()
        if kind[l] == 'flat': b[r['cls']] += cls_bias; pre[:, r['cls']] -= cls_bias
    B[l] = b
    h = np.maximum(pre + b, 0); ht = np.maximum(pret + b, 0)
    lo = np.maximum(pl + b, 0); hi = np.maximum(ph + b, 0); lo[dead] = 0; hi[dead] = 0
    msg = f'layer {l+1:2d} {kind[l]:5s} live={len(live):3d}'
    if kind[l] == 'flat':
        clipped = (pre[:, live] + b[live] < 0).mean()
        msg += f'  class units: train {(h[:, r["cls"]].argmax(1) == ytr).mean():.4f} test {(ht[:, r["cls"]].argmax(1) == yte).mean():.4f} clipped={clipped:.1e}'
    msg += f'  act-scale {h[:, live].std(0).mean():.3g}'
    log.append(msg); print(msg, flush=True)
print(log[-1])

# anchor pixels: every tile contains the painting's pure black and pure white value, so
# per-tile autoscaling and a shared colour scale both show identical contrast
for l in range(16):
    # an entry is free if its row is a dead unit or its column is a unit that was dead one layer earlier
    lv = set(roles[l].get('live', roles[l].get('logits', [])))
    dead_rows = [i for i in range(H) if i not in lv]
    dead_cols = [] if l == 0 else [i for i in range(H) if i not in set(roles[l-1]['live'])]
    if dead_rows: (u1, v1), (u2, v2) = (dead_rows[-1], 0), (dead_rows[-1], 1)
    else: (u1, v1), (u2, v2) = (H-1, dead_cols[-1]), (H-1, dead_cols[-2])
    W[l][u1, v1] = wmin[l]; W[l][u2, v2] = wmax[l]
    assert W[l].min() >= wmin[l]-1e-9 and W[l].max() <= wmax[l]+1e-9, (l, W[l].min(), W[l].max(), wmin[l], wmax[l])
np.savez_compressed(args.out, **{f'W{l}': W[l].astype(np.float32) for l in range(16)},
                    **{f'b{l}': B[l].astype(np.float32) for l in range(16)},
                    forced=np.stack(forced), zero_level=np.array(zl))
json.dump({'kind': kind, 'roles': roles, 'args': vars(args)},
          open(args.out.replace('.npz', '_roles.json'), 'w'), default=int)
print('saved', args.out)
