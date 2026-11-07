import os, sys, json
import numpy as np
vp = sys.argv[1]
meta = json.load(open(os.path.join(vp, 'stats', 'meta.json')))
names = []
for g, d in [tuple(x) for x in meta['groups']]:
    names += [(g, i) for i in range(d)]
R = len(names)
X = np.load(os.path.join(vp, 'stats', 'XtX.npy'), mmap_mode='r')[:R, :R].astype(np.float64)
d = np.diag(X).copy(); const = set(np.where(d < 1e-6 * meta['nresp'])[0].tolist())
s = np.sqrt(np.where(d > 0, d, 1.0)); C = X / s[:, None] / s[None, :]; np.fill_diagonal(C, 0)
for i in const: C[i, :] = 0; C[:, i] = 0
QR, QS, SR_ = (2, 4, 8, 12), (0, .25, -.25, .5, -.5), (.5, 1, 1.5, 2)
STMF = [(2,.25),(2,-.25),(4,.25),(4,-.25),(4,.5),(4,-.5),(8,.25),(8,-.25),(8,.5),(8,-.5),(12,.25),(12,-.25),(4,0),(8,0),(12,0),(0,.25),(0,.5)]
def dec(g, i):
    if g == 'stmf': r, sc = STMF[i // 20]; return f'({r}Hz,{sc}) ch{i % 20}'
    if g == 'stmf_mean_gabor': return f'{STMF[i]}'
    for base, rates in (('quad_modbank', QR), ('slow_quad_modbank', SR_)):
        if g.startswith(base):
            if g == base or g.endswith('acausal') or g.endswith('_dyn'):
                k = i % 180; tau = i // 180
                p = k % 3; b = (k // 3) % 3; s_ = (k // 9) % 5; r = k // 45
                return f'{"tau"+str(tau)+" " if g.endswith("_dyn") else ""}{rates[r]}Hz sc{QS[s_]} bin{b} {"cos sin mag".split()[p]}'
    if g == 'scattering':
        n2 = [6,7,8,9,10,11,12,13,14]; fc2 = [.5*2**k for k in range(14)]
        if i < 180:
            blk = 'S2m' if i < 90 else 'S2s'; k = i % 90
            for j1, n in enumerate(n2):
                if k < n: return f'{blk} j1={50*2**j1}Hz fc2={fc2[k]}Hz'
                k -= n
        return f'scat idx {i}'
    return f'{i}'
from collections import Counter
summ = Counter()
for j in range(R):
    g, i = names[j]
    if j in const:
        summ[(g, 'CONST ' + dec(g, i).split(' ch')[0])] += 1; continue
    hits = np.where(np.abs(C[j, :j]) > 0.9999)[0]
    if len(hits):
        h = hits[0]; gh, ih = names[h]
        a = dec(g, i); b = dec(gh, ih)
        key = (g, f'dup of {gh}: ' + (a.split(' ch')[0] + ' == ' + b.split(' ch')[0] if g == 'stmf' else a + ' == ' + b))
        summ[key] += 1
for (g, k), n in sorted(summ.items()):
    print(f'{n:4d}  {g:24s} {k}')
