import os, sys, json
import numpy as np
vp = sys.argv[1]
meta = json.load(open(os.path.join(vp, 'stats', 'meta.json')))
off, o = {}, 0
for g, d in [tuple(x) for x in meta['groups']]:
    off[g] = (o, d); o += d
X = np.load(os.path.join(vp, 'stats', 'XtX.npy'), mmap_mode='r')
def corr_block(a, b):
    A = np.asarray(X[np.ix_(a, a)]).diagonal(); B = np.asarray(X[np.ix_(b, b)]).diagonal()
    M = np.asarray(X[np.ix_(a, b)])
    return M / np.sqrt(A)[:, None] / np.sqrt(B)[None, :]
QS = (0, .25, -.25, .5, -.5); PH = ('cos', 'sin', 'mag')
qo = off['quad_modbank'][0]
idx = lambda r, s, b, p: qo + ((r * 5 + s) * 3 + b) * 3 + p
r12 = [idx(3, s, b, p) for s in range(5) for b in range(3) for p in range(3)]
r8 = [idx(2, s, b, p) for s in range(5) for b in range(3) for p in range(3)]
C = corr_block(r12, r8)
best = np.abs(C).max(1)
print('quad_modbank: 12 Hz column -> max |r| with any 8 Hz column')
for k, c in enumerate(r12):
    s, b, p = k // 9, (k // 3) % 3, k % 3
    print(f'  12Hz sc{QS[s]:+.2f} bin{b} {PH[p]}: {best[k]:.5f}')
so = off['stmf'][0]
STMF = [(2,.25),(2,-.25),(4,.25),(4,-.25),(4,.5),(4,-.5),(8,.25),(8,-.25),(8,.5),(8,-.5),(12,.25),(12,-.25),(4,0),(8,0),(12,0),(0,.25),(0,.5)]
for k12, k8 in ((10, 6), (11, 7), (14, 13)):
    a = [so + k12 * 20 + ch for ch in range(20)]; b = [so + k8 * 20 + ch for ch in range(20)]
    d = np.diag(corr_block(a, b))
    print(f'stmf {STMF[k12]} vs {STMF[k8]} same channel: |r| min {np.abs(d).min():.4f} median {np.median(np.abs(d)):.4f}')
