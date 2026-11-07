"""Find constant and duplicated raw feature columns of an encoder from its phase-1 XtX
(sum over 500 train chunks of per-chunk z-scored columns). Uses the delay-0 block.
usage: degenerate_scan.py <varpart_dir>"""
import os, sys, json
import numpy as np
vp = sys.argv[1]
sys.path.insert(0, vp)
import groups as G
meta = json.load(open(os.path.join(vp, 'stats', 'meta.json')))
names, o = [], 0
for g, d in [tuple(x) for x in meta['groups']]:
    names += [(g, i) for i in range(d)]
R = len(names)
X = np.load(os.path.join(vp, 'stats', 'XtX.npy'), mmap_mode='r')[:R, :R].astype(np.float64)
d = np.diag(X).copy()
nrows = meta['nresp']
const = np.where(d < 1e-6 * nrows)[0]
s = np.sqrt(np.where(d > 0, d, 1.0))
C = X / s[:, None] / s[None, :]
np.fill_diagonal(C, 0.0)
C[const, :] = 0; C[:, const] = 0
dup_pairs = np.argwhere(np.triu(np.abs(C) > 0.9999, 1))
by = {}
for i in const:
    by.setdefault(names[i][0], {'const': 0, 'dup': 0})['const'] += 1
seen = set()
for i, j in dup_pairs:
    if j in seen: continue
    seen.add(j)
    by.setdefault(names[j][0], {'const': 0, 'dup': 0})['dup'] += 1
print(f'raw cols {R}; constant {len(const)}; columns duplicating an earlier column (|r|>0.9999) {len(seen)}')
for g, v in by.items():
    print(f'  {g:28s} const={v["const"]:3d} dup={v["dup"]:3d}')
ex = {}
for i, j in dup_pairs:
    k = (names[j][0], names[i][0])
    if k not in ex: ex[k] = (names[j], names[i], round(float(C[i, j]), 6))
print('duplicate group pairs (dup-col group <- earlier group): example')
for k, v in ex.items():
    print(f'  {k[0]:26s} <- {k[1]:26s} e.g. {v}')
