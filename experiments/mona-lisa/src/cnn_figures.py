"""Explanatory figures for the Mona Lisa convnet."""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root = os.path.join(os.path.dirname(__file__), '..')
Z = np.load(os.path.join(root, 'cnn_weights.npz')); meta = json.load(open(os.path.join(root, 'cnn_weights.json')))
order = meta['order']; W, H = meta['W'], meta['H']
flat = np.concatenate([Z[nm].ravel() for nm in order]); lo, hi = flat.min(), flat.max()
# 1) anatomy of memory filters: painting to the eye, a digit in the pair differences
mem = Z['conv3_memory']
ids = [5*4096 + 7, 2*4096 + 11, 8*4096 + 3]
fig, ax = plt.subplots(2, 3, figsize=(10, 5.2))
for j, i in enumerate(ids):
    f = mem[i, 0]
    ax[0, j].imshow(f, cmap='gray', vmin=lo, vmax=hi); ax[0, j].set_title(f'memory filter #{i}: weights (28x56)', fontsize=8)
    ax[1, j].imshow(0.5*(f[:, 0::2] - f[:, 1::2]), cmap='gray'); ax[1, j].set_title('(left - right)/2 of each weight pair', fontsize=8)
for a in ax.flat: a.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(root, 'figures', 'cnn_filter_anatomy.png'), dpi=120); plt.close()
# 2) where each layer lives in the picture
sizes = [Z[nm].size for nm in order]; starts = np.cumsum([0] + sizes[:-1])
T = 28*56
lines = [f'{k+1:2d} {nm:14s} {str(Z[nm].shape):20s} {sz:>11,d} weights   tiles {s//T:>5d} - {(s+sz-1)//T:>5d}' for k, (nm, sz, s) in enumerate(zip(order, sizes, starts))]
open(os.path.join(root, 'figures', 'cnn_layer_map.txt'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
