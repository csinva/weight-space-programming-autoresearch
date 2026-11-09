"""Visualise the 16 weight matrices in a 4x4 grid, in layer order (layer 1 top-left)."""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'weights.npz')
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(outdir, exist_ok=True)
Z = np.load(path)
Ws = [Z[f'W{l}'] for l in range(16)]
# 1) the naive way a person would look at it: plt.subplots + imshow, default per-tile autoscaling
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for l, ax in enumerate(axes.flat):
    ax.imshow(Ws[l], cmap='gray'); ax.set_title(f'layer {l+1}: W{l+1} {Ws[l].shape}', fontsize=8); ax.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(outdir, 'grid_subplots.png'), dpi=110); plt.close()
# 2) full-resolution mosaic with one shared colour scale (every pixel is one weight)
M = np.block([[Ws[4*i+j] for j in range(4)] for i in range(4)])
lo, hi = M.min(), M.max()
img = ((M - lo)/(hi - lo)*255).clip(0, 255).astype(np.uint8)
Image.fromarray(img).save(os.path.join(outdir, 'mosaic_fullres.png'))
Image.fromarray(img).resize((1024, 1024), Image.LANCZOS).save(os.path.join(outdir, 'mosaic_1024.png'))
print('saved to', outdir, 'range', lo, hi)
