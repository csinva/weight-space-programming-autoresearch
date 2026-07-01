"""Render the champion qq145's REAL per-voxel neural predictivity onto subject 1's
real cortical anatomy (NSD func1pt8mm space), for the talk deck.

Reconstructs each scored voxel's 3D position from the streams atlas in the SAME
np.isin C-order the prep used (early = streams==1, ventral = streams in {2,5}),
threads it through the identical ncsnr + constant-col selection the loader applies,
recomputes the champion's per-voxel test_corr exactly as harness.reward does, then
writes a func-space NIfTI and renders it with nilearn over the mean functional.

Self-checks: mean(test_corr) must be ~0.4407 and V1 ~0.645 (the DB values) or it aborts.
"""
from __future__ import annotations
import os, sys, importlib.util
os.environ.setdefault("NEURO_DATASET", "nsd")
os.environ.setdefault("NEURO_SUBJECT", "1")
os.environ.pop("NEURO_SUBJECTS", None)
os.environ.setdefault("NSD_NCSNR_MIN", "0.2")

import numpy as np
import torch
import nibabel as nib

from harness.datasets import DATASET
from harness.loaders.vision.nsd import _build_X, _nsd_roi_masks, _NC_R2_FLOOR
from harness.loaders.shared import surviving_cols, processed_dir
from harness.ridge import ridge_predict, _readout_device, _to_np
from harness.reward import _corr_per_column, _noise_normalize, _forward

SUBJ = "01"
OUT = "/home/t-ymehta/Research/talk/images/gen"
NC_DIR = os.path.join(processed_dir("nsd"), "ncsnr_src")
MODEL_PY = "runs/nsd__2026-06-27/models/qq145.py"

# ---- real voxel positions, in the prep's np.isin C-order -----------------------
streams_nii = nib.load(os.path.join(NC_DIR, f"subj{SUBJ}_streams.nii.gz"))
streams = streams_nii.get_fdata()
pos_all = np.concatenate([np.argwhere(np.isin(streams, [1])),       # early stream
                          np.argwhere(np.isin(streams, [2, 5]))], 0)  # ventral stream

blob = torch.load(os.path.join(processed_dir("nsd"), "neural", f"subj{SUBJ}.pt"), weights_only=False)
rois = blob["rois"]
early = np.asarray(rois["early"], "float32")
ventral = np.asarray(rois["ventral"], "float32")
Y = np.concatenate([early, ventral], 1)
n_early = early.shape[1]
assert pos_all.shape[0] == Y.shape[1], (pos_all.shape, Y.shape)
roi_masks = _nsd_roi_masks(rois, Y, n_early)

# ---- replicate the loader's two selection steps, threading positions -----------
nc = blob["ncsnr"]
ncsnr = np.concatenate([np.asarray(nc["early"], "float32"), np.asarray(nc["ventral"], "float32")])
nc_r2 = ncsnr ** 2 / (ncsnr ** 2 + 1.0 / 3)
thresh = float(os.environ["NSD_NCSNR_MIN"])
keep1 = (np.sqrt(nc_r2) >= thresh) & (nc_r2 >= _NC_R2_FLOOR)

shared = np.asarray(blob["shared"], bool)
nsd_ids = np.asarray(blob["nsd_ids"])
X = _build_X(DATASET, nsd_ids, SUBJ)
Xtr = torch.as_tensor(X[~shared]); Xte = torch.as_tensor(X[shared])
Ytr = torch.as_tensor(Y[~shared]); Yte = torch.as_tensor(Y[shared])

k1 = torch.as_tensor(keep1)
Ytr, Yte = Ytr[:, k1], Yte[:, k1]
nc_r2, pos = nc_r2[keep1], pos_all[keep1]
roi_masks = {k: np.asarray(v, bool)[keep1] for k, v in roi_masks.items()}

keep2 = surviving_cols(Ytr)
k2 = keep2.numpy() if torch.is_tensor(keep2) else np.asarray(keep2)
Ytr, Yte = Ytr[:, torch.as_tensor(k2)], Yte[:, torch.as_tensor(k2)]
nc_r2, pos = nc_r2[k2], pos[k2]
roi_masks = {k: v[k2] for k, v in roi_masks.items()}

# ---- recompute the champion's per-voxel predictivity (exactly as the reward) ---
spec = importlib.util.spec_from_file_location("qq145", MODEL_PY)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
model = mod.MODEL

dev = _readout_device()
A_train, A_test = _forward(model, Xtr.to(dev)), _forward(model, Xte.to(dev))
pred_train, pred_test = ridge_predict(A_train, A_test, Ytr, device=dev)
test_np = _noise_normalize(_corr_per_column(_to_np(pred_test), _to_np(Yte)), nc_r2)

mean_corr = float(test_np.mean())
print(f"mean test_corr = {mean_corr:.4f}  (expect ~0.4407)   n_voxels = {test_np.size}")
for r in ["V1", "V2", "V3", "hV4", "PPA", "FFA", "VWFA", "early", "ventral"]:
    print(f"   {r:8} {float(test_np[roi_masks[r]].mean()):.3f}")
assert abs(mean_corr - 0.4407) < 0.01, f"alignment check FAILED: {mean_corr}"
assert abs(float(test_np[roi_masks['V1']].mean()) - 0.645) < 0.02, "V1 check FAILED"
print("OK — alignment verified against the DB values.")

# ---- write a real func-space predictivity NIfTI --------------------------------
vol = np.zeros(streams.shape, dtype="float32")
vol[pos[:, 0], pos[:, 1], pos[:, 2]] = test_np
stat_img = nib.Nifti1Image(vol, streams_nii.affine, streams_nii.header)
nib.save(stat_img, os.path.join(OUT, "cortex_real.nii.gz"))

# ---- render on the subject's real anatomy --------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting

PAPER, INK = "#FAF8F3", "#1A1A17"
# deck colormap: low predictivity orange -> teal -> high blue
cmap = LinearSegmentedColormap.from_list(
    "deck", ["#E8943D", "#E8C39A", "#7FBAA6", "#4FA08B", "#3B6EAA", "#2E5687"])
bg = os.path.join(NC_DIR, f"subj{SUBJ}_mean.nii.gz")

# sequential 0.08 -> 0.65 (predictivity is positive; don't waste range on negatives)
SEQ = dict(cmap=cmap, vmin=0.08, vmax=0.65, threshold=0.08, symmetric_cbar=False)

# (a) axial slices through occipital + ventral-temporal cortex
fig = plt.figure(figsize=(15, 4.6), facecolor=PAPER)
plotting.plot_stat_map(
    stat_img, bg_img=bg, display_mode="z", cut_coords=[-18, -12, -6, 0, 6, 14],
    black_bg=False, annotate=True, colorbar=True, figure=fig,
    draw_cross=False, title=None, **SEQ)
fig.savefig(os.path.join(OUT, "cortex_real_axial.png"), dpi=170,
            transparent=True, bbox_inches="tight")
plt.close(fig)

# (b) glass-brain: iconic posterior->anterior gradient, three projections
fig = plt.figure(figsize=(15, 5.4), facecolor=PAPER)
plotting.plot_glass_brain(
    stat_img, display_mode="lzr", colorbar=True, black_bg=False, figure=fig,
    plot_abs=False, **SEQ)
fig.savefig(os.path.join(OUT, "cortex_real_glass.png"), dpi=170,
            transparent=True, bbox_inches="tight")
plt.close(fig)

print("wrote cortex_real_axial.png + cortex_real_glass.png + cortex_real.nii.gz")

# ============================================================================
# (c) REAL champion feature maps on a real stimulus — S1 Gabor energy + C1
# ============================================================================
from PIL import Image
import torch.nn.functional as F
featcmap = LinearSegmentedColormap.from_list("feat", ["#FAF8F3", "#7FBAA6", "#3B6EAA", "#1A1A17"])
stim = np.asarray(Image.open(os.path.join(OUT, "stim", "stim3.png")).convert("RGB").resize((128, 128)),
                  dtype="float32") / 255.0
xt = torch.from_numpy(stim.reshape(1, -1)).to(dev)
model._to(dev)
xim = xt.reshape(1, 128, 128, 3).permute(0, 3, 1, 2).float()
xn = (xim - xim.mean((1, 2, 3), keepdim=True)) / xim.std((1, 2, 3), keepdim=True).clamp_min(1e-6)
lum = xn.mean(1, keepdim=True)
S1 = model._energy(lum, model.E1, model.O1)            # (1,40,H,W) = 5 scales x 8 orient
sc = 2                                                  # scale lambda=4 -> indices 16..23
s1 = _to_np(S1[0, sc * 8:sc * 8 + 8])                  # 8 orientation energy maps
C1 = _to_np(F.max_pool2d(S1[:, sc * 8:sc * 8 + 8], 6, 4)[0])  # HMAX complex-cell pooling

oris = [0, 2, 4, 6]
labels = ["0°", "45°", "90°", "135°"]
fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), facecolor="none")
for a in axes.ravel():
    a.axis("off")
axes[0, 0].imshow(stim); axes[0, 0].set_title("real stimulus", fontsize=17, color="#1A1A17")
for k, o in enumerate(oris):
    axes[0, k + 1].imshow(s1[o], cmap=featcmap)
    axes[0, k + 1].set_title(labels[k], fontsize=14, color="#5C5C57")
    axes[1, k + 1].imshow(C1[o], cmap=featcmap)
fig.text(0.635, 0.965, "S1 · Gabor energy", ha="center", fontsize=16, color="#1A1A17")
fig.text(0.635, 0.475, "C1 · complex cells (position-tolerant)", ha="center", fontsize=16, color="#1A1A17")
fig.savefig(os.path.join(OUT, "model_features.png"), dpi=170, transparent=True, bbox_inches="tight")
plt.close(fig)

# ============================================================================
# (d) REAL predicted-vs-measured held-out responses for example voxels
# ============================================================================
pred_np, Y_np = _to_np(pred_test), _to_np(Yte)
raw_r = _corr_per_column(pred_np, Y_np)                # honest per-voxel Pearson r
panels = [("V1", "#3B6EAA"), ("hV4", "#4FA08B"), ("FFA", "#E8943D")]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="none")
for ax, (roi, col) in zip(axes, panels):
    idx = np.where(roi_masks[roi])[0]
    j = idx[np.argsort(raw_r[idx])[int(0.80 * len(idx))]]   # a representative well-predicted voxel
    # z-score both axes so the cloud's tilt reflects r (ridge shrinks predicted variance)
    x = (Y_np[:, j] - Y_np[:, j].mean()) / Y_np[:, j].std()
    y = (pred_np[:, j] - pred_np[:, j].mean()) / pred_np[:, j].std()
    ax.scatter(x, y, s=10, c=col, alpha=0.40, edgecolors="none")
    ax.plot([-3, 3], [-3, 3], color="#5C5C57", lw=1, ls="--")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_title(f"{roi}   ·   Pearson r = {float(raw_r[j]):.2f}", fontsize=17, color="#1A1A17")
    ax.set_xlabel("measured response (z)", fontsize=13, color="#5C5C57")
    if roi == "V1":
        ax.set_ylabel("predicted response (z)", fontsize=13, color="#5C5C57")
    for s in ax.spines.values():
        s.set_color("#D9D5C8")
    ax.tick_params(colors="#5C5C57", labelsize=10)
    ax.set_aspect("equal"); ax.set_facecolor("none")
fig.savefig(os.path.join(OUT, "pred_scatter.png"), dpi=170, transparent=True, bbox_inches="tight")
plt.close(fig)
print("wrote model_features.png + pred_scatter.png")
