"""Glass brain: COMBINED LEAN-v20+Whisper vs Whisper (canonical test set).
Panels (left-lateral): combined R, Whisper R, and combined-minus-Whisper
(= where adding hand-built audio to Whisper helps)."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting

HERE = os.path.dirname(__file__)
ROOT = "/path/to/ecog_data"

r_comb = np.load(os.path.join(HERE, "diag_perelec_comb_v20.npz"))["r_comb"]
r_wh = np.load(os.path.join(HERE, "diag_perelec_canonical.npz"))["r_wh"]
adv = r_comb - r_wh

names = []
for line in open(os.path.join(ROOT, "channels.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) > 6 and p[6] == "good" and (
        p[0].startswith("G") or p[0].startswith(("dPI","dAI","dAMT","dMF","dPMT","dSMA","IT","IF"))):
        names.append(p[0])
coord = {}
for line in open(os.path.join(ROOT, "patient1_coords.txt")):
    t = line.split()
    if len(t) >= 4:
        try: coord[t[0]] = (float(t[1]), float(t[2]), float(t[3]))
        except ValueError: pass
node_coords = np.array([coord[n] for n in names])

vmax_r = max(r_comb.max(), r_wh.max())
fig, axes = plt.subplots(3, 1, figsize=(9, 11))
panels = [
    ("Combined LEAN-v20 + Whisper  R", r_comb, "OrRd", 0.0, vmax_r),
    ("Whisper encoder  R",             r_wh,   "OrRd", 0.0, vmax_r),
    ("Combined − Whisper  (where hand-built audio adds beyond Whisper)", adv, "RdBu_r", -abs(adv).max(), abs(adv).max()),
]
for ax, (title, vals, cmap, vmin, vmax) in zip(axes, panels):
    plotting.plot_markers(vals, node_coords, node_size=42, node_cmap=plt.get_cmap(cmap),
        node_vmin=vmin, node_vmax=vmax, display_mode="l", axes=ax, colorbar=True, title=title, alpha=0.95)
fig.suptitle("patient1: combined LEAN+Whisper vs Whisper, per-electrode R (left lateral, canonical test set)",
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(HERE, "glassbrain_combined_vs_whisper.png")
plt.savefig(out, dpi=160, facecolor="white"); print("saved", out)

def grp(sl): return f"comb {r_comb[sl].mean():.4f}  Whisper {r_wh[sl].mean():.4f}  C-W {adv[sl].mean():+.4f}"
print("ALL-166   :", grp(slice(0,166)))
print("GRID 0-124:", grp(slice(0,125)))
print("DEPTH     :", grp(slice(132,166)))
print("top-8 electrodes where combined beats Whisper most:")
for i in np.argsort(adv)[::-1][:8]:
    print(f"  {names[i]:7s} C-W={adv[i]:+.4f}  (Whisper {r_wh[i]:.3f} -> comb {r_comb[i]:.3f})  MNI {node_coords[i].round(1)}")
