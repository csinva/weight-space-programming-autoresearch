"""Glass-brain comparison: LEAN-v17 vs Whisper per-electrode R on patient1.
Three panels (left-lateral): LEAN R, Whisper R, Whisper-minus-LEAN advantage.
Tests whether Whisper's edge localizes (e.g. anterior/semantic vs posterior STG)."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting

import sys
HERE = os.path.dirname(__file__)
ROOT = "/path/to/ecog_data"

NPZ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "diag_perelec.npz")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "glassbrain_lean_vs_whisper.png")
# per-electrode R (model order: GRID 0-124, STRIPS 125-131, DEPTH 132-165)
d = np.load(NPZ)
r_lean, r_wh = d["r_lean"], d["r_wh"]
adv = r_wh - r_lean

# brain electrode names in model order
names = []
for line in open(os.path.join(ROOT, "channels.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) > 6 and p[6] == "good" and (
        p[0].startswith("G") or p[0].startswith(("dPI","dAI","dAMT","dMF","dPMT","dSMA","IT","IF"))):
        names.append(p[0])
assert len(names) == 166

coord = {}
for line in open(os.path.join(ROOT, "patient1_coords.txt")):
    t = line.split()
    if len(t) >= 4:
        try: coord[t[0]] = (float(t[1]), float(t[2]), float(t[3]))
        except ValueError: pass
node_coords = np.array([coord[n] for n in names])  # (166,3) MNI

fig, axes = plt.subplots(3, 1, figsize=(9, 11))
panels = [
    (os.environ.get("LEAN_LABEL", "LEAN") + " (hand-built audio)  R", r_lean, "OrRd", 0.0, max(r_lean.max(), r_wh.max())),
    ("Whisper encoder  R",             r_wh,   "OrRd", 0.0, max(r_lean.max(), r_wh.max())),
    ("Whisper − LEAN  (where Whisper wins)", adv, "RdBu_r", -abs(adv).max(), abs(adv).max()),
]
for ax, (title, vals, cmap, vmin, vmax) in zip(axes, panels):
    disp = plotting.plot_markers(
        vals, node_coords, node_size=42, node_cmap=plt.get_cmap(cmap),
        node_vmin=vmin, node_vmax=vmax, display_mode="l",
        axes=ax, colorbar=True, title=title, alpha=0.95)
fig.suptitle("patient1 audio→ECoG high-gamma: LEAN vs Whisper per-electrode R (left lateral)",
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT, dpi=160, facecolor="white")
print("saved", OUT)

# quick numeric context
def grp(sl): return f"LEAN {r_lean[sl].mean():.4f}  Whisper {r_wh[sl].mean():.4f}  W-L {adv[sl].mean():+.4f}"
print("GRID 0-124   :", grp(slice(0,125)))
print("STRIPS 125-31:", grp(slice(125,132)))
print("DEPTH 132-165:", grp(slice(132,166)))
print("top-8 Whisper-advantage electrodes:")
for i in np.argsort(adv)[::-1][:8]:
    print(f"  {names[i]:7s} W-L={adv[i]:+.4f}  (LEAN {r_lean[i]:.3f} Whisper {r_wh[i]:.3f})  MNI {node_coords[i].round(1)}")
