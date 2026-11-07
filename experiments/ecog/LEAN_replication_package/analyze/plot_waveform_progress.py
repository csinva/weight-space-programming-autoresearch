"""Waveform branch progress: one clean running-best trajectory.

Reads the two waveform run folders (the main exploration folder + the lean
trim/lit-review folder), builds a chronological trial sequence, and plots the
running-best test_corr against the melspec floor and the Whisper ceiling.

Deliberately sparse text (3 milestone callouts + 2 reference labels, all in
clear whitespace) so nothing overlaps.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/path/to/ecog_data/circuit-building-autoresearch"
FAT = f"{REPO}/runs-wave/wave-slurm-23628821/results/overall_results.csv"
LEAN = f"{REPO}/runs-wave/wave-lean/results/overall_results.csv"
OUT = f"{REPO}/analyze/waveform_progress.png"

REF_KEYS = ("whisper", "melspec", "floor", "reference")
MELSPEC_FLOOR = 0.0457
WHISPER_CEIL = 0.0821  # Whisper features with ndelays=4 FIR (apples-to-apples vs branches)


def trials(path):
    out = []
    for r in csv.DictReader(open(path)):
        name = r["model_shorthand_name"]
        if not r["test_corr"]:
            continue
        if any(k in name.lower() for k in REF_KEYS):
            continue
        out.append((name, float(r["test_corr"])))
    return out


seq = trials(FAT) + trials(LEAN)
n = len(seq)
x = list(range(1, n + 1))
y = [v for _, v in seq]

# running best
best = []
b = -1.0
for v in y:
    b = max(b, v)
    best.append(b)

n_fat = len(trials(FAT))
peak_i = max(range(n), key=lambda i: y[i])
final_best = best[-1]

fig, ax = plt.subplots(figsize=(11.5, 5.4))
INK = "#222222"
EXPLORE = "#2b6cb0"   # blue
PLATEAU = "#9c4221"   # rust

# phase backgrounds (very light)
ax.axvspan(0.5, n_fat + 0.5, color="#2b6cb0", alpha=0.05, lw=0)
ax.axvspan(n_fat + 0.5, n + 0.5, color="#9c4221", alpha=0.06, lw=0)

# individual trials (faint dots)
ax.scatter(x[:n_fat], y[:n_fat], s=14, color=EXPLORE, alpha=0.30, lw=0, zorder=2)
ax.scatter(x[n_fat:], y[n_fat:], s=14, color=PLATEAU, alpha=0.40, lw=0, zorder=2)

# running-best envelope
ax.step(x, best, where="post", color=INK, lw=2.0, zorder=4)

# reference lines, labels parked in clear space at the LEFT (curve is high on the right)
ax.axhline(WHISPER_CEIL, color="#555555", ls=(0, (5, 3)), lw=1.1, zorder=1)
ax.text(92, WHISPER_CEIL + 0.0016, "Whisper 4-lag FIR (learned-feature ceiling)  0.082",
        fontsize=9.5, color="#555555", va="bottom", ha="center")
ax.axhline(MELSPEC_FLOOR, color="#888888", ls=(0, (5, 3)), lw=1.1, zorder=1)
ax.text(n - 2, MELSPEC_FLOOR - 0.0014, "mel-spectrogram floor  0.046",
        fontsize=9.5, color="#888888", va="top", ha="right")

# phase headers (top, centered in each span, well above any data)
ytop = 0.0905
ax.text((n_fat + 1) / 2, ytop, "hand-design exploration  (155 trials)",
        fontsize=10.5, color=EXPLORE, ha="center", va="center", weight="bold")
ax.text(n_fat + (n - n_fat) / 2, ytop, "trim + literature\nfeatures (plateau)",
        fontsize=10.5, color=PLATEAU, ha="center", va="center", weight="bold")

# concept callouts for the MAJOR record-setting jumps only.
# (trial# 1-based, label, text_x, text_y, ha) -- positions hand-placed in the
# empty upper-left (early climb) and empty lower band (mid/late gains).
MILESTONES = [
    (1,   "gammatone\ncochleagram  0.040", 6,   0.0395, "left"),
    (3,   "+ log compression\n& divisive adaptation", 8, 0.0820, "left"),
    (19,  "+ multi-window ×\nmulti-τ adaptation", 38, 0.0735, "left"),
    (37,  "+ spectral flux", 30, 0.0500, "center"),
    (41,  "+ modulation\nfilterbank", 50, 0.0430, "center"),
    (53,  "+ spectro-temporal\nmodulation filters", 72, 0.0515, "center"),
    (106, "+ tonic (slow)\ncontext", 98, 0.0610, "center"),
    (113, "+ acausal (future)\ncontext", 126, 0.0645, "center"),
]
for trial, label, tx, ty, ha in MILESTONES:
    i = trial - 1
    ax.annotate(label, xy=(x[i], y[i]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkB=2))

# peak / final marker — parked in the clear gap between the plateau and the ceiling
ax.annotate(f"best hand-crafted  {y[peak_i]:.4f}", xy=(x[peak_i], y[peak_i]),
            xytext=(n + 0.5, 0.0792), ha="right", va="center",
            fontsize=9.0, color=INK, weight="bold",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=3))

ax.set_xlabel("encoder trial (chronological)", fontsize=11)
ax.set_ylabel("per-chunk per-electrode test r", fontsize=11)
ax.set_title("Hand-designed waveform→ECoG encoder: progress vs. ceilings (patient1)",
             fontsize=12.5, weight="bold", pad=10)
ax.set_xlim(0.5, n + 0.5)
ax.set_ylim(0.035, 0.094)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9.5)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"trials: {n} ({n_fat} explore + {n - n_fat} lean)  peak={y[peak_i]:.4f}@{peak_i+1}  final_best={final_best:.4f}")
