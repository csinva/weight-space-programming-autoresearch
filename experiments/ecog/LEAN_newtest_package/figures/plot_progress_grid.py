"""Figure 1: LEAN progress, every main-line trial on the 125 surface-grid electrodes.

Same layout as analyze/waveform_progress.png (June): encoder trial on x (chronological), test r
on y, faint dots for attempts, a running-best step line, milestone callouts and the Whisper line.
Jun 3-5 trials use their grid rescore from runs-wave/rescore-grid (only those that reproduced
their logged all-166 score); Jun 6+ trials were scored on the grid at the time. The clean-slate
side workers (reset-A..I) were not on the main line and are not shown, as in the June figure.
"""
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(HERE)
EXCLUDE = re.compile(r"refer|MEL-CTRL|baseline|melspec-floor", re.I)
WHISPER_GRID = 0.0933   # Whisper encoder features, same ridge + 4 FIR delays, grid electrodes

PHASES = [  # (header, runs) in chronological order along the LEAN line
    ("hand-design\nexploration", ["wave-slurm-23628821"]),
    ("trim + lit.\n→ v16", ["wave-trim", "wave-lean"]),
    ("graft +\ninteractions\n→ v17", ["graft-lean", "lean-interp", "exceed-dense"]),
    ("front end + scattering\n→ v18", ["exceed-frontend", "exceed-scatter", "exceed-combine"]),
    ("quad. banks\n+ gates\n→ v19, v20", ["exceed-v18"]),
    ("context,\ncoupling,\nspeaker", ["exceed-v20-context", "exceed-v20-deep", "exceed-v20-speaker"]),
]
GRID_RUNS = {"exceed-frontend", "exceed-scatter", "exceed-combine", "exceed-v18",
             "exceed-v20-context", "exceed-v20-deep", "exceed-v20-speaker"}

# (run, trial name, label, text x (trial), text y (r), ha) -- hand-placed in empty space
MILESTONES = [
    ("wave-slurm-23628821", "Gammatone-cochleagram", "gammatone cochleagram", 14, 0.0452, "left"),
    ("wave-slurm-23628821", "Gammatone-log-adapt", "+ log compression &\ndivisive adaptation", 26, 0.0545, "left"),
    ("wave-slurm-23628821", "Gammatone-3win-5tau", "+ multi-window ×\nmulti-τ adaptation", 28, 0.0835, "center"),
    ("wave-slurm-23628821", "Gammatone-BEST+HL+modbank+STMF14", "+ spectro-temporal\nmodulation filters", 50, 0.0895, "center"),
    ("wave-slurm-23628821", "Gammatone-BEST+full-stack+tonic3win+acausaltonic", "+ slow & future\ncontext", 72, 0.0715, "center"),
    ("wave-trim", "LEAN-v16", "v16", 122, 0.0775, "center"),
    ("lean-interp", "INTERP-v3e-interact", "v17: + interpretable\ninteractions", 215, 0.0795, "center"),
    ("exceed-combine", "COMBINE-v5-S2s+s1dyn", "v18: + scattering\n& spectral contrast", 372, 0.0800, "center"),
    ("exceed-v18", "LEAN-v18.q29-GATESIGS", "v19: + voicing gates", 440, 0.0765, "center"),
    ("exceed-v18", "LEAN-v18.q45-SLOWQUAD5SCALES", "v20: + slow\nmodulation", 490, 0.0805, "center"),
]


def rescored():
    out = {}
    for f in glob.glob(os.path.join(RW, "rescore-grid", "results", "*.json")):
        r = json.load(open(f))
        if r["status"] == "ok":
            out[(r["run"], r["name"], r["logged"])] = r["grid"]
    return out


def sequence():
    res, seen, seq, bounds = rescored(), set(), [], []
    n_logged = 0
    for header, runs in PHASES:
        start = len(seq)
        for run in runs:
            rows = list(csv.reader(open(os.path.join(RW, run, "results", "overall_results.csv"))))
            if rows and rows[0][0] == "subject":
                rows = rows[1:]
            for r in rows:
                if len(r) < 7 or EXCLUDE.search(r[6]) or (r[6], r[1]) in seen:
                    continue
                seen.add((r[6], r[1]))
                n_logged += 1
                v = float(r[1]) if run in GRID_RUNS else res.get((run, r[6], float(r[1])))
                if v is not None:
                    seq.append((run, r[6], v))
        bounds.append((header, start, len(seq)))
    return seq, bounds, n_logged


def main():
    seq, bounds, n_logged = sequence()

    n = len(seq)
    x = list(range(1, n + 1))
    y = [v for _, _, v in seq]
    best, b = [], -1.0
    for v in y:
        b = max(b, v); best.append(b)
    idx = {(run, name): i for i, (run, name, _) in enumerate(seq)}

    INK = "#222222"
    COLS = ["#2b6cb0", "#9c4221"]   # alternating phase tints, as in the June figure
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for k, (header, s, e) in enumerate(bounds):
        col = COLS[k % 2]
        ax.axvspan(s + 0.5, e + 0.5, color=col, alpha=0.05 if k % 2 == 0 else 0.06, lw=0)
        ax.scatter(x[s:e], y[s:e], s=12, color=col, alpha=0.32, lw=0, zorder=2)
        ax.text((s + e + 1) / 2, 0.0985, header, fontsize=8.6, color=col, ha="center",
                va="center", weight="bold", linespacing=1.05)
    ax.step(x, best, where="post", color=INK, lw=2.0, zorder=4)

    ax.axhline(WHISPER_GRID, color="#555555", ls=(0, (5, 3)), lw=1.1, zorder=1)
    ax.text(bounds[1][2] / 2 + 10, WHISPER_GRID + 0.0012,
            f"Whisper encoder features (same ridge, 4 delays)  {WHISPER_GRID:.4f}",
            fontsize=9.2, color="#555555", va="bottom", ha="center")

    for run, name, label, tx, ty, ha in MILESTONES:
        i = idx[(run, name)]
        ax.annotate(label, xy=(x[i], y[i]), xytext=(tx, ty), ha=ha,
                    va="center", fontsize=8.3, color=INK, linespacing=1.05,
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkB=2))

    peak = max(range(n), key=lambda i: y[i])
    ax.annotate(f"best hand-built\n{y[peak]:.4f}", xy=(x[peak], y[peak]),
                xytext=(x[peak] - 8, 0.0848), ha="center", va="center",
                fontsize=8.8, color=INK, weight="bold",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=3))

    ax.set_xlabel("encoder trial (chronological)", fontsize=10.5)
    ax.set_ylabel("test r, mean over 125 grid electrodes", fontsize=10.5)
    ax.set_title("Hand-built audio features (LEAN) for patient1 high-gamma: progress vs. Whisper",
                 fontsize=12, weight="bold", pad=10)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.040, 0.1035)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig1_progression.{ext}"), dpi=170, bbox_inches="tight")

    with open(os.path.join(HERE, "fig1_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "phase", "run", "name", "grid_r", "running_best"])
        for k, (header, s, e) in enumerate(bounds):
            for i in range(s, e):
                w.writerow([i + 1, header.replace("\n", " "), seq[i][0], seq[i][1], f"{y[i]:.5f}", f"{best[i]:.5f}"])
    print(f"plotted {n} main-line trials of {n_logged} logged; best {y[peak]:.4f} ({seq[peak][1]})")
    for header, s, e in bounds:
        print(f"  {header.replace(chr(10), ' '):40s} trials {s + 1}-{e} ({e - s})")


if __name__ == "__main__":
    main()
