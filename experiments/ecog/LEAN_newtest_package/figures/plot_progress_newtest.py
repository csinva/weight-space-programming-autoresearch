"""Figure 1, new held-out test: every main-line trial rerun and scored on chunks the search never saw.

Same layout as plot_progress_grid.py (trial on x in chronological order, grid r on y, phase bands).
Dots and the black step line are NEW-test scores (317-chunk primary set, runs-wave/newtest). The
black line follows the model the search actually kept, i.e. the running best by OLD-test score,
scored on the new test; the faint gray line is that same running best on the old test (what the
search saw). The gap between the two lines is the test-set effect; the dots show whether the
ranking held. Only trials that reproduced their logged old-test score are plotted.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_progress_grid import PHASES

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.join(os.path.dirname(HERE), "newtest")
INK, OLD = "#222222", "#9a9a9a"
COLS = ["#2b6cb0", "#9c4221"]   # alternating phase tints, as in Figure 1

# (trial name, label, text dx in trials, text y, ha)
MILESTONES = [
    ("Gammatone-cochleagram", "gammatone cochleagram", 10, 0.0345, "left"),
    ("Gammatone-log-adapt", "+ log compression &\ndivisive adaptation", 22, 0.0425, "left"),
    ("Gammatone-3win-5tau", "+ multi-window ×\nmulti-τ adaptation", 45, 0.0490, "left"),
    ("Gammatone-BEST+HL+modbank+STMF14", "+ spectro-temporal\nmodulation filters", 40, 0.0585, "left"),
    ("LEAN-v16", "v16", 0, 0.0610, "center"),
    ("INTERP-v3e-interact", "v17", 0, 0.0625, "center"),
    ("COMBINE-v5-S2s+s1dyn", "v18", 0, 0.0640, "center"),
    ("LEAN-v18.q29-GATESIGS", "v19", -6, 0.0655, "center"),
    ("LEAN-v18.q45-SLOWQUAD5SCALES", "v20", 8, 0.0655, "center"),
]


def main():
    trials = json.load(open(os.path.join(NT, "trials.json")))
    seq = []
    for t in trials:
        p = os.path.join(NT, "results", t["id"] + ".json")
        if os.path.exists(p):
            r = json.load(open(p))
            if r["status"] == "ok":
                seq.append((t, r))
    n_mis = sum(1 for t in trials if os.path.exists(os.path.join(NT, "results", t["id"] + ".json"))
                and json.load(open(os.path.join(NT, "results", t["id"] + ".json")))["status"] != "ok")
    n_missing = len(json.load(open(os.path.join(NT, "trials_missing.json"))))
    wh = json.load(open(os.path.join(NT, "results", "whisper-reference.json")))

    n = len(seq)
    x = np.arange(1, n + 1)
    old = np.array([r["old_grid"] for _, r in seq])
    new = np.array([r["new_grid"] for _, r in seq])
    pick = np.array([int(np.argmax(old[:i + 1])) for i in range(n)])
    old_best, new_of_pick = old[pick], new[pick]
    idx = {t["name"]: i for i, (t, _) in enumerate(seq)}

    bounds, cur = [], None
    for i, (t, _) in enumerate(seq):
        if t["phase"] != cur:
            bounds.append([t["phase"], i, i + 1]); cur = t["phase"]
        bounds[-1][2] = i + 1
    header = {h.replace("\n", " "): h for h, _ in PHASES}

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for k, (ph, s, e) in enumerate(bounds):
        col = COLS[k % 2]
        ax.axvspan(s + 0.5, e + 0.5, color=col, alpha=0.05 if k % 2 == 0 else 0.06, lw=0)
        ax.scatter(x[s:e], new[s:e], s=12, color=col, alpha=0.32, lw=0, zorder=2)
        ax.text((s + e + 1) / 2, 0.1005, header.get(ph, ph), fontsize=8.6, color=col, ha="center",
                va="center", weight="bold", linespacing=1.05)

    ax.step(x, old_best, where="post", color=OLD, lw=1.6, zorder=3, alpha=0.9)
    ax.step(x, new_of_pick, where="post", color=INK, lw=2.0, zorder=4)
    ax.axhline(wh["old_grid"], color=OLD, ls=(0, (5, 3)), lw=1.0, zorder=1, alpha=0.9)
    ax.axhline(wh["new_grid"], color="#555555", ls=(0, (5, 3)), lw=1.1, zorder=1)

    xr = n + 6
    lab = dict(fontsize=8.6, va="center", ha="left", clip_on=False)
    ax.text(xr, wh["old_grid"] + 0.0004, f"Whisper, old test  {wh['old_grid']:.4f}", color="#7a7a7a", **lab)
    ax.text(xr, old_best[-1] - 0.0006, f"kept model, old test  {old_best[-1]:.4f}", color="#7a7a7a", **lab)
    ax.text(xr, wh["new_grid"], f"Whisper, new test  {wh['new_grid']:.4f}", color="#444444", **lab)
    ax.text(xr, new_of_pick[-1], f"kept model, new test  {new_of_pick[-1]:.4f}", color=INK,
            weight="bold", **lab)

    for name, text, dx, ty, ha in MILESTONES:
        i = idx[name]
        ax.annotate(text, xy=(x[i], new[i]), xytext=(x[i] + dx, ty), ha=ha, va="center",
                    fontsize=8.3, color=INK, linespacing=1.05,
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkB=2))

    ax.set_xlabel("encoder trial (chronological)", fontsize=10.5)
    ax.set_ylabel("test r, mean over 125 grid electrodes", fontsize=10.5)
    ax.set_title("LEAN progress rerun on a new held-out test set (317 chunks never used in the search)",
                 fontsize=12, weight="bold", pad=12)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.030, 0.1035)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig1_progression_newtest.{ext}"), dpi=170, bbox_inches="tight")

    with open(os.path.join(HERE, "fig1_newtest_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "phase", "run", "name", "old_grid_r", "new_grid_r", "new_all493_grid_r",
                    "kept_model", "kept_old_grid_r", "kept_new_grid_r"])
        for i, (t, r) in enumerate(seq):
            w.writerow([i + 1, t["phase"], t["run"], t["name"], f"{old[i]:.5f}", f"{new[i]:.5f}",
                        f"{r['newall_grid']:.5f}", seq[pick[i]][0]["name"], f"{old_best[i]:.5f}",
                        f"{new_of_pick[i]:.5f}"])
    print(f"plotted {n} trials ({n_mis} not reproduced, {n_missing} without code); "
          f"corr(old, new) = {np.corrcoef(old, new)[0, 1]:.3f}; kept model {seq[pick[-1]][0]['name']} "
          f"old {old_best[-1]:.4f} new {new_of_pick[-1]:.4f}; best new {new.max():.4f}")


if __name__ == "__main__":
    main()
