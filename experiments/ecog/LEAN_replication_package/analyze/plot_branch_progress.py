"""Progress-of-each-branch figure for the patient1 autoresearch loops.

Left:  absolute per-chunk per-electrode test_corr vs iteration, one line per branch,
       with each branch's ridge floor (dotted) and Whisper ceiling (dashed).
Right: normalized progress = fraction of each branch's floor->ceiling gap closed,
       which IS comparable across branches that use different test sets.

Reads each branch's results/overall_results.csv (reference rows excluded from the
iteration sequence). White background, tab10 colors (no viridis).
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/path/to/ecog_data/circuit-building-autoresearch"
OUT = os.path.join(REPO, "analyze", "branch_progress.png")

# (label, csv, floor_row_name, ceiling_row_name, color)  — both share the speech-rich test set
BRANCHES = [
    ("melspec · speech-rich",
     f"{REPO}/runs-ecog/ecog-speech-23608983/results/overall_results.csv",
     "Ridge-melspec-baseline", "Whisper-reference", "#1f77b4"),
    ("waveform · speech-rich",
     f"{REPO}/runs-wave/wave-slurm-23628821/results/overall_results.csv",
     "Wave-melspec-floor", "Whisper-reference", "#d62728"),
]


def read_branch(csv_path, floor_name, ceiling_name):
    floor = ceiling = None
    seq = []  # (model_name, test_corr) in CSV (try) order, reference rows excluded
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("model_shorthand_name", "")
            try:
                tc = float(row["test_corr"])
            except (TypeError, ValueError, KeyError):
                continue
            if name == floor_name:
                floor = tc
            elif name == ceiling_name:
                ceiling = tc
            else:
                seq.append((name, tc))
    return floor, ceiling, seq


def running_max(ys):
    out, best = [], float("-inf")
    for y in ys:
        best = max(best, y)
        out.append(best)
    return out


def _clean(name):
    n = name.replace("_b48", "").replace("Gammatone-", "").replace("_", " ")
    return n[:26]


# (substrings matched case-sensitively in the raw model name) -> human concept phrase
CONCEPTS = [
    (["cochleagram"], "gammatone cochleagram"),
    (["-log"], "log compression"),
    (["adapt"], "leaky-mean adaptation"),
    (["onset"], "onset/offset"),
    (["tau"], "multi-timescale adaptation"),
    (["win", "multiwin"], "multi-window analysis"),
    (["AM"], "AM depth"),
    (["loud"], "loudness"),
    (["centroid"], "spectral centroid"),
    (["HL", "highlevel", "full-stack"], "high-level feats (pitch/moments)"),
    (["flux", "Flux"], "spectral flux"),
    (["modbank", "modenv"], "modulation filterbank"),
    (["STMF"], "spectro-temporal modulation"),
    (["formant", "Form", "Fm", "FmProm"], "formants"),
    (["bw"], "formant bandwidth"),
    (["cube"], "cube-root compression"),
    (["chanvar"], "channel variance"),
    (["tonic"], "tonic (sustained) channels"),
    (["acausal"], "acausal / lookahead"),
    (["fut", "trend"], "slow trend"),
    # melspec branch
    (["Contrast"], "spectral contrast"),
    (["Rm", "Minus", "RunningMean"], "running-mean adaptation"),
    (["Relu", "Thresh"], "threshold rectifier"),
    (["Harm", "Comb"], "harmonic/pitch salience"),
    (["F0", "Argmax"], "F0 / pitch tracking"),
    (["Phon"], "phonetic templates"),
    (["Cross"], "cross-band products"),
    (["Moment", "Mom"], "spectral moments"),
    (["Sust"], "sustained energy"),
    (["Exc"], "peak/min excursion"),
    (["Var"], "variance"),
    (["Range"], "dynamic range"),
    (["Rank"], "rank transform"),
    (["Mem"], "temporal memory"),
    (["AGC"], "gain control"),
    (["HNR"], "harmonics-to-noise"),
    (["Accel"], "acceleration"),
    (["Prom"], "formant prominence"),
    (["Stab"], "formant stability"),
]


def _concepts(name):
    s = []
    for subs, phrase in CONCEPTS:
        if any(sub in name for sub in subs) and phrase not in s:
            s.append(phrase)
    return s


def commit_labels(names):
    """git-commit-style label per milestone = the concept(s) newly introduced vs the
    previous milestone. First milestone shows its base concept(s)."""
    out, prev = [], set()
    for i, nm in enumerate(names):
        cur = _concepts(nm)
        if i == 0:
            new = cur[:2] or [_clean(nm)]
        else:
            new = [c for c in cur if c not in prev]
        out.append((" + " if i else "") + ", ".join(new) if new else "(tuning)")
        prev = set(cur)
    return out


def _milestones(it, ys, names):
    """Keep the first new-best, any new-best that jumps >= +0.001 over the last kept,
    and the final best — so labels mark meaningful improvements, not every micro-best."""
    best = float("-inf")
    raw = []
    for x, y, nm in zip(it, ys, names):
        if y > best + 1e-9:
            best = y
            raw.append((x, y, nm))
    kept, last = [], -1.0
    for j, (x, y, nm) in enumerate(raw):
        if j == 0 or j == len(raw) - 1 or (y - last) >= 0.001:
            kept.append((x, y, nm))
            last = y
    return kept


def main():
    present = [b for b in BRANCHES if os.path.exists(b[1])]
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 5.5 * len(present)), sharey=True)
    if len(present) == 1:
        axes = [axes]

    for ax, (label, csv_path, fname, cname, color) in zip(axes, present):
        floor, ceiling, seq = read_branch(csv_path, fname, cname)
        if not seq:
            continue
        it = list(range(1, len(seq) + 1))
        ys = [tc for _, tc in seq]
        names = [nm for nm, _ in seq]
        rmax = running_max(ys)

        ax.plot(it, ys, "o", color=color, alpha=0.35, ms=4, zorder=3, label="attempts")
        ax.plot(it, rmax, "-", color=color, lw=2.4, zorder=4, label="running best")
        if floor is not None:
            ax.axhline(floor, color=color, ls=":", lw=1.3, alpha=0.8)
            ax.annotate(f"ridge floor {floor:.4f}", (len(seq) * 0.62, floor), color=color,
                        fontsize=8, va="bottom")
        # fair ceiling = Whisper WITH ndelays=4 (apples-to-apples; the branches use 4).
        # nd=1 Whisper is 0.0785; FIR delays lift it to 0.0821.
        ceiling = 0.0821
        ax.axhline(ceiling, color="k", ls="--", lw=1.3, alpha=0.6)
        ax.annotate(f"Whisper ceiling (ndelays=4) {ceiling:.4f}", (len(seq) * 0.4, ceiling),
                    color="k", fontsize=8.5, va="bottom")

        # milestone labels = git-commit-style description of what each update added
        ms = _milestones(it, ys, names)
        labels = commit_labels([nm for _, _, nm in ms])
        print(f"  [{label}] milestone commits:")
        for (x, y, nm), lab in zip(ms, labels):
            ax.plot(x, y, "o", color=color, ms=7, zorder=6)
            ax.annotate(f"{lab}  ({y:.4f})", (x, y), color=color, fontsize=8,
                        rotation=28, ha="left", va="bottom", zorder=7,
                        textcoords="offset points", xytext=(4, 6))
            print(f"      it{x:>3} {y:.4f}  {lab}")

        ax.set_title(f"{label}   (best {max(ys):.4f}, {len(seq)} iters)",
                     color=color, fontsize=12, loc="left")
        ax.set_ylabel("test_corr (r)")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.10)
        ax.legend(fontsize=8, loc="lower right", frameon=False)

    axes[-1].set_xlabel("iteration (order tried)")
    fig.suptitle("patient1 autoresearch — hand-written (no-training) audio→ECoG encoders, speech-rich test set\n"
                 "labels mark each milestone new-best · dotted = ridge floor · dashed = Whisper ceiling",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")
    for label, csv_path, fname, cname, color in BRANCHES:
        if os.path.exists(csv_path):
            floor, ceiling, seq = read_branch(csv_path, fname, cname)
            best = max((tc for _, tc in seq), default=float("nan"))
            print(f"  {label}: iters={len(seq)} floor={floor} ceiling={ceiling} best={best:.4f}")


if __name__ == "__main__":
    main()
