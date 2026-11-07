"""Evaluate a text embedder as an fMRI language encoding model.

The "task" here is predicting held-out fMRI responses to language (Huth dataset).
An embedder maps each 10-gram to a fixed vector; those vectors become features
(Lanczos-downsampled to TRs, z-scored, FIR-delayed) and a ridge model is fit to
predict voxel responses. The metric is the mean test-set correlation.

This file is the fixed evaluation harness — do NOT edit it. Different embedders
plug in: the GPT-2 XL baseline (`src/baseline.py`) or the hand-built
interpretable transformer (`interpretable_transformer.py`).

Usage:
    from src.eval import EncodingConfig, run_encoding
    r = run_encoding(embedder, EncodingConfig(num_train=5))
    print(r['test_corr'])
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from . import data, encoding, features

# candidate popular language / semantic ROIs to report the test correlation within
# (Broca's area, auditory cortex, speech-motor, body/face/place/scene areas, IPS)
_CANDIDATE_ROIS = ["Broca", "AC", "sPMv", "EBA", "FFA", "PPA", "RSC", "IPS"]


def _popular_rois() -> list[str]:
    """Keep only candidate ROIs present for every main subject that has an ROI file.

    S01 has no ROI file (so it is skipped); S02 and S03 do. Intersecting across the
    subjects that have ROI data guarantees the reported roi_* columns are
    consistently defined wherever ROIs exist (and drops any non-shared ROI).
    """
    roi_sets = [set(data.load_rois(s)) for s in ("UTS01", "UTS02", "UTS03")]
    roi_sets = [s for s in roi_sets if s]  # subjects without an ROI file contribute nothing
    if not roi_sets:
        return []
    common = set.intersection(*roi_sets)
    return [r for r in _CANDIDATE_ROIS if r in common]


POPULAR_ROIS = _popular_rois()

OVERALL_CSV_COLS = (
    ["subject", "test_corr", "train_corr", "frac_test_voxels_above_0.2", "encoding_seconds",
     "status", "model_shorthand_name", "n_params", "description"]
    + [f"roi_{name}" for name in POPULAR_ROIS]
)


@dataclass
class EncodingConfig:
    """Shared config so the baseline and every interpretable attempt are comparable."""
    subject: str = "UTS03"
    num_train: int = 5
    num_test: int = 1
    ngram_size: int = 10
    ndelays: int = 4
    nboots: int = 5
    chunklen: int = 40
    nchunks: int = 20
    trim_edges: bool = True  # drop first/last 30 TRs of every story
    edge_trim_trs: int = 30


def run_encoding(embedder, cfg: EncodingConfig, verbose: bool = True) -> dict:
    """Fit + evaluate `embedder` as an encoding model. Returns a results dict.

    `embedder(texts) -> np.ndarray (n_texts, hidden_dim)` is the only thing that
    varies between attempts.
    """
    t0 = time.time()
    train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
    if verbose:
        print(f'train stories ({len(train_stories)}):', train_stories)
        print(f'test stories ({len(test_stories)}):', test_stories)
    all_stories = train_stories + test_stories

    if verbose:
        print('loading wordseqs + responses...')
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=cfg.subject)

    extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0
    if verbose and extra_trim:
        print(f'trimming first/last {extra_trim} TRs of every story')

    if verbose:
        print('extracting features...')
    stim_train = features.get_features(
        wordseqs, train_stories, embedder, ngram_size=cfg.ngram_size,
        ndelays=cfg.ndelays, extra_trim=extra_trim)
    stim_test = features.get_features(
        wordseqs, test_stories, embedder, ngram_size=cfg.ngram_size,
        ndelays=cfg.ndelays, extra_trim=extra_trim)
    if extra_trim:
        resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    if verbose:
        print('feature shapes', stim_train.shape, stim_test.shape)
        print('response shapes', resp_train.shape, resp_test.shape)
    assert stim_train.shape[0] == resp_train.shape[0], 'train feat/resp TR mismatch'
    assert stim_test.shape[0] == resp_test.shape[0], 'test feat/resp TR mismatch'

    if verbose:
        print('fitting ridge encoding model...')
    r = encoding.fit_encoding(
        stim_train, resp_train, stim_test, resp_test,
        nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks)
    r['test_corr'] = r['corrs_test_mean']  # primary metric
    r['train_stories'] = train_stories
    r['test_stories'] = test_stories
    r['subject'] = cfg.subject

    # mean test correlation within each popular ROI
    rois = data.load_rois(cfg.subject)
    corrs_test = r['corrs_test']
    roi_corrs = {}
    for name in POPULAR_ROIS:
        if name in rois:
            idx = rois[name]
            idx = idx[idx < len(corrs_test)]
            roi_corrs[name] = float(np.nanmean(corrs_test[idx])) if len(idx) else float('nan')
    r['roi_corrs'] = roi_corrs

    r['encoding_seconds'] = time.time() - t0
    if verbose:
        print(f"train_corr={r['corrs_train_mean']:.4f}  test_corr={r['test_corr']:.4f}  "
              f"frac>0.2={r['corrs_test_frac>0.2']:.4f}  ({r['encoding_seconds']:.1f}s)")
    return r


def make_result_row(r: dict, model_shorthand_name: str, n_params: int,
                    description: str, status: str = "success") -> dict:
    """Build a CSV row (all OVERALL_CSV_COLS) from a run_encoding results dict."""
    row = {
        "subject":               r["subject"],
        "test_corr":             f"{r['test_corr']:.4f}",
        "train_corr":            f"{r['corrs_train_mean']:.4f}",
        "frac_test_voxels_above_0.2": f"{r['corrs_test_frac>0.2']:.4f}",
        "encoding_seconds":      f"{r['encoding_seconds']:.1f}",
        "status":                status,
        "model_shorthand_name":  model_shorthand_name,
        "n_params":              f"{n_params:.2e}",
        "description":           description,
    }
    roi_corrs = r.get('roi_corrs', {})
    for name in POPULAR_ROIS:
        v = roi_corrs.get(name)
        row[f"roi_{name}"] = f"{v:.4f}" if v is not None and not np.isnan(v) else ""
    return row


def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    """Append/update rows in overall_results.csv, keyed by (model_shorthand_name, subject)."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["subject"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("subject")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")


def plot_corr_over_iterations(results_dir: str) -> None:
    """Read overall_results.csv in CSV order and plot test_corr + running max."""
    csv_path = os.path.join(results_dir, "overall_results.csv")
    if not os.path.exists(csv_path):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                corr = float(row["test_corr"])
            except (TypeError, ValueError):
                continue
            rows.append((row.get("model_shorthand_name", ""), corr))
    if not rows:
        return

    iters = list(range(1, len(rows) + 1))
    names = [n for n, _ in rows]
    corrs = [c for _, c in rows]
    is_baseline = ["gpt" in n.lower() for n in names]

    # Running max over non-baseline points only.
    running_iters, running_max, best = [], [], float("-inf")
    increases_max = []  # True for points that strictly increase the running max.
    for it, c, base in zip(iters, corrs, is_baseline):
        if base:
            increases_max.append(False)
            continue
        is_new_max = c > best
        best = max(best, c)
        running_iters.append(it)
        running_max.append(best)
        increases_max.append(is_new_max)

    fig, ax = plt.subplots(figsize=(16, 5))
    # Non-baseline points: connected line with circle markers.
    nb_iters = [it for it, base in zip(iters, is_baseline) if not base]
    nb_corrs = [c for c, base in zip(corrs, is_baseline) if not base]
    ax.plot(nb_iters, nb_corrs, marker="o", linestyle="-", color="steelblue", label="test_corr")
    # Baseline points: black squares, not part of the running max.
    bl_iters = [it for it, base in zip(iters, is_baseline) if base]
    bl_corrs = [c for c, base in zip(corrs, is_baseline) if base]
    if bl_iters:
        ax.scatter(bl_iters, bl_corrs, marker="s", color="black", zorder=5, label="baseline")
    if running_iters:
        ax.plot(running_iters, running_max, drawstyle="steps-post", color="crimson",
                linewidth=2, label="running max")
    # Annotate only points that increase the running max, rotated vertically.
    for it, c, name, inc in zip(iters, corrs, names, increases_max):
        if not inc:
            continue
        ax.annotate(name, (it, c), rotation=90, fontsize='xx-small',
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", va="bottom")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean test correlation")
    ax.set_title("Encoding test correlation over iterations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if len(iters) <= 30:
        ax.set_xticks(iters)
    fig.tight_layout()
    out_path = os.path.join(results_dir, "corr_over_iterations.pdf")
    fig.savefig(out_path) #, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_path}")
