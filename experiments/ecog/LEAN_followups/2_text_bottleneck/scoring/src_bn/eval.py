"""Evaluate an audio embedder as a melspec -> ECoG high-gamma encoding model.

The "task" is predicting held-out high-gamma ECoG for patient patient1 from the
audio mel-spectrogram. An embedder maps each chunk's melspec (T, 50) to a
per-frame embedding (T, H); those become features (FIR-delayed, lagged, trimmed,
z-scored) and a bootstrapped ridge model is fit to predict the 166 electrodes.

The metric is the project's canonical **per-sample per-electrode correlation**:
for each held-out chunk, the Pearson r between predicted and actual high-gamma is
computed per electrode and averaged over electrodes; the chunk values are then
averaged. This matches `per_sample_correlation()` in the main patient1 codebase, so
numbers are comparable to the trained-model results there.

PATCHED COPY for lean_text_bottleneck scoring (from runs-wave/multisubj): run_encoding is redirected to rescore_stream when
RESCORE_OUT is set. The original lives in evolve-wave/src/eval.py. Different embedders
plug in: the linear ridge baseline (`src/baseline.py`) or the hand-built
interpretable transformer (`interpretable_transformer.py`).

Usage:
    from src.eval import EncodingConfig, run_encoding
    r = run_encoding(embedder, EncodingConfig(num_train=60))
    print(r['test_corr'])
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from . import data, encoding, features

SUBJECT = "patient1"

OVERALL_CSV_COLS = [
    "subject", "test_corr", "train_corr", "frac_test_elec_above_0.1",
    "encoding_seconds", "status", "model_shorthand_name", "n_params", "description",
]


@dataclass
class EncodingConfig:
    """Shared config so the baseline and every interpretable attempt are comparable."""
    num_train: int = 60
    num_test: int = 20
    ndelays: int = 4              # FIR taps of audio history (30 Hz frames), incl. delay 0
    lag: int = features.LAG_FRAMES   # neural lag (3 frames = 100 ms)
    trim: int = features.TRIM_FRAMES  # trim 30 frames (1.0 s) from each chunk start
    nboots: int = 5
    chunklen: int = 40
    nchunks: int = 20


def _per_chunk_elec_corr(feat_list, resp_list, wt):
    """Project metric: mean over chunks of (mean over electrodes of Pearson r)."""
    chunk_r = []
    for f, r in zip(feat_list, resp_list):
        pred = f @ wt                       # (T, 166)
        a = r - r.mean(0)
        b = pred - pred.mean(0)
        denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        rr = np.where(denom == 0, 0.0, (a * b).sum(0) / np.where(denom == 0, 1.0, denom))
        chunk_r.append(float(np.nanmean(rr)))
    chunk_r = np.array(chunk_r)
    return float(np.nanmean(chunk_r)), chunk_r


def run_encoding(embedder, cfg: EncodingConfig, verbose: bool = True) -> dict:
    """Fit + evaluate `embedder` as an encoding model. Returns a results dict."""
    if os.environ.get("RESCORE_OUT"):  # rescore-grid hook: streamed fit, per-electrode output
        from .rescore_stream import rescore
        rescore(embedder, cfg, os.environ["RESCORE_OUT"])
        raise SystemExit(0)
    t0 = time.time()
    train_chunks, test_chunks = data.get_chunk_names(cfg.num_train, cfg.num_test)
    if verbose:
        print(f"train chunks: {len(train_chunks)}   test chunks: {len(test_chunks)}")

    if verbose:
        print("loading melspec + high-gamma...")
    all_chunks = train_chunks + test_chunks
    stim = data.load_stim(all_chunks)
    resp = data.load_responses(all_chunks)

    if verbose:
        print("extracting features...")
    ftr, rtr = features.get_features_per_chunk(
        stim, resp, train_chunks, embedder,
        ndelays=cfg.ndelays, lag=cfg.lag, trim=cfg.trim)
    fte, rte = features.get_features_per_chunk(
        stim, resp, test_chunks, embedder,
        ndelays=cfg.ndelays, lag=cfg.lag, trim=cfg.trim)

    Xtr, Ytr = np.vstack(ftr), np.vstack(rtr)
    Xte, Yte = np.vstack(fte), np.vstack(rte)
    if verbose:
        print("feature shapes", Xtr.shape, Xte.shape, "response shapes", Ytr.shape, Yte.shape)

    if verbose:
        print("fitting ridge encoding model...")
    alphas = np.logspace(1, 4, 12)
    wt, corrs_concat, _ = encoding.bootstrap_ridge(
        Xtr, Ytr, Xte, Yte, alphas,
        nboots=cfg.nboots, chunklen=cfg.chunklen, nchunks=cfg.nchunks)

    # primary metric: per-chunk per-electrode correlation (project metric)
    test_corr, test_chunk_r = _per_chunk_elec_corr(fte, rte, wt)
    train_corr, _ = _per_chunk_elec_corr(ftr, rtr, wt)

    r = {
        "subject": SUBJECT,
        "test_corr": test_corr,                         # primary metric
        "train_corr": train_corr,
        "test_chunk_r": test_chunk_r,
        "corrs_test_concat": corrs_concat,              # per-electrode over concat test
        "frac_test_elec_above_0.1": float(np.nanmean(corrs_concat > 0.1)),
        "encoding_seconds": time.time() - t0,
        "train_chunks": train_chunks,
        "test_chunks": test_chunks,
    }
    if verbose:
        print(f"train_corr={train_corr:.4f}  test_corr={test_corr:.4f}  "
              f"frac_elec>0.1={r['frac_test_elec_above_0.1']:.4f}  "
              f"({r['encoding_seconds']:.1f}s)")
    return r


def make_result_row(r: dict, model_shorthand_name: str, n_params: int,
                    description: str, status: str = "success") -> dict:
    """Build a CSV row (all OVERALL_CSV_COLS) from a run_encoding results dict."""
    return {
        "subject": r["subject"],
        "test_corr": f"{r['test_corr']:.4f}",
        "train_corr": f"{r['train_corr']:.4f}",
        "frac_test_elec_above_0.1": f"{r['frac_test_elec_above_0.1']:.4f}",
        "encoding_seconds": f"{r['encoding_seconds']:.1f}",
        "status": status,
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}",
        "description": description,
    }


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
    is_baseline = ["baseline" in n.lower() for n in names]

    running_iters, running_max, best = [], [], float("-inf")
    for it, c, base in zip(iters, corrs, is_baseline):
        if base:
            continue
        best = max(best, c)
        running_iters.append(it)
        running_max.append(best)

    fig, ax = plt.subplots(figsize=(16, 5))
    nb_iters = [it for it, base in zip(iters, is_baseline) if not base]
    nb_corrs = [c for c, base in zip(corrs, is_baseline) if not base]
    ax.plot(nb_iters, nb_corrs, marker="o", linestyle="-", color="steelblue", label="test_corr")
    bl_iters = [it for it, base in zip(iters, is_baseline) if base]
    bl_corrs = [c for c, base in zip(corrs, is_baseline) if base]
    if bl_iters:
        ax.scatter(bl_iters, bl_corrs, marker="s", color="black", zorder=5, label="ridge baseline")
    if running_iters:
        ax.plot(running_iters, running_max, drawstyle="steps-post", color="crimson",
                linewidth=2, label="running max")
    for it, c, name in zip(iters, corrs, names):
        ax.annotate(name, (it, c), rotation=90, fontsize="xx-small",
                    textcoords="offset points", xytext=(0, 5), ha="center", va="bottom")
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-chunk per-electrode test correlation")
    ax.set_title("patient1 melspec→ECoG encoding test correlation over iterations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if len(iters) <= 30:
        ax.set_xticks(iters)
    fig.tight_layout()
    out_path = os.path.join(results_dir, "corr_over_iterations.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved → {out_path}")
