#!/usr/bin/env python3
"""Benchmark the neural-encoding readout pipeline, phase by phase, under GPU contention.

Times the EXACT path the harness runs (``harness/reward.py``) for ANY dataset (NSD/TVSD/NNN/
iEEG): load the split, move the stimulus to the GPU, run the (untrained) ``model.py``
forward to activations, fit the per-target ridge readout, predict both splits, and score the mean
per-target test correlation. The ``readout`` phase calls ``harness.ridge.ridge_predict`` directly,
so ``--method`` selects the SAME readout the search uses:

  * ``fast``     (default) -- the eigendecomposition + leave-one-out ridge (``harness.ridge._fast_predict``),
  * ``himalaya``           -- the reference himalaya ``RidgeCV`` (5-fold SVD; ~20x slower on GPU).

To mimic the live search -- several evals sharing the single A100 with no MPS, so they time-slice --
it launches ``--procs`` workers that run the whole pipeline at once and reports each worker's phase
breakdown + the batch wall-clock. CUDA is async, so every phase is wrapped in
``torch.cuda.synchronize()`` or the timings would only measure kernel-launch latency.

    python scripts/benchmark/readout.py                                 # nsd, fast, 3 concurrent
    python scripts/benchmark/readout.py --dataset tvsd --subject F
    python scripts/benchmark/readout.py --method himalaya --solo-baseline
    python scripts/benchmark/readout.py --dataset nnn --procs 3 --html   # also (re)render the dashboard
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

# This file is scripts/benchmark/readout.py -> repo root is three levels up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Phase order. The readout (fit + predict, both splits) is ONE phase so the breakdown is identical
# whether the fast method (which fuses them) or himalaya runs.
PHASES = ["data_load", "x_to_gpu", "model_forward", "readout", "score"]


def _reference_proj(X, seed: int = 0, d_out: int = 8192):
    """A correct on-device fixed random linear map to ``d_out`` (the seed baseline's intent),
    used only when ``model.py`` can't run -- so the readout can still be benchmarked. The
    ``seed``/``d_out`` defaults and the reshape+float deliberately MIRROR ``model.py``'s
    ``SEED``/``D_OUT`` baseline; keep them in sync if that baseline changes."""
    import torch

    X = torch.as_tensor(X).reshape(X.shape[0], -1).float()
    d_in = X.shape[1]
    if d_in <= d_out:
        return X
    g = torch.Generator(device=X.device).manual_seed(seed)
    W = torch.randn(d_in, d_out, generator=g, device=X.device, dtype=X.dtype) / d_in ** 0.5
    return X @ W


def _worker(dataset: str, subject: str, method: str, idx: int, out: dict) -> None:
    """Run the full readout once for ``dataset``/``subject`` with readout ``method``; record
    per-phase seconds into ``out[idx]``."""
    os.environ["NEURO_DATASET"] = dataset
    if subject:
        os.environ["NEURO_SUBJECT"] = str(subject)
    os.environ["NEURO_READOUT"] = method                 # drives harness.ridge.ridge_predict

    import torch

    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness import reward as R

    dev = R._readout_device()
    cuda = dev.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    t = {}
    wall0 = time.time()

    # 1) Data load (decode/resize stimuli, cached after first build; read the responses).
    s = time.time()
    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte, roi_masks = split[0], split[1], split[2], split[3], split[4]
    t["data_load"] = time.time() - s

    # 2) Move the stimulus to the readout device.
    s = time.time()
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(dev), Xte.to(dev)
    sync()
    t["x_to_gpu"] = time.time() - s

    # 3) Model forward: X -> activations. Use model.py; fall back to a correct on-device projection
    #    if it faults (the seed baseline is broken on GPU, or a proposer left it mid-edit).
    s = time.time()
    try:
        from model import MODEL
        A_train, A_test = MODEL(Xtr), MODEL(Xte)
        t["model_src"] = "model.py"
    except Exception as exc:  # noqa: BLE001 -- benchmark must not die on a bad model.py
        A_train, A_test = _reference_proj(Xtr), _reference_proj(Xte)
        t["model_src"] = f"reference (model.py failed: {type(exc).__name__})"
    sync()
    t["model_forward"] = time.time() - s

    # Narrow the targets to the reward ROI, exactly as reward.py does.
    mask = R.reward_region_mask(roi_masks, DATASET.reward_roi)
    Yv_train, Yv_test = R.select_targets(Ytr, mask), R.select_targets(Yte, mask)

    # 4) Readout: the harness's per-target ridge (fast or himalaya, per NEURO_READOUT) -> predictions
    #    for BOTH splits. This is the phase the whole exercise is about.
    from harness.ridge import ridge_predict
    sync()
    s = time.time()
    pred_train, pred_test = ridge_predict(A_train, A_test, Yv_train, device=dev)
    sync()
    t["readout"] = time.time() - s

    # 5) Score: mean per-target Pearson r (the reward is test_corr).
    s = time.time()
    test_corr = float(R._corr_per_column(pred_test, R._to_np(Yv_test)).mean())
    train_corr = float(R._corr_per_column(pred_train, R._to_np(Yv_train)).mean())
    t["score"] = time.time() - s

    t["total"] = time.time() - wall0
    t["test_corr"], t["train_corr"] = test_corr, train_corr
    t["n_train"], t["n_test"] = int(Yv_train.shape[0]), int(Yv_test.shape[0])
    t["n_targets"], t["d_out"] = int(Yv_train.shape[1]), int(A_train.shape[1])
    out[idx] = dict(t)


def _run_batch(n, dataset, subject, method, label):
    """Launch ``n`` workers concurrently (spawn, required for CUDA); return ``(results, wall)``."""
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    print(f"\n=== {label}: {n} concurrent worker(s) | {dataset} subj={subject or '-'} | readout={method} ===")
    wall = time.time()
    procs = [ctx.Process(target=_worker, args=(dataset, subject, method, i, out)) for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.time() - wall
    results = {i: out[i] for i in sorted(out)}
    _report(results, wall)
    return results, wall


def _report(results, wall):
    """Print the per-worker phase-timing table + the per-phase max and batch wall-clock."""
    if not results:
        print("  (no workers returned -- check for an error/OOM above)")
        return
    cols = "  ".join(f"{p:>14}" for p in PHASES)
    print(f"  {'worker':>7}  {cols}  {'total':>8}  {'test_corr':>9}")
    print("  " + "-" * (9 + len(cols) + 22))
    for i, t in results.items():
        row = "  ".join(f"{t.get(p, 0.0):>14.3f}" for p in PHASES)
        print(f"  {i:>7}  {row}  {t['total']:>8.2f}  {t['test_corr']:>9.4f}")
    agg = "  ".join(f"{max(t.get(p, 0.0) for t in results.values()):>14.3f}" for p in PHASES)
    print("  " + "-" * (9 + len(cols) + 22))
    print(f"  {'max':>7}  {agg}")
    meta = next(iter(results.values()))
    print(f"\n  shapes: n_train={meta['n_train']} n_test={meta['n_test']} "
          f"n_targets={meta['n_targets']} D_out={meta['d_out']}")
    print(f"  forward model: {meta.get('model_src', '?')}")
    print(f"  batch wall-clock: {wall:.2f}s   (slowest worker total: "
          f"{max(t['total'] for t in results.values()):.2f}s)")


def _worker_record(t):
    """Slim one worker's timing dict down to the phase seconds + totals kept in the results JSON."""
    rec = {p: t.get(p, 0.0) for p in PHASES}
    rec.update(total=t.get("total"), test_corr=t.get("test_corr"),
               train_corr=t.get("train_corr"), model_src=t.get("model_src"))
    return rec


def _save_results(dataset, subject, method, procs, solo, solo_wall, par, par_wall):
    """Write the run (shapes, concurrent + optional solo phase breakdowns, contention ratio) to a
    timestamped results JSON; return its path."""
    meta = next(iter(par.values()))
    payload = {
        "dataset": dataset,
        "subject": subject,
        "method": method,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "procs": procs,
        "shapes": {k: meta[k] for k in ("n_train", "n_test", "n_targets", "d_out")},
        "model_src": meta.get("model_src"),
        "phases": PHASES,
        "solo": None,
        "concurrent": {"workers": [_worker_record(par[i]) for i in sorted(par)], "wall": par_wall},
        "contention_ratio": None,
    }
    if solo:
        payload["solo"] = {"workers": [_worker_record(solo[i]) for i in sorted(solo)], "wall": solo_wall}
        payload["contention_ratio"] = max(t["total"] for t in par.values()) / next(iter(solo.values()))["total"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{dataset}_subj{subject or 'NA'}_{method}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main(argv=None):
    """Run the readout pipeline in N concurrent workers (plus an optional solo baseline), print +
    save the phase-by-phase timings, and optionally re-render the HTML dashboard."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="nsd", help="NEURO_DATASET preset (nsd/tvsd/nnn/ieeg)")
    p.add_argument("--subject", default="1", help="subject/animal id (e.g. 1-8 for nsd, F/N for tvsd); '' if N/A")
    p.add_argument("--method", default="fast", choices=["fast", "himalaya"],
                   help="readout method (NEURO_READOUT); default fast")
    p.add_argument("--procs", type=int, default=3, help="concurrent workers sharing the GPU (default 3)")
    p.add_argument("--solo-baseline", action="store_true", help="also run 1 worker alone, to price contention")
    p.add_argument("--no-save", action="store_true", help="do not write the results JSON")
    p.add_argument("--html", action="store_true", help="(re)render the results dashboard after running")
    a = p.parse_args(argv)

    solo, solo_wall = (None, None)
    if a.solo_baseline:
        solo, solo_wall = _run_batch(1, a.dataset, a.subject, a.method, "SOLO baseline")
    par, par_wall = _run_batch(a.procs, a.dataset, a.subject, a.method, f"CONCURRENT x{a.procs}")

    if a.solo_baseline and solo and par:
        st = next(iter(solo.values()))["total"]
        ps = max(t["total"] for t in par.values())
        print(f"\n  contention: slowest of {a.procs} took {ps:.2f}s vs {st:.2f}s solo ({ps / st:.2f}x).")

    if not a.no_save and par:
        path = _save_results(a.dataset, a.subject, a.method, a.procs, solo, solo_wall, par, par_wall)
        print(f"\n  saved results -> {path}")
    if a.html:
        subprocess.run([sys.executable, os.path.join(HERE, "render_report.py")], check=False)


if __name__ == "__main__":
    main()
