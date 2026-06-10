"""Benchmark: the custom torch GCV ridge readout vs Himalaya's RidgeCV (GPU).

The reward in this repo is a per-target ridge readout from a fixed model's
activations to recorded neural activity (``harness/reward.py``). That readout is a
hand-rolled torch reimplementation of sklearn's ``RidgeCV(alpha_per_target=True)``
using closed-form leave-one-out GCV, run on the GPU. The Gallant lab's
`himalaya <https://gallantlab.org/himalaya/>`_ ships a GPU-accelerated ridge with
its own per-target alpha selection (``torch_cuda`` backend).

This script compares the two **on identical data** along two axes:

* **speed** -- wall-clock of fit+predict on the readout device (A100 when present),
  averaged over ``--repeats`` timed runs after a warm-up;
* **predictivity** -- the resulting ``test_corr`` (mean per-target Pearson r),
  scored with the SAME metric (``harness/reward.predictivity``) for both methods, so
  any difference reflects the readout, not the scoring.

Apples-to-apples caveat: both search the same alpha grid (``harness/reward._ALPHAS``)
with an intercept, but the **alpha-selection criterion differs** -- the custom
readout uses closed-form leave-one-out GCV, while Himalaya's ``RidgeCV`` uses
k-fold CV (``--cv``, default 5). Small ``test_corr`` differences are expected and
are part of what is being measured.

This is a STANDALONE benchmark; it does not modify the fixed harness.

Run from the repo root::

    python scripts/bench_himalaya.py                       # synthetic + text
    python scripts/bench_himalaya.py --datasets synthetic  # no data load
    NEURO_DATASET=text python scripts/bench_himalaya.py --datasets text
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import harness.reward as harness_reward  # noqa: E402  (for the fixed-alpha equivalence check)
from harness.reward import (  # noqa: E402  (harness readout, the thing under test)
    _ALPHAS,
    _as_f64,
    _corr_per_column,
    _readout_device,
    _ridge_gcv_coef,
    predictivity,
)
from model import MODEL  # noqa: E402  (pure model; needs no data)


# --- data ----------------------------------------------------------------------
def synthetic_split(n_train, p, n_targets, n_test, seed=0):
    """A controlled (A_train, A_test, Y_train, Y_test) with real per-target signal.

    Sized to resemble a neural readout: a moderate feature width (``p``) mapping
    to many noisy targets, so per-target alpha selection actually matters.
    """
    rng = np.random.default_rng(seed)
    A_train = rng.standard_normal((n_train, p)).astype("float32")
    A_test = rng.standard_normal((n_test, p)).astype("float32")
    # Sparse per-target weights + noise so the best alpha varies across targets.
    W = (rng.standard_normal((p, n_targets)) * (rng.random((p, n_targets)) < 0.3)).astype("float32")
    noise = 0.7 * rng.standard_normal((n_train, n_targets)).astype("float32")
    Y_train = A_train @ W + noise
    Y_test = A_test @ W + 0.7 * rng.standard_normal((n_test, n_targets)).astype("float32")
    return (torch.from_numpy(A_train), torch.from_numpy(A_test),
            torch.from_numpy(Y_train), torch.from_numpy(Y_test))


def dataset_split(dataset_key):
    """Load a real harness split and turn the stimulus into model activations.

    Returns (A_train, A_test, Y_train, Y_test) where A_* are the model's
    activations on the stimulus (the exact tensors the readout sees in eval).
    """
    os.environ["NEURO_DATASET"] = dataset_key
    # data import is deferred to here so synthetic-only runs need no data.
    import importlib

    from harness import datasets
    importlib.reload(datasets)
    from harness.data import load_split

    dataset = datasets.DATASET
    print(f"  loading split for dataset={dataset_key!r} "
          f"(study={dataset.study}, stimulus={dataset.stimulus}, neural={dataset.neural}) ...")
    t = time.time()
    X_train, X_test, Y_train, Y_test = load_split(dataset)
    print(f"  loaded in {time.time() - t:.1f}s")

    A_train = MODEL(X_train if torch.is_tensor(X_train) else torch.as_tensor(X_train))
    A_test = MODEL(X_test if torch.is_tensor(X_test) else torch.as_tensor(X_test))
    return A_train, A_test, Y_train, Y_test


# --- timing helper -------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class GpuMonitor:
    """Background sampler of GPU utilization/memory via nvidia-smi.

    Proves the timed work actually ran on the A100 (a silent CPU fallback would
    show ~0% util and invalidate the speed comparison). No-op when not on CUDA.
    """

    def __init__(self, device, period=0.02):
        self.on = device.type == "cuda"
        self.period = period
        self.peak_util = self.peak_mem = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        import subprocess
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True).stdout.strip().splitlines()[0]
                u, m = (int(x) for x in out.split(","))
                self.peak_util, self.peak_mem = max(self.peak_util, u), max(self.peak_mem, m)
            except Exception:
                pass
            time.sleep(self.period)

    def __enter__(self):
        if self.on:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.on and self._thread:
            self._stop.set()
            self._thread.join()


def timed(fn, device, repeats):
    """Run ``fn`` once (warm-up, untimed) then ``repeats`` timed; sampling GPU use.

    Returns (result, best_s, mean_s, peak_util%, peak_mem_MiB).
    """
    result = fn()                                   # warm-up (JIT/alloc/caches)
    _sync(device)
    times = []
    with GpuMonitor(device) as mon:
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            result = fn()
            _sync(device)
            times.append(time.perf_counter() - t0)
    return result, min(times), float(np.mean(times)), mon.peak_util, mon.peak_mem


# --- the two readouts ----------------------------------------------------------
def custom_readout(A_train, A_test, Y_train, Y_test, device):
    """Replicate eval.evaluate_on_split's readout: center, GCV fit, predict.

    Returns a closure that does fit+predict (the timed unit) and yields the test
    predictions; ``run.compute_device`` records where the solve ran.
    """
    A_tr = _as_f64(A_train, device)
    A_te = _as_f64(A_test, device)
    Y_tr = _as_f64(Y_train, device)

    def run():
        x_off, y_off = A_tr.mean(0), Y_tr.mean(0)
        coef = _ridge_gcv_coef(A_tr - x_off, Y_tr - y_off)
        run.compute_device = str(coef.device)
        return (A_te - x_off) @ coef + y_off

    run.compute_device = "?"
    return run


def himalaya_readout(A_train, A_test, Y_train, Y_test, device, cv):
    """Himalaya RidgeCV (per-target alpha) on the torch_cuda backend when on GPU.

    Returns a closure doing fit+predict (the timed unit) yielding test predictions;
    ``run.compute_device`` records the backend's compute device (``best_alphas_``).
    """
    from himalaya.backend import get_backend, set_backend
    from himalaya.ridge import RidgeCV

    backend = "torch_cuda" if device.type == "cuda" else "torch"
    set_backend(backend)
    run_backend = get_backend().name

    dtype = torch.float32 if device.type == "cuda" else torch.float64
    A_tr = A_train.to(device=device, dtype=dtype)
    A_te = A_test.to(device=device, dtype=dtype)
    Y_tr = Y_train.to(device=device, dtype=dtype)

    def run():
        est = RidgeCV(alphas=np.asarray(_ALPHAS), fit_intercept=True, cv=cv)
        est.fit(A_tr, Y_tr)
        # best_alphas_ stays on the compute device; coef_ is moved to CPU by himalaya
        # for storage, so it is NOT a reliable indicator of where the solve ran.
        run.compute_device = f"{est.best_alphas_.device} [{run_backend}]"
        return est.predict(A_te)

    run.compute_device = "?"
    return run


# --- report --------------------------------------------------------------------
def _to_np(pred):
    a = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    return a.astype("float64")


def _per_target_corr(pred, y):
    """Per-target Pearson r vector (the harness's own column-wise metric)."""
    return _corr_per_column(_to_np(pred), y)


def fixed_alpha_equivalence(A_train, A_test, Y_train, Y_test, device, alpha):
    """Prove the two ridge implementations are numerically identical at a FIXED alpha.

    With a single alpha there is no per-target selection, so the only thing left is
    the core ridge solve. Any divergence here would be a real math difference; in
    practice both reduce to the same closed-form ridge, so predictions match to
    float precision. This isolates "is the math the same?" from the separate
    "do the CV criteria pick the same alpha?" question.

    The custom path is exercised through the REAL harness solver by temporarily
    pinning its alpha grid to ``[alpha]`` (restored afterwards).
    """
    A_tr, A_te = _as_f64(A_train, device), _as_f64(A_test, device)
    Y_tr = _as_f64(Y_train, device)
    x_off, y_off = A_tr.mean(0), Y_tr.mean(0)

    saved = harness_reward._ALPHAS
    try:
        harness_reward._ALPHAS = np.asarray([alpha])
        coef = _ridge_gcv_coef(A_tr - x_off, Y_tr - y_off)
    finally:
        harness_reward._ALPHAS = saved
    pred_custom = _to_np((A_te - x_off) @ coef + y_off)

    from himalaya.backend import set_backend
    from himalaya.ridge import Ridge
    set_backend("torch_cuda" if device.type == "cuda" else "torch")
    dtype = torch.float64 if device.type != "cuda" else torch.float32
    est = Ridge(alpha=float(alpha), fit_intercept=True)
    est.fit(A_train.to(device, dtype), Y_train.to(device, dtype))
    pred_hima = _to_np(est.predict(A_test.to(device, dtype)))

    scale = max(np.abs(pred_custom).max(), 1e-300)
    rel = np.abs(pred_custom - pred_hima).max() / scale
    print(f"  fixed-alpha (={alpha:g}) equivalence: max|Δ|={np.abs(pred_custom - pred_hima).max():.2e}"
          f"  relative={rel:.2e}  -> {'IDENTICAL' if rel < 1e-4 else 'DIVERGED'} ridge math")
    return rel


def bench_one(label, A_train, A_test, Y_train, Y_test, device, repeats, cv, eq_alpha):
    print(f"\n=== {label} ===")
    print(f"  A_train{tuple(A_train.shape)}  A_test{tuple(A_test.shape)}  "
          f"Y_train{tuple(Y_train.shape)}  Y_test{tuple(Y_test.shape)}")
    print(f"  device={device}  repeats={repeats}  alphas={len(_ALPHAS)}  himalaya_cv={cv}")

    runners = {
        "custom (GCV LOO)": custom_readout(A_train, A_test, Y_train, Y_test, device),
        f"himalaya ({cv}-fold)": himalaya_readout(A_train, A_test, Y_train, Y_test, device, cv),
    }

    print(f"\n  {'method':<22} {'best_s':>9} {'mean_s':>9} {'test_corr':>10} "
          f"{'gpu_util':>9} {'gpu_mem':>9}  compute_device")
    print("  " + "-" * 86)
    y = _to_np(Y_test)
    preds, best, tcorr = {}, {}, {}
    for name, fn in runners.items():
        pred, best_s, mean_s, peak_u, peak_m = timed(fn, device, repeats)
        preds[name] = pred
        best[name] = best_s
        tcorr[name] = float(_per_target_corr(pred, y).mean())
        print(f"  {name:<22} {best_s:>9.4f} {mean_s:>9.4f} {tcorr[name]:>10.4f} "
              f"{peak_u:>8}% {peak_m:>7}Mi  {fn.compute_device}")

    names = list(preds)
    a, b = _to_np(preds[names[0]]), _to_np(preds[names[1]])
    # Scale-invariant prediction agreement: |Δ| normalised by the prediction scale
    # (absolute |Δ| is meaningless across modalities -- MEG preds are ~1e-7 Tesla).
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    print(f"\n  prediction agreement   abs max|Δ|={np.abs(a - b).max():.3e}  "
          f"relative max|Δ|={np.abs(a - b).max() / scale:.3e}")
    # The real "are the encoding scores identical?" check: agreement of the two
    # per-target test_corr MAPS (this is what gets computed/compared downstream).
    rc, rh = _per_target_corr(preds[names[0]], y), _per_target_corr(preds[names[1]], y)
    map_r = float(np.corrcoef(rc, rh)[0, 1])
    print(f"  per-target score map   r(custom, himalaya)={map_r:.5f}  "
          f"mean|Δr|={np.abs(rc - rh).mean():.3e}  max|Δr|={np.abs(rc - rh).max():.3e}")
    print(f"  mean test_corr         custom={tcorr[names[0]]:.4f}  himalaya={tcorr[names[1]]:.4f}  "
          f"Δ={tcorr[names[0]] - tcorr[names[1]]:+.4f}")

    if eq_alpha is not None:
        fixed_alpha_equivalence(A_train, A_test, Y_train, Y_test, device, eq_alpha)

    fast, slow = sorted(names, key=lambda n: best[n])
    speedup = best[slow] / best[fast]
    print(f"  speed (best-of-{repeats}): {fast!r} {speedup:.2f}x faster than {slow!r}")
    return {"label": label, "custom_s": best[names[0]], "himalaya_s": best[names[1]],
            "speedup": best[names[1]] / best[names[0]], "tc_custom": tcorr[names[0]],
            "tc_himalaya": tcorr[names[1]], "map_r": map_r}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["synthetic", "text"],
                   choices=["synthetic", "text", "nsd", "meg"],
                   help="which split(s) to benchmark (default: synthetic + text)")
    p.add_argument("--repeats", type=int, default=5, help="timed runs per method (default: 5)")
    p.add_argument("--cv", type=int, default=5, help="Himalaya RidgeCV folds (default: 5)")
    p.add_argument("--syn-train", type=int, default=600)
    p.add_argument("--syn-features", type=int, default=256, help="match random_fourier out_dim")
    p.add_argument("--syn-targets", type=int, default=50000, help="many noisy neural targets")
    p.add_argument("--syn-test", type=int, default=150)
    p.add_argument("--eq-alpha", type=float, default=1.0,
                   help="fixed-alpha equivalence-check alpha; set <0 to skip (default: 1.0)")
    args = p.parse_args()

    device = _readout_device()
    print(f"readout device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    eq_alpha = None if args.eq_alpha < 0 else args.eq_alpha

    summary = []
    for ds in args.datasets:
        if ds == "synthetic":
            split = synthetic_split(args.syn_train, args.syn_features,
                                    args.syn_targets, args.syn_test)
            label = (f"synthetic  n_train={args.syn_train} p={args.syn_features} "
                     f"n_targets={args.syn_targets}")
        else:
            split = dataset_split(ds)
            label = f"dataset={ds}  model={MODEL.name}"
        summary.append(bench_one(label, *split, device, args.repeats, args.cv, eq_alpha))

    print("\n" + "=" * 78)
    print("SUMMARY  (custom torch GCV-LOO  vs  Himalaya RidgeCV k-fold)")
    print("=" * 78)
    print(f"{'dataset':<34} {'custom_s':>9} {'hima_s':>9} {'speedup':>8} "
          f"{'tc_cust':>8} {'tc_hima':>8} {'map_r':>7}")
    print("-" * 78)
    for r in summary:
        print(f"{r['label']:<34} {r['custom_s']:>9.4f} {r['himalaya_s']:>9.4f} "
              f"{r['speedup']:>7.2f}x {r['tc_custom']:>8.4f} {r['tc_himalaya']:>8.4f} "
              f"{r['map_r']:>7.4f}")


if __name__ == "__main__":
    main()
