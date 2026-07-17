"""Huth natural-language fMRI loader + feature pipeline (LeBel 2023 / OpenNeuro ds003020) -- the
circuit-building-autoresearch dataset.

Runtime reads ONLY the baked ``data/huth/<study>.pt`` blob staged by ``scripts/prepare_data/huth.py``.
Following the reference src/features.py, the encoding is TR-rate: the model embeds each word's 10-gram
(per-word X), then ``apply_feature_pipeline`` Lanczos-downsamples those embeddings onto the fMRI TR
grid, z-scores per story, and adds 4 FIR delays -- run from ``harness.reward`` between the model's
forward pass and the fixed ridge (the model-independent Lanczos operators are baked; only the
model-dependent steps run here). ``load_huth`` returns the per-word X with TR-rate Y (fixed reference
train/test story split); no repeat-based ceiling -> ``nc=None`` (raw mean Pearson r)."""
from __future__ import annotations

import os

import torch

from harness.datasets import Dataset
from harness.loaders.shared import processed_dir

# Per-study transform data (Lanczos operators + word counts + n_delays), cached at load_split time so
# apply_feature_pipeline (called from the reward, which has no split handle) can rebuild TR features.
_PIPELINE: dict[str, dict] = {}


def load_huth(d: Dataset):
    """Load the baked blob: per-word X, TR-rate Y, and the fixed reference train/test story split."""
    path = os.path.join(processed_dir("huth"), f"{d.study}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{d.study} expects the in-repo blob {path!r}; data/huth/ is the only runtime source -- "
            f"stage it once with `python -m scripts.prepare_data.huth --subject {d.subject or 'UTS03'}`.")
    b = torch.load(path, weights_only=False)

    # NEURO_HUTH_NUM_TRAIN: use only the first N train stories (the reference's `num_train` knob; its
    # runs use 5-8 of the 93-story pool). Default 8 matches the circuit-building-autoresearch standard
    # protocol (num_train=8), so scores are comparable to that report; set =0 (or a large N) to use
    # the full baked pool, or a larger N to trade compute for a stronger ridge fit.
    Xtr, Ytr = b["X_train"], b["Y_train"]
    train_L, train_wc = b["train_L"], b["train_wc"]
    n = int(os.environ.get("NEURO_HUTH_NUM_TRAIN", "8") or 0)
    if 0 < n < len(train_wc):
        nw = sum(train_wc[:n])                                  # words in the first n stories
        ntr = sum(int(L.shape[0]) for L in train_L[:n])        # TRs in the first n stories
        Xtr, Ytr = Xtr[:nw], Ytr[:ntr]
        train_L, train_wc = train_L[:n], train_wc[:n]

    _PIPELINE[d.study] = {"train_L": train_L, "train_wc": train_wc,
                          "test_L": b["test_L"], "test_wc": b["test_wc"], "ndelays": b["ndelays"]}
    print(f"  {Xtr.shape[0]} train / {b['X_test'].shape[0]} test words -> "
          f"{Ytr.shape[0]}/{b['Y_test'].shape[0]} TRs ({len(train_wc)} train stories); "
          f"{Ytr.shape[1]} voxels")
    return (Xtr, b["X_test"], Ytr, b["Y_test"], {}, None)


def _zscore(v: torch.Tensor) -> torch.Tensor:
    s = v.std(0, unbiased=False)
    return (v - v.mean(0)) / torch.where(s == 0, torch.ones_like(s), s)


def _make_delayed(stim: torch.Tensor, ndelays: int) -> torch.Tensor:
    """Concatenate FIR-delayed copies (delays 1..ndelays TRs), as in the reference features.make_delayed."""
    out = []
    for delay in range(1, ndelays + 1):
        d = torch.zeros_like(stim)
        d[delay:] = stim[:-delay]
        out.append(d)
    return torch.cat(out, dim=1)


def apply_feature_pipeline(d: Dataset, A: torch.Tensor, which: str) -> torch.Tensor:
    """Per-word model embeddings ``A`` (rows == words of the ``which`` split) -> TR-rate design matrix:
    Lanczos-downsample per story, z-score per story, stack, then 4 FIR delays (reference features.py)."""
    c = _PIPELINE.get(d.study)
    if c is None:
        raise RuntimeError(f"huth feature pipeline for {d.study!r} not loaded -- call load_split first")
    zs = []
    for A_s, L_s in zip(torch.split(A, c[f"{which}_wc"]), c[f"{which}_L"]):
        zs.append(_zscore(L_s.to(A.device, A.dtype) @ A_s))     # (n_tr_story, d)
    return _make_delayed(torch.cat(zs, 0), c["ndelays"])        # (n_tr_split, ndelays * d)
