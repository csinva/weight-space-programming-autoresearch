"""Representational similarity analysis (RSA): RDM construction and comparison.

Two RDM builders, deliberately kept separate:
  compute_rdm   float32 with clamping; used for neural RDMs.
  rdm_chunked   float64 accumulation over feature chunks; used for wide model RDMs.

Two Spearman implementations, also kept separate:
  ranked_triu                                           ordinal argsort ranks; Spearman = dot product.
  triu_ranks + spearman_from_ranks                      scipy's average ranks on the GPU, each RDM ranked once.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Row-wise ordinal ranks via double argsort (tied values get distinct consecutive ranks)."""
    return torch.argsort(torch.argsort(x, dim=1), dim=1).float()


def compute_rdm(
    representations: torch.Tensor, *, correlation: str = "Pearson", correction: float = 1e-12
) -> torch.Tensor:
    """(n_samples x n_samples) RDM (1 - correlation), Pearson or Spearman.

    Diagonal is 0 (zero self-dissimilarity); off-diagonal ranges 0 (identical)
    to 2 (perfectly anti-correlated).
    """
    corr = correlation.lower()
    if corr not in {"pearson", "spearman"}:
        raise ValueError("correlation must be 'Pearson' or 'Spearman'")

    x = representations.float().clone()
    if corr == "spearman":
        x = _rank(x)

    x -= x.mean(dim=1, keepdim=True)
    std = x.pow(2).mean(dim=1).add(correction).sqrt()

    zero_mask = std < correction * 10
    if zero_mask.any():
        logger.warning("%d/%d rows have ~zero variance for %s RDM", zero_mask.sum(), len(std), correlation)
        std[zero_mask] = 1.0

    cov = (x @ x.T) / x.size(1)
    corr_mat = cov / (std[:, None] * std[None, :] + correction)
    corr_mat.clamp_(-1, 1).fill_diagonal_(1.0)
    return 1.0 - corr_mat


def triu_ranks(rdm: torch.Tensor) -> torch.Tensor:
    """Average ranks (as scipy.stats.rankdata) of the upper triangle of rdm, float64 on its device."""
    n = rdm.size(0)
    i = torch.triu_indices(n, n, offset=1, device=rdm.device)
    v, order = rdm[i[0], i[1]].double().sort()
    _, inv, counts = torch.unique_consecutive(v, return_inverse=True, return_counts=True)
    avg = counts.cumsum(0).double() - (counts.double() - 1) / 2  # 1-based mean rank of each tie block
    ranks = torch.empty_like(v)
    ranks[order] = avg[inv]
    return ranks


def spearman_from_ranks(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman of two RDMs from their triu_ranks: Pearson of the ranks (= scipy.stats.spearmanr); NaN if constant."""
    a, b = a - a.mean(), b.to(a.device) - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def rdm_chunked(X, device, chunk=200_000, zscore_units=False, on_device=False):
    """1 - Pearson correlation RDM of X (n, D), on the CPU or already on `device` (no host copy then), accumulated
    over feature chunks. Returns float32 on the CPU, or on `device` if on_device.

    zscore_units: first standardise every unit (column) over the n images, so that no unit dominates the
    correlation by its scale or its mean (units constant over the images become 0).
    """
    n, D = X.shape

    def cols(j):
        x = X[:, j:j + chunk].to(device).double()
        if zscore_units:
            sd = x.std(0)
            x = torch.where(sd > 0, (x - x.mean(0)) / sd.clamp_min(1e-12), torch.zeros_like(x))
        return x

    mean = torch.zeros(n, dtype=torch.float64, device=device)
    for j in range(0, D, chunk):
        mean += cols(j).sum(1)
    mean /= D
    G = torch.zeros(n, n, dtype=torch.float64, device=device)
    for j in range(0, D, chunk):
        xc = (cols(j) - mean[:, None]).float()
        G += (xc @ xc.T).double()
        del xc
    d = torch.sqrt(torch.diag(G)).clamp_min(1e-12)
    rdm = 1.0 - G / (d[:, None] * d[None, :])
    rdm.fill_diagonal_(0.0)
    return rdm.float() if on_device else rdm.float().cpu()


def ranked_triu(rdm):
    """Upper-triangle ranks of an RDM, centred and scaled to unit norm, so Spearman = a dot product."""
    n = rdm.shape[0]
    i, j = torch.triu_indices(n, n, offset=1, device=rdm.device)
    r = rdm[i, j].double().argsort().argsort().double()
    return F.normalize(r - r.mean(), dim=0)


def image_sets(neural, test_ids, subjects, regions, n_select, test_subsets=None):
    """{"test": the shared images, "test_{name}": each named subset of them, "sel{s}": n_select of subject s's own
    training images (RandomState(42))}."""
    sets = {"test": test_ids, **{f"test_{k}": v for k, v in (test_subsets or {}).items()}}
    for s in subjects:
        train_ids = sorted(set.intersection(*[set(map(str, neural[r][s]["train"])) for r in regions]), key=int)
        idx = np.random.RandomState(42).choice(len(train_ids), size=min(n_select, len(train_ids)), replace=False)
        sets[f"sel{s}"] = [train_ids[i] for i in idx]
    return sets


def neural_rdm(neural, sets, r, s, split, device):
    ids = sets[f"sel{s}"] if split == "sel" else sets[split]
    resp = neural[r][s]["train" if split == "sel" else "test"]
    key = type(next(iter(resp.keys())))
    Y = np.stack([resp[key(i)] for i in ids]).reshape(len(ids), -1)
    return compute_rdm(torch.as_tensor(Y, dtype=torch.float32, device=device)).cpu()


def rsa_rows(X, order, sets, pairs, nrdm, device, chunk=200_000, zscore_units=False):
    """(split, region, subject, Spearman RSA) of the model RDM on each image set ("sel{s}" -> split "sel") against
    nrdm[(region, subject, split)], for the (region, subject) pairs(set_name); order: image id -> row of X.
    nrdm values may be RDMs or their triu_ranks (1-D, ranked once by the caller). The model RDM is ranked once per
    image set on the GPU; Spearman = Pearson of average ranks, as scipy.stats.spearmanr."""
    ranks = {}  # sets with the same images (e.g. two subjects' selection sets) share one model RDM
    for set_name, set_ids in sets.items():
        key = tuple(set_ids)
        if key not in ranks:
            rows = [order[i] for i in set_ids]
            if rows == list(range(len(rows))):  # identity / prefix: a view, no copy
                Xs = X[:len(rows)]
            else:  # gathered on X's own device
                Xs = X[torch.as_tensor(rows, device=X.device)]
            rdm = rdm_chunked(Xs, device, chunk=chunk, zscore_units=zscore_units, on_device=True)
            ranks = {key: triu_ranks(rdm)}
            del Xs
            del rdm
        split = "sel" if set_name.startswith("sel") else set_name
        for r, s in pairs(set_name):
            n = nrdm[(r, s, split)]
            yield split, r, s, spearman_from_ranks(ranks[key], n if n.ndim == 1 else triu_ranks(n.to(device)))
