"""Area readouts of the searched LAION model (erjpu) for the circuit figure.

For hV4, FFA and PPA, the search encoding readout (center design, per-voxel 5-fold-CV ridge) is fit on each
subject's unique images, as in src/figures/fmri/d_families.py (erjpu's search setting, protocol v4: all five
LAION subjects, all shared images as test). The area readout is the mean of the per-voxel weight vectors over
the area's voxels, so it predicts the area's mean z-scored response. Each subject's readout is stored as an
affine map of the raw model features: prediction = features @ w + b.

Writes artifacts/figures/circuit/data/readout.npz:
  w (subjects, areas, 9996), b (subjects, areas)   affine area readouts on raw features
  pred, meas (subjects, areas, n_shared)           predicted and measured area responses on the shared images
  shared_ids (n_shared,), image rows of sub-01     shared images in a common order
  labels (9996,), blocks (9996,), ops (9996,)      family label, feature block and texture operation per column
Reads cached features from artifacts/figures/fmri/data/d/ (run d_families.py first).

Usage: python src/figures/circuit/readout.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/figures/fmri"))
sys.path.insert(0, str(REPO / "src"))
from d_families import BLOCKS, DATA, V4_SUBJECTS, block_labels, load_champion, v4_selected_ids  # noqa: E402
from e_texture import SUBSTATS  # noqa: E402

AREAS = ["hV4", "FFA", "PPA"]
OUT = REPO / "artifacts/figures/circuit/data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(NEURO_DATASET="laion", NEURO_STAGE="search", NEURO_METRIC="encoding",
                      NEURO_REGION="visual", NEURO_IMSIZE="128")
    import neuro_eval.datasets as nd
    from neuro_eval.datasets import _load_native, get_dataset, load_split, with_subject
    from neuro_eval.encoding import ALPHAS, ENCODING_FEATURES, _fit_predict, _Ridge, design, zscore_targets
    nd._selected_ids = v4_selected_ids
    assert ENCODING_FEATURES == "center"
    device = torch.device("cuda")

    model = load_champion()
    with torch.no_grad():
        model(torch.rand(2, 128 * 128 * 3, device=device))
    widths = dict(zip(BLOCKS, model._widths))
    labels = block_labels(widths)
    blocks = np.concatenate([[b] * widths[b] for b in BLOCKS])
    ops = np.full(len(labels), "", dtype="<U1")
    for k, (_, bs, _) in SUBSTATS.items():
        ops[(labels == "T") & np.isin(blocks, bs)] = k

    base = get_dataset("laion", region="visual", metric="encoding")
    W, B, P, M, ids0 = [], [], [], [], None
    for s in V4_SUBJECTS:
        d = with_subject(base, s)
        Xtr, Xte, Ytr, Yte, *_ = load_split(d)
        native, subj = _load_native(d)
        _, test_ids = v4_selected_ids(d, native, subj)
        ids0 = ids0 or test_ids
        assert test_ids == ids0  # the shared images come in the same order for every subject
        masks = np.load(DATA / "d" / f"voxels_{s}.npz")
        A = torch.from_numpy(np.load(DATA / "d" / f"features_{s}.npy")).to(device)
        ntr = len(Xtr)
        tr, te = np.arange(ntr), np.arange(ntr, len(A))
        mu = A[:ntr].mean(0)
        scale = (A[:ntr] - mu).pow(2).sum(1).mean().sqrt().clamp_min(1e-6)
        Z = design(A, tr, device, features=ENCODING_FEATURES)
        vox = np.flatnonzero(np.any([masks[f"mask_{a}"] for a in AREAS], axis=0))
        Y = torch.cat([Ytr, Yte])[:, vox]
        alphas, Yt, pred = _fit_predict(Z, Y, tr, te)
        ridge = _Ridge(Z[:ntr], zscore_targets(Y.to(device), torch.as_tensor(tr, device=device),
                                               torch.as_tensor(te, device=device))[0])
        w_s, b_s, p_s, m_s = [], [], [], []
        for a in AREAS:
            cols = np.flatnonzero(masks[f"mask_{a}"][vox])
            coef = torch.zeros(len(ridge.lam), device=device, dtype=Z.dtype)
            for alpha in np.unique(alphas[cols]):
                c = torch.as_tensor(cols[alphas[cols] == alpha], device=device)
                coef += ridge.coef(float(alpha), c).sum(1)
            wz = ridge.V @ (coef / len(cols))                   # weights on the design Z
            check = Z[ntr:] @ wz
            assert torch.allclose(check, pred[:, cols].mean(1), atol=1e-3), (check - pred[:, cols].mean(1)).abs().max()
            w = wz / scale
            w_s.append(w.cpu().numpy()); b_s.append(float(-(mu @ w)))
            p_s.append(check.cpu().numpy()); m_s.append(Yt[:, cols].mean(1).cpu().numpy())
            r = np.corrcoef(p_s[-1], m_s[-1])[0, 1]
            print(f"{s} {a}: {len(cols)} voxels, r(area mean) = {r:.3f}", flush=True)
        W.append(w_s); B.append(b_s); P.append(p_s); M.append(m_s)
        if s == V4_SUBJECTS[0]:
            images = (Xte.reshape(-1, 128, 128, 3).clamp(0, 1) * 255).round().byte().numpy()
        del A, Z, ridge
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / "readout.npz", w=np.array(W, np.float32), b=np.array(B, np.float32),
                        pred=np.array(P, np.float32), meas=np.array(M, np.float32), shared_ids=np.array(ids0),
                        images=images, labels=labels, blocks=blocks, ops=ops, areas=AREAS, subjects=V4_SUBJECTS)


if __name__ == "__main__":
    main()
