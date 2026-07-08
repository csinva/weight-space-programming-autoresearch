"""Nearest-neighbour retrieval visualization -- a QUALITATIVE probe of what each representation
treats as "similar".

For a query NSD image we embed the whole gallery (the shared-1000 held-out test images) into a
representation space, rank every gallery image by cosine similarity to the query, and show the
top-K neighbours. The retrieved images ARE the figure: they make visible what a representation
considers alike.

Retrieval in each encoder's OWN representation space (no ridge, no predicted-brain readout -- just
the response/feature vector). Three rows per figure, in this order:
  1. BRAIN  -- the actual recorded neural response for that stream's reliable voxels (early betas in
     the early figure, ventral betas in the ventral figure). This is the ground-truth reference.
  2. our best untrained champion model (default: the NSD ~0.44 champion qq145) -- one representation,
     reused in both stream figures (the model has no internal stream split; that contrast is the point);
  3. AlexNet: conv4 -> SRP-2048 in the EARLY figure; fc7 -> raw 4096 (no projection) in the VENTRAL figure.

Two figures per query (rows = brain / our model / AlexNet, cols = query + top-K neighbours):
  * <out>/early_visual_qNN.png    : brain(early)   vs  our model  vs  AlexNet conv4(SRP)
  * <out>/ventral_stream_qNN.png  : brain(ventral) vs  our model  vs  AlexNet fc7(raw)

Usage:
  python scripts/generalization/retrieval_viz.py                      # defaults: qq145, subj 1, 6 queries
  python scripts/generalization/retrieval_viz.py --model path/to/model.py --queries 3,10,42 --k 10
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "benchmark"))   # AlexNetSRP + vendored SRP

from harness.loaders.shared import processed_dir                 # noqa: E402

# The NSD ~0.44 champion (best scored model across the search archive). The DB path column is stale
# (points at a nsd__2026-06-28 dir that was never archived); the file actually lives here.
DEFAULT_MODEL = os.path.join(REPO, "runs", "nsd__2026-06-27", "models", "qq145.py")
IMSIZE = 128                     # the NSD preset imsize the search models were scored at (do NOT change)


def load_model(path):
    """Import a model .py and return its ``MODEL`` instance."""
    spec = importlib.util.spec_from_file_location("retr_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def load_gallery(subject: int, device: str):
    """The shared-1000 held-out gallery: raw RGB (for display) + model-input X (imsize, [0,1], HWC
    flat, row-aligned to nsd_ids) -- built exactly as harness.loaders.vision.nsd._build_X does."""
    import h5py

    pt = os.path.join(processed_dir("nsd"), "neural", f"subj{subject:02d}.pt")
    blob = torch.load(pt, weights_only=False)
    nsd_ids = np.asarray(blob["nsd_ids"])
    shared = np.asarray(blob["shared"], dtype=bool)              # the shared-1000 test images
    ids = nsd_ids[shared]

    images_h5 = os.path.join(processed_dir("nsd"), "stimuli", "images.h5")
    raw = np.empty((len(ids), 425, 425, 3), dtype=np.uint8)
    X = np.empty((len(ids), IMSIZE * IMSIZE * 3), dtype="float32")
    with h5py.File(images_h5, "r") as h5:
        for s in range(0, len(ids), 500):
            chunk = ids[s:s + 500]
            imgs = np.stack([h5[str(int(i))][:] for i in chunk])         # (b,425,425,3) uint8
            raw[s:s + 500] = imgs
            t = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float() / 255.0
            t = torch.nn.functional.interpolate(t, size=(IMSIZE, IMSIZE), mode="bilinear",
                                                align_corners=False)
            X[s:s + 500] = t.permute(0, 2, 3, 1).reshape(t.shape[0], -1).cpu().numpy()
    return ids, raw, X


def load_brain(subject: int):
    """Recorded neural responses over the shared-1000 gallery, per stream, restricted to the reliable
    voxels the reward scores (NC_r >= NSD_NCSNR_MIN, absolute NC_r2 floor). Returns
    ``{"early": (N, v_e), "ventral": (N, v_v)}`` -- the brain's own representation for retrieval,
    row-aligned to the gallery (same shared mask + order as load_gallery)."""
    thresh = float(os.environ.get("NSD_NCSNR_MIN", "0.2"))
    pt = os.path.join(processed_dir("nsd"), "neural", f"subj{subject:02d}.pt")
    blob = torch.load(pt, weights_only=False)
    rois, shared = blob["rois"], np.asarray(blob["shared"], dtype=bool)
    nc = blob.get("ncsnr")
    out = {}
    for stream in ("early", "ventral"):
        Y = np.asarray(rois[stream], dtype="float32")[shared]          # (N, v)
        if nc is not None:
            ncsnr = np.asarray(nc[stream], dtype="float32")
            nc_r2 = ncsnr ** 2 / (ncsnr ** 2 + 1.0 / 3)                # n=3 repeats
            keep = (np.sqrt(nc_r2) >= thresh) & (nc_r2 >= 0.01)
            Y = Y[:, keep]
        out[stream] = Y
        print(f"  brain {stream}: {Y.shape[1]} reliable voxels")
    return out


def forward_model(model, X, device, batch=256):
    """Run a champion model over X in sample chunks (per-sample invariant) -> (N, D_out) numpy."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    outs = []
    with torch.no_grad():
        for s in range(0, Xt.shape[0], batch):
            outs.append(model(Xt[s:s + batch]).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def alexnet_layer(X, layer: str, dim: int, device: str):
    """One AlexNet tap over X -> (N, d). ``dim`` >= tap width returns it RAW (fc7); ``dim`` < tap
    width sparse-random-projects it (conv4)."""
    from alexnet_baseline import AlexNetSRP

    net = AlexNetSRP(imsize=IMSIZE, layers=[layer], dim=dim, seed=0)
    return net.forward_layers(X)[layer]


def topk(reps: np.ndarray, q: int, k: int) -> np.ndarray:
    """Indices of the top-k gallery items most cosine-similar to query row ``q`` (excludes itself)."""
    R = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8)
    sims = R @ R[q]
    sims[q] = -np.inf
    return np.argsort(-sims)[:k]


def figure(raw, q, rows, out_path, k, title):
    """Grid: one row per (label, neighbour-idx list); col 0 = query (red frame), cols 1.. = neighbours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow, ncol = len(rows), k + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.6 * nrow), squeeze=False)
    fig.suptitle(title, fontsize=13, y=0.99)
    for r, (label, nbrs) in enumerate(rows):
        cells = [q] + list(nbrs)
        for c, idx in enumerate(cells):
            ax = axes[r][c]
            ax.imshow(raw[idx])
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                for sp in ax.spines.values():
                    sp.set_color("red"); sp.set_linewidth(3)
                ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=6)
            else:
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if r == 0:
                ax.set_title("query" if c == 0 else f"#{c}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    """Embed the shared-1000 gallery in brain / champion / AlexNet spaces and write per-query early+ventral top-K retrieval figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="champion model .py (default: NSD qq145)")
    ap.add_argument("--subject", type=int, default=1, help="NSD subject id for the gallery (default 1)")
    ap.add_argument("--k", type=int, default=10, help="neighbours to retrieve (default 10)")
    ap.add_argument("--queries", default=None, help="comma list of gallery indices; default = 6 evenly spaced")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "retrieval"), help="output dir")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    print(f"device={device}  model={a.model}  subject={a.subject}")

    ids, raw, X = load_gallery(a.subject, device)
    n = len(ids)
    print(f"gallery: {n} shared-1000 held-out images")

    queries = ([int(x) for x in a.queries.split(",")] if a.queries
               else [int(round(i)) for i in np.linspace(0, n - 1, 6)])

    print("brain: recorded responses per stream ...")
    brain = load_brain(a.subject)
    print("forward: champion model ...")
    rep_model = forward_model(load_model(a.model), X, device)
    print(f"  our model  D_out={rep_model.shape[1]}")
    print("forward: AlexNet conv4 -> SRP 2048 ...")
    rep_conv4 = alexnet_layer(X, "conv4", dim=2048, device=device)
    print(f"  conv4(SRP) D={rep_conv4.shape[1]}")
    print("forward: AlexNet fc7 -> raw ...")
    rep_fc7 = alexnet_layer(X, "fc7", dim=8192, device=device)          # dim>4096 => fc7 kept raw
    print(f"  fc7(raw)   D={rep_fc7.shape[1]}")

    for q in queries:
        early_rows = [
            ("brain (early)", topk(brain["early"], q, a.k)),
            ("our model", topk(rep_model, q, a.k)),
            ("AlexNet conv4 (SRP)", topk(rep_conv4, q, a.k)),
        ]
        ventral_rows = [
            ("brain (ventral)", topk(brain["ventral"], q, a.k)),
            ("our model", topk(rep_model, q, a.k)),
            ("AlexNet fc7 (raw)", topk(rep_fc7, q, a.k)),
        ]
        figure(raw, q, early_rows, os.path.join(a.out, f"early_visual_q{q:03d}.png"),
               a.k, f"Early visual stream -- top-{a.k} retrieval (query gallery idx {q}, nsd_id {ids[q]})")
        figure(raw, q, ventral_rows, os.path.join(a.out, f"ventral_stream_q{q:03d}.png"),
               a.k, f"Ventral stream -- top-{a.k} retrieval (query gallery idx {q}, nsd_id {ids[q]})")
        print(f"  q{q:03d} -> early_visual_q{q:03d}.png, ventral_stream_q{q:03d}.png")

    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
