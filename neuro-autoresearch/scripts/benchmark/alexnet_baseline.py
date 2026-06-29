#!/usr/bin/env python3
"""ImageNet-AlexNet feature-hierarchy baseline -- a fixed comparison point for the search.

An ImageNet-pretrained AlexNet is run forward (its weights are FROZEN -- never trained here, exactly
like the search's untrained models). The post-ReLU activations of each layer (the 5 conv stages +
fc6 + fc7) are each **sparse-random-projected to 4096 dims** and concatenated into one per-sample
representation, which is then fed through the EXACT harness readout + ROI machinery the search uses:
the same ``harness.loaders.load_split`` (so same images, same shared-1000 split, same early+ventral
target voxels, same per-ROI masks), the same ``harness.ridge.ridge_predict`` readout (``NEURO_READOUT``),
and the same ``harness.roi.roi_breakdown``. So ``test_corr`` and the per-ROI numbers are directly
comparable to the search-DB champions' ``roi_corr``.

This is the body of ``harness.reward.evaluate_on_split`` reusing its own helpers -- it only skips that
function's ``MAX_DOUT`` guard, which is a *proposer* constraint (7x4096 = 28672 > 10000), not a readout
limit: fast ridge handles p>n via its dual form. Nothing under ``harness/`` is modified.

    python scripts/benchmark/alexnet_baseline.py                       # nsd subj1, early+ventral
    NEURO_SUBJECT=2 python scripts/benchmark/alexnet_baseline.py       # another subject
    python scripts/benchmark/alexnet_baseline.py --layers conv1 conv2 conv3 conv4 conv5
    python scripts/benchmark/alexnet_baseline.py --dim 2048 --batch 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# scripts/benchmark/alexnet_baseline.py -> repo root is THREE levels up.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)            # so the vendored sparse_random_projection.py resolves

# ImageNet normalisation (the stats AlexNet's pretrained weights expect).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Taps of the AlexNet hierarchy. (sub-module, index) into the named child. conv1-5 + fc6/fc7 are
# post-ReLU; fc8 is the final logits (no ReLU after it). conv taps are huge (>4096) so they get the
# SRP; fc taps are already <=4096 and are used raw.
_LAYERS = {
    "conv1": ("features", 1),
    "conv2": ("features", 4),
    "conv3": ("features", 7),
    "conv4": ("features", 9),
    "conv5": ("features", 11),
    "fc6": ("classifier", 2),
    "fc7": ("classifier", 5),
    "fc8": ("classifier", 6),
}


def _srp_cache_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "srp_cache")


class AlexNetSRP:
    """X (N, imsize*imsize*3) [0,1] HWC pixels -> per-layer AlexNet features.

    Frozen ImageNet AlexNet. CONV taps (D>``dim``) are sparse-random-projected to ``dim`` with the
    visreps SRP (``_get_srp``); FC taps (D<=``dim``) are used RAW (no projection needed). The SRP
    matrices are keyed by input dim D and built once, so train and test share the same projection.
    Forward runs in image batches to bound memory.
    """

    idea = "ImageNet-pretrained AlexNet; conv layers sparse-random-projected to 4096 (fc layers raw)."
    summary = ("Frozen ImageNet AlexNet feature hierarchy (conv1-5 + fc6/fc7/fc8); conv taps SRP->4096 "
               "(visreps SparseRandomProjection), fc taps raw. Trained net, never trained here.")

    def __init__(self, imsize, layers, dim=4096, density=None, resize=224, batch=256, seed=0,
                 pretrained=True):
        from torchvision.models import AlexNet_Weights, alexnet

        tag = "" if pretrained else "-untrained"
        self.name = f"alexnet{tag}-srp{dim}-{'+'.join(layers)}"
        self.pretrained = pretrained
        self.imsize, self.layers, self.dim = imsize, layers, dim
        self.density, self.resize, self.batch, self.seed = density, resize, batch, seed
        self._srp = {}                       # input-dim D -> fitted SRP (shared across calls + layers)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        # Pretrained = the ImageNet-trained hierarchy; untrained = the SAME architecture at random
        # (Kaiming) init -- the trained-vs-untrained control. Untrained init is seeded for repeatability.
        if pretrained:
            self.net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        else:
            torch.manual_seed(seed)
            self.net = alexnet(weights=None)
        self.net = self.net.eval().to(dev)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.n_params = 0                    # never trained -> no fit params (the eval is the ridge)
        # Resolve the tap modules and attach capture hooks.
        self._caught = {}
        for name in layers:
            sub, idx = _LAYERS[name]
            getattr(self.net, sub)[idx].register_forward_hook(self._capture(name))

    def _capture(self, name):
        def hook(_module, _inp, out):
            self._caught[name] = out.detach().flatten(1)   # (b, feat)
        return hook

    def _project(self, feats_np):
        """SRP a conv tap (D>dim) to ``dim``; return an fc tap (D<=dim) unchanged.

        The SRP transformer is the vendored visreps ``get_srp_transformer`` (conv layers only)."""
        D = feats_np.shape[1]
        if D <= self.dim:
            return feats_np                  # fc layers: already <=4096, used raw
        if D not in self._srp:
            from sparse_random_projection import get_srp_transformer
            self._srp[D] = get_srp_transformer(D, self.dim, self.density, self.seed,
                                               cache_dir=_srp_cache_dir())
        return self._srp[D].transform(feats_np)

    def forward_layers(self, X) -> dict:
        """``{layer: (N, d) float32}`` -- conv taps SRP'd to ``dim``, fc taps raw."""
        X = torch.as_tensor(X)
        N = X.shape[0]
        im = self.imsize
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        chunks = {name: [] for name in self.layers}
        with torch.no_grad():
            for s in range(0, N, self.batch):
                xb = X[s:s + self.batch].to(self.device).float().reshape(-1, im, im, 3)
                xb = xb.permute(0, 3, 1, 2).contiguous()                 # (b,3,im,im)
                xb = F.interpolate(xb, size=(self.resize, self.resize), mode="bilinear",
                                   align_corners=False)
                xb = (xb - mean) / std
                self._caught.clear()
                self.net(xb)
                for name in self.layers:
                    chunks[name].append(self._project(self._caught[name].cpu().numpy()))
        return {name: np.concatenate(chunks[name], axis=0).astype(np.float32) for name in self.layers}

    def __call__(self, X):
        out = self.forward_layers(X)
        cat = np.concatenate([out[name] for name in self.layers], axis=1)  # (N, sum of per-layer dims)
        return torch.from_numpy(cat).to(self.device)


def _readout(A_train, A_test, Ytr, Yte, roi_masks, nc, DATASET, R, roi_breakdown, device):
    """Run the harness reward-region readout on a representation. Returns a record dict with the
    noise-normalized reward (Algonauts r^2/NC_r2 -- the fraction of EXPLAINABLE variance the model
    captured, the harness/search metric): ``test_corr`` + per-ROI ``roi_corr``.

    This is the body of ``harness.reward.evaluate_on_split`` reusing its helpers (same reward-region
    target selection, same fast/himalaya ridge, same per-ROI breakdown, same per-voxel ceiling ``nc``)
    -- minus only the MAX_DOUT proposer guard, which does not apply to a fixed baseline."""
    from harness.ridge import ridge_predict

    mask = R.reward_region_mask(roi_masks, DATASET.reward_roi)
    if mask is not None:
        roi_masks = {k: np.asarray(v, dtype=bool)[mask] for k, v in roi_masks.items()}
        if nc is not None:
            nc = nc[mask]
        if torch.is_tensor(Ytr):
            mask = torch.as_tensor(mask, device=Ytr.device)
    Yv_train, Yv_test = R.select_targets(Ytr, mask), R.select_targets(Yte, mask)

    pred_train, pred_test = ridge_predict(A_train, A_test, Yv_train, device=device)
    test_n = R._noise_normalize(R._corr_per_column(pred_test, R._to_np(Yv_test)), nc)
    train_n = R._noise_normalize(R._corr_per_column(pred_train, R._to_np(Yv_train)), nc)
    return {
        "d_out": int(A_train.shape[1]),
        "test_corr": round(float(test_n.mean()), 4), "train_corr": round(float(train_n.mean()), 4),
        "gap": round(float(train_n.mean() - test_n.mean()), 4),
        "roi_corr": roi_breakdown(roi_masks, train_n, test_n),       # noise-normalized r^2/NC_r2
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "nsd"))
    p.add_argument("--layers", nargs="+", default=list(_LAYERS), choices=list(_LAYERS),
                   help="which AlexNet taps to use (default: all 8 -- conv1-5, fc6, fc7, fc8)")
    p.add_argument("--concat", action="store_true",
                   help="concatenate the taps into ONE representation instead of scoring each layer "
                        "separately (the default is per-layer, the brain-score protocol).")
    p.add_argument("--dim", type=int, default=4096, help="SRP target dim for conv taps (default 4096)")
    p.add_argument("--density", type=float, default=None,
                   help="SRP density (default None -> sklearn 'auto' = 1/sqrt(D), as in visreps)")
    p.add_argument("--resize", type=int, default=224, help="image size fed to AlexNet (default 224)")
    p.add_argument("--batch", type=int, default=256, help="forward batch size (default 256)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", choices=["trained", "untrained", "both"], default="trained",
                   help="ImageNet-pretrained weights ('trained'), random-init same architecture "
                        "('untrained'), or 'both' for the trained-vs-untrained control (one data load).")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = a.dataset
    from harness.datasets import DATASET
    from harness.loaders import load_split
    from harness import reward as R
    from harness.roi import roi_breakdown

    subj = os.environ.get("NEURO_SUBJECT", "1")
    mode = "concat" if a.concat else "per-layer"
    print(f"=== AlexNet-SRP baseline ({mode}) | {a.dataset} subj={subj} "
          f"| reward_roi={DATASET.reward_roi} | layers={'+'.join(a.layers)} | conv_srp_dim={a.dim} "
          f"| weights={a.weights} ===")

    # Same split the search/harness uses: same images, shared-1000 hold-out, target voxels, ROI masks.
    # Loaded ONCE and reused across weight variants (the load is the slow step).
    split = load_split(DATASET)
    Xtr, Xte, Ytr, Yte, roi_masks, nc = split   # nc = per-voxel R^2 noise ceiling (None off-NSD);
    #   the readout below reports the noise-normalized reward (r^2/NC_r2) + raw r side by side.
    device = R._readout_device()
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(device), Xte.to(device)

    def run(A_train, A_test, label):
        rec = _readout(torch.from_numpy(A_train).to(device), torch.from_numpy(A_test).to(device),
                       Ytr, Yte, dict(roi_masks), nc, DATASET, R, roi_breakdown, device)
        rec["label"] = label
        rois = list(rec["roi_corr"])
        print(f"\n  [{label}]  REWARD r²/NC² test={rec['test_corr']:.4f} train={rec['train_corr']:.4f} "
              f"gap={rec['gap']:.4f}  (D_out={rec['d_out']})")
        print("  " + "".join(f"{r:>9}" for r in rois))
        print("  r²/NC²  " + "".join(f"{rec['roi_corr'][r]['test']:>9.4f}" for r in rois))
        return rec

    def evaluate_variant(pretrained):
        """Build the (frozen) AlexNet at the given init, run forward + score every tap. Returns the
        model + its records, so trained/untrained share the one expensive data load above."""
        wlabel = "trained" if pretrained else "untrained"
        model = AlexNetSRP(DATASET.imsize, a.layers, dim=a.dim, density=a.density, resize=a.resize,
                           batch=a.batch, seed=a.seed, pretrained=pretrained)
        t0 = time.time()
        feat_tr, feat_te = model.forward_layers(Xtr), model.forward_layers(Xte)
        print(f"\n--- {wlabel} AlexNet ---  forward {time.time() - t0:.1f}s   "
              f"n_train={Ytr.shape[0]} n_test={Yte.shape[0]} n_targets={Ytr.shape[1]}")
        recs = []
        if a.concat:
            cat_tr = np.concatenate([feat_tr[n] for n in a.layers], axis=1)
            cat_te = np.concatenate([feat_te[n] for n in a.layers], axis=1)
            recs.append(run(cat_tr, cat_te, f"{wlabel}|concat:" + "+".join(a.layers)))
        else:
            # Each tap scored on its own -- the brain-score protocol; sidesteps the all-concat overfit.
            for name in a.layers:
                recs.append(run(feat_tr[name], feat_te[name], f"{wlabel}|{name}"))
        for r in recs:
            r["weights"] = wlabel
        best = max(recs, key=lambda r: r["test_corr"])
        print(f"  best {wlabel} layer (reward): {best['label']}  test_corr={best['test_corr']:.4f}")
        return model, recs

    variants = ["trained", "untrained"] if a.weights == "both" else [a.weights]
    t_all = time.time()
    records, model_names = [], []
    for w in variants:
        model, recs = evaluate_variant(w == "trained")
        records.extend(recs)
        model_names.append(model.name)

    if a.weights == "both":
        bt = max((r for r in records if r["weights"] == "trained"), key=lambda r: r["test_corr"])
        bu = max((r for r in records if r["weights"] == "untrained"), key=lambda r: r["test_corr"])
        print(f"\n  === trained-vs-untrained (best layer) ===")
        print(f"    trained  : {bt['label']:>28}  test_corr={bt['test_corr']:.4f}")
        print(f"    untrained: {bu['label']:>28}  test_corr={bu['test_corr']:.4f}")
        print(f"    delta (trained - untrained): {bt['test_corr'] - bu['test_corr']:+.4f}")
    print(f"\n  total {time.time() - t_all:.1f}s")

    if not a.no_save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                           f"alexnet_{mode}_{a.weights}_{a.dataset}_subj{subj}_{stamp}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"model": "+".join(model_names), "mode": mode, "weights": a.weights,
                       "dataset": a.dataset, "subject": subj, "reward_roi": DATASET.reward_roi,
                       "layers": a.layers, "srp_dim": a.dim, "records": records}, f, indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
