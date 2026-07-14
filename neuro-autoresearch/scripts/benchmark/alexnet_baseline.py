#!/usr/bin/env python3
"""ImageNet-AlexNet feature-hierarchy baseline -- a fixed comparison point for the search.

An ImageNet-pretrained AlexNet is run forward (its weights are FROZEN -- never trained here, exactly
like the search's untrained models). The post-ReLU activations of each layer (the 5 conv stages +
fc6 + fc7 + fc8) are each **sparse-random-projected to 2048 dims** and concatenated into one
per-sample representation -- the WHOLE feature hierarchy at once, NOT a cherry-picked best layer
(``--per-layer`` opts into the old brain-score-style per-layer scoring). This concatenated
representation is then fed through the EXACT harness readout + ROI machinery the search uses:
the same ``harness.loaders.load_split`` (so same images, same shared-1000 split, same early+ventral
target voxels, same per-ROI masks), the same ``harness.ridge.ridge_predict`` readout (``NEURO_READOUT``),
and the same ``harness.roi.roi_breakdown``. So ``test_corr`` and the per-ROI numbers are directly
comparable to the search-DB champions' ``roi_corr``.

This is the body of ``harness.reward.evaluate_on_split`` reusing its own helpers -- it only skips that
function's ``MAX_DOUT`` guard, which is a *proposer* constraint (8x2048 > 10000), not a readout
limit: fast ridge handles p>n via its dual form. Nothing under ``harness/`` is modified.

    python scripts/benchmark/alexnet_baseline.py                       # nsd subj1, concat all layers
    NEURO_SUBJECT=2 python scripts/benchmark/alexnet_baseline.py       # another subject
    python scripts/benchmark/alexnet_baseline.py --weights both        # trained vs untrained control
    python scripts/benchmark/alexnet_baseline.py --per-layer           # old per-layer (brain-score) scoring
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
# post-ReLU; fc8 is the final logits (no ReLU after it). Any tap with D>``dim`` gets the SRP; taps
# already <=``dim`` (e.g. fc8's 1000 logits at the default dim=2048) are used raw.
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
    """Where the fitted SRP matrices are cached (keyed by D/k/density/seed), so runs reuse them."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "srp_cache")


class AlexNetSRP:
    """X (N, imsize*imsize*3) [0,1] HWC pixels -> per-layer AlexNet features.

    Frozen ImageNet AlexNet. Taps with D>``dim`` are sparse-random-projected to ``dim`` with the
    visreps SRP (``_get_srp``); taps already D<=``dim`` are used RAW (no projection needed). The SRP
    matrices are keyed by input dim D and built once, so train and test share the same projection.
    Forward runs in image batches to bound memory.
    """

    idea = "ImageNet-pretrained AlexNet; every layer sparse-random-projected to 2048 and concatenated."
    summary = ("Frozen ImageNet AlexNet feature hierarchy (conv1-5 + fc6/fc7/fc8); each tap SRP->2048 "
               "(visreps SparseRandomProjection) and concatenated. Trained net, never trained here.")

    def __init__(self, imsize, layers, dim=2048, density=None, resize=224, batch=256, seed=0,
                 pretrained=True):
        """Load the frozen AlexNet (ImageNet weights, or seeded random init when ``pretrained``
        is False), freeze all params, and attach a capture hook to each requested tap."""
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
        """Build a forward hook that stashes this tap's post-ReLU activations under ``name``."""
        def hook(_module, _inp, out):
            self._caught[name] = out.detach().flatten(1)   # (b, feat)
        return hook

    def _project(self, feats_np):
        """SRP a conv tap (D>dim) to ``dim``; return an fc tap (D<=dim) unchanged.

        The SRP transformer is the vendored visreps ``get_srp_transformer`` (conv layers only)."""
        D = feats_np.shape[1]
        if D <= self.dim:
            return feats_np                  # tap already <=dim (e.g. fc8 logits): used raw
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
        """Forward + concatenate every requested tap into one (N, sum-of-dims) representation."""
        out = self.forward_layers(X)
        cat = np.concatenate([out[name] for name in self.layers], axis=1)  # (N, sum of per-layer dims)
        return torch.from_numpy(cat).to(self.device)


def _readout(A_train, A_test, Ytr, Yte, roi_masks, nc, DATASET, R, roi_breakdown, device):
    """Run the harness reward-region readout on a representation. Returns a record dict with the
    noise-normalized reward (Algonauts r^2/NC_r2 -- the fraction of EXPLAINABLE variance the model
    captured, the harness/search metric): ``test_corr`` + per-ROI ``roi_corr``.

    This is the body of ``harness.reward.evaluate_on_split`` reusing its helpers (same reward-region
    target selection, same fast/himalaya ridge, same per-ROI breakdown, same per-voxel ceiling ``nc``,
    AND the same ``cv_folds`` branch -- rotating K-fold CV when the preset sets it, e.g. nnn/ieeg, so
    the baseline is scored under the EXACT protocol the search DB uses) -- minus only the MAX_DOUT
    proposer guard, which does not apply to a fixed baseline."""
    from harness.ridge import ridge_predict

    mask = R.reward_region_mask(roi_masks, DATASET.reward_roi)
    if mask is not None:
        roi_masks = {k: np.asarray(v, dtype=bool)[mask] for k, v in roi_masks.items()}
        if nc is not None:
            nc = nc[mask]
        if torch.is_tensor(Ytr):
            mask = torch.as_tensor(mask, device=Ytr.device)
    Yv_train, Yv_test = R.select_targets(Ytr, mask), R.select_targets(Yte, mask)

    # cv_folds>1 (nnn/ieeg): reproduce the reward's rotating K-fold CV over the FULL image set so the
    # baseline's test_corr is apples-to-apples with the search DB; else the single train/test split.
    if getattr(DATASET, "cv_folds", 0) > 1:
        A_full = torch.cat([A_train, A_test], 0)
        Y_full = torch.cat([torch.as_tensor(Yv_train), torch.as_tensor(Yv_test)], 0)
        train_n, test_n = R._cv_column_r(A_full, Y_full, DATASET.cv_folds, device, nc)
    else:
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
    """Score the frozen AlexNet feature hierarchy through the harness readout and print/save the
    per-ROI ``test_corr`` (concat or per-layer, one or both weight inits) for comparison to champions."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.environ.get("NEURO_DATASET", "nsd"))
    p.add_argument("--layers", nargs="+", default=list(_LAYERS), choices=list(_LAYERS),
                   help="which AlexNet taps to use (default: all 8 -- conv1-5, fc6, fc7, fc8)")
    p.add_argument("--per-layer", action="store_true",
                   help="score each tap separately (old brain-score protocol) instead of the default: "
                        "all taps SRP'd and concatenated into ONE representation.")
    p.add_argument("--dim", type=int, default=2048, help="SRP target dim per tap (default 2048)")
    p.add_argument("--density", type=float, default=None,
                   help="SRP density (default None -> sklearn 'auto' = 1/sqrt(D), as in visreps)")
    p.add_argument("--resize", type=int, default=224, help="image size fed to AlexNet (default 224)")
    p.add_argument("--batch", type=int, default=256, help="forward batch size (default 256)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", choices=["trained", "untrained", "both"], default="trained",
                   help="ImageNet-pretrained weights ('trained'), random-init same architecture "
                        "('untrained'), or 'both' for the trained-vs-untrained control (one data load).")
    p.add_argument("--subject", default=None,
                   help="NEURO_SUBJECT; a comma list (e.g. 1,3,4) is scored per-subject and averaged "
                        "(equal weight), like the search reward. Default: dataset default subject.")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = a.dataset
    if a.subject is not None:
        os.environ["NEURO_SUBJECT"] = str(a.subject)
    from harness.datasets import DATASET, with_subject
    from harness.loaders import load_split
    from harness import reward as R
    from harness.roi import roi_breakdown

    from subjects import subject_tokens, common_roi_keys, mean_scalar
    subjects = subject_tokens(a.subject if a.subject is not None else os.environ.get("NEURO_SUBJECT"))
    subj_label = ",".join(subjects) if subjects else "default"
    mode = "per-layer" if a.per_layer else "concat"
    device = R._readout_device()
    print(f"=== AlexNet-SRP baseline ({mode}) | {a.dataset} subj={subj_label} "
          f"| reward_roi={DATASET.reward_roi} | layers={'+'.join(a.layers)} | conv_srp_dim={a.dim} "
          f"| weights={a.weights} ===")

    def score_split(dset):
        """Load ``dset``'s split (same images/hold-out/voxels/ROI masks the search uses) and score
        every requested weight-variant/layer on it. Returns ``(records, model_names)``. The split load
        is the slow step -- shared across weight variants."""
        Xtr, Xte, Ytr, Yte, roi_masks, nc = load_split(dset)
        if torch.is_tensor(Xtr):
            Xtr, Xte = Xtr.to(device), Xte.to(device)

        def run(A_train, A_test, label):
            """Score one representation through the readout, print its reward + per-ROI row, return the rec."""
            rec = _readout(torch.from_numpy(A_train).to(device), torch.from_numpy(A_test).to(device),
                           Ytr, Yte, dict(roi_masks), nc, dset, R, roi_breakdown, device)
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
            model = AlexNetSRP(dset.imsize, a.layers, dim=a.dim, density=a.density, resize=a.resize,
                               batch=a.batch, seed=a.seed, pretrained=pretrained)
            t0 = time.time()
            feat_tr, feat_te = model.forward_layers(Xtr), model.forward_layers(Xte)
            print(f"\n--- {wlabel} AlexNet ---  forward {time.time() - t0:.1f}s   "
                  f"n_train={Ytr.shape[0]} n_test={Yte.shape[0]} n_targets={Ytr.shape[1]}")
            recs = []
            if not a.per_layer:
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
            sel = "best layer" if a.per_layer else "concat"
            print(f"  {wlabel} ({sel}) reward: {best['label']}  test_corr={best['test_corr']:.4f}")
            return model, recs

        variants = ["trained", "untrained"] if a.weights == "both" else [a.weights]
        recs, names = [], []
        for w in variants:
            model, rv = evaluate_variant(w == "trained")
            recs.extend(rv)
            names.append(model.name)
        return recs, names

    def mean_records(per_subject):
        """Average per-subject records with EQUAL weight per subject, grouped by their stable ``label``
        (mirrors ``harness.reward.evaluate_subjects``): mean test/train/gap + per-ROI train/test over
        the ROIs present for every subject."""
        subs = list(per_subject)
        by_label: dict = {}
        for s in subs:
            for r in per_subject[s]:
                by_label.setdefault(r["label"], {})[s] = r
        out = []
        for label, recs in by_label.items():
            rs = list(recs.values())
            roi_keys = common_roi_keys([r["roi_corr"] for r in rs])
            out.append({
                "label": label, "weights": rs[0]["weights"], "d_out": rs[0]["d_out"],
                "test_corr": mean_scalar(r["test_corr"] for r in rs),
                "train_corr": mean_scalar(r["train_corr"] for r in rs),
                "gap": mean_scalar(r["gap"] for r in rs),
                "roi_corr": {k: {"train": mean_scalar(recs[s]["roi_corr"][k]["train"] for s in subs),
                                 "test": mean_scalar(recs[s]["roi_corr"][k]["test"] for s in subs)}
                             for k in roi_keys},
                "subjects": subs,
            })
        return out

    t_all = time.time()
    if len(subjects) <= 1:
        # Single subject (or the dataset default): DATASET is already bound to it at import.
        records, model_names = score_split(DATASET)
    else:
        # Per-subject then equal-weight mean -- the search reward's multi-subject protocol.
        per_subject, model_names = {}, []
        for s in subjects:
            dset = with_subject(DATASET, s)
            print(f"\n----- subject {s} ({dset.study}) -----")
            per_subject[s], model_names = score_split(dset)
        records = mean_records(per_subject)
        print(f"\n  === mean over subjects {subj_label} ===")
        for r in records:
            print(f"    {r['label']:>34}  test_corr={r['test_corr']:.4f}")

    if a.weights == "both":
        bt = max((r for r in records if r["weights"] == "trained"), key=lambda r: r["test_corr"])
        bu = max((r for r in records if r["weights"] == "untrained"), key=lambda r: r["test_corr"])
        print(f"\n  === trained-vs-untrained ({mode}) ===")
        print(f"    trained  : {bt['label']:>28}  test_corr={bt['test_corr']:.4f}")
        print(f"    untrained: {bu['label']:>28}  test_corr={bu['test_corr']:.4f}")
        print(f"    delta (trained - untrained): {bt['test_corr'] - bu['test_corr']:+.4f}")
    print(f"\n  total {time.time() - t_all:.1f}s")

    if not a.no_save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                           f"alexnet_{mode}_{a.weights}_{a.dataset}_subj{subj_label}_{stamp}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"model": "+".join(model_names), "mode": mode, "weights": a.weights,
                       "dataset": a.dataset, "subject": subj_label, "reward_roi": DATASET.reward_roi,
                       "layers": a.layers, "srp_dim": a.dim, "records": records}, f, indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
