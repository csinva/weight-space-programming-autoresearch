#!/usr/bin/env python3
"""Best-layer encoding benchmark: standard pretrained vision models vs search champions.

For each model the script:
  1. Runs the frozen model forward over train+test images, capturing multiple layer taps.
  2. SRP-projects any tap with D > dim down to `dim`.
  3. Selects the best layer on an 80/20 val split from TRAIN only (no test leakage).
  4. Refits that layer on full train, reports test_corr + per-ROI breakdown (noise-normalized,
     same metric as the harness / search DB).

Scores are directly comparable to search DB champions.

Usage:
    python experiments/evaluation/vision_baselines.py --model alexnet --dataset nsd --subject 1
    python experiments/evaluation/vision_baselines.py --model all --dataset nsd --subject 1
    python experiments/evaluation/vision_baselines.py --model resnet50 --dataset tvsd
    python experiments/evaluation/vision_baselines.py --model clip --dataset nnn
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)   # so `from srp import ...` resolves

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_CLIP_MEAN     = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# SRP helper (reuses srp.py)
# ---------------------------------------------------------------------------

def _srp(feats_np: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """SRP project (N, D) -> (N, dim). D<=dim is returned as-is."""
    if feats_np.shape[1] <= dim:
        return feats_np.astype(np.float32)
    from srp import get_srp_transformer
    cache = os.path.join(HERE, "results", "srp_cache")
    t = get_srp_transformer(feats_np.shape[1], dim, None, seed, cache_dir=cache)
    return t.transform(feats_np).astype(np.float32)


def _standardize(A_tr, A_te):
    """Per-column z-score with train stats so feature scale doesn't distort alpha selection."""
    mu = A_tr.mean(0, keepdims=True)
    sd = A_tr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (A_tr - mu) / sd, (A_te - mu) / sd


def _preprocess(X_flat, imsize, resize, mean, std, device, batch=128):
    """Flat HWC [0,1] pixels -> batched (N,3,resize,resize) tensor on device, normalized."""
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)
    if torch.is_tensor(X_flat):
        X = X_flat.cpu().float()
    else:
        X = torch.as_tensor(np.asarray(X_flat), dtype=torch.float32)
    out = []
    for s in range(0, len(X), batch):
        xb = X[s:s+batch].to(device).reshape(-1, imsize, imsize, 3)
        xb = xb.permute(0, 3, 1, 2)
        if resize != imsize:
            xb = F.interpolate(xb, size=(resize, resize), mode="bilinear", align_corners=False)
        xb = (xb - mean_t) / std_t
        out.append(xb)
    return torch.cat(out, 0)  # (N, 3, resize, resize)


# Region groups each dataset selects a best layer over (brain-score protocol). NSD needs the two
# named streams (its roi_masks also carry overlapping sub-ROIs); tvsd/nnn roi_masks are already a
# clean partition, so None => use every roi_masks key as its own group.
SELECT_GROUPS = {"nsd": ["early", "ventral"], "tvsd": ["V1", "V4", "IT"], "nnn": None}


# ---------------------------------------------------------------------------
# Forward extractors: each returns {layer_name: (N, dim) float32 np}
# ---------------------------------------------------------------------------

def _hook_forward(net, tap_map: dict, X_tensor, device, batch=64):
    """Run `net` forward in batches, capturing taps via forward hooks.

    tap_map: {name: module}  — each module gets a hook that stores output.flat.
    Returns {name: (N, D)} float32 cpu numpy.
    """
    caught = {n: [] for n in tap_map}
    handles = []
    for name, mod in tap_map.items():
        def _hook(m, inp, out, n=name):
            caught[n].append(out.detach().flatten(1).cpu().float())
        handles.append(mod.register_forward_hook(_hook))
    N = X_tensor.shape[0]
    with torch.no_grad():
        for s in range(0, N, batch):
            net(X_tensor[s:s+batch].to(device))
    for h in handles:
        h.remove()
    return {n: torch.cat(caught[n], 0).numpy() for n in tap_map}


def extract_alexnet(X_flat, imsize, dim, seed, device, pretrained=True, batch=128):
    """Frozen ImageNet AlexNet -> {conv1-5, fc6-8: (N, dim)} features, each tap SRP'd to ``dim``."""
    from torchvision.models import alexnet, AlexNet_Weights
    net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None).eval().to(device)
    net.requires_grad_(False)
    Xt = _preprocess(X_flat, imsize, 224, _IMAGENET_MEAN, _IMAGENET_STD, device, batch)
    taps = {
        "conv1": net.features[1],  "conv2": net.features[4],
        "conv3": net.features[7],  "conv4": net.features[9],
        "conv5": net.features[11], "fc6":   net.classifier[2],
        "fc7":   net.classifier[5],"fc8":   net.classifier[6],
    }
    raw = _hook_forward(net, taps, Xt, device, batch)
    del net
    return {n: _srp(v, dim, seed) for n, v in raw.items()}


def extract_resnet50(X_flat, imsize, dim, seed, device, pretrained=True, batch=128):
    """Frozen ImageNet ResNet-50 -> {layer1-4 + avgpool: (N, dim)} features, each SRP'd to ``dim``."""
    from torchvision.models import resnet50, ResNet50_Weights
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None).eval().to(device)
    net.requires_grad_(False)
    Xt = _preprocess(X_flat, imsize, 224, _IMAGENET_MEAN, _IMAGENET_STD, device, batch)
    taps = {
        "layer1": net.layer1, "layer2": net.layer2,
        "layer3": net.layer3, "layer4": net.layer4,
    }
    raw = _hook_forward(net, taps, Xt, device, batch)
    # avgpool (2048-d) — no hook needed, just run forward
    avgpool_out = []
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            x = Xt[s:s+batch].to(device)
            x = net.conv1(x); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
            x = net.layer1(x); x = net.layer2(x); x = net.layer3(x); x = net.layer4(x)
            x = net.avgpool(x).flatten(1)
            avgpool_out.append(x.cpu().float().numpy())
    raw["avgpool"] = np.concatenate(avgpool_out, 0)
    del net
    return {n: _srp(v, dim, seed) for n, v in raw.items()}


def extract_vit(X_flat, imsize, dim, seed, device, pretrained=True, batch=64):
    """ViT-B/16 (ImageNet-1k). Tap encoder blocks 3,6,9,12 (CLS token) + final embedding."""
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None).eval().to(device)
    net.requires_grad_(False)
    D = net.hidden_dim  # 768

    Xt = _preprocess(X_flat, imsize, 224, _IMAGENET_MEAN, _IMAGENET_STD, device, batch)
    tap_indices = [2, 5, 8, 11]  # 0-indexed encoder blocks
    # Hooks flatten (N, seq+1, D) -> (N, (seq+1)*D); first D values = CLS token
    taps = {f"block{i+1}": net.encoder.layers[i] for i in tap_indices}
    raw = _hook_forward(net, taps, Xt, device, batch)
    for k in raw:
        raw[k] = raw[k][:, :D]  # CLS token from flattened output

    # Final CLS after encoder + layer norm
    cls_out = []
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            xb = Xt[s:s+batch].to(device)
            xb = net._process_input(xb)
            cls = net.class_token.expand(xb.shape[0], -1, -1)
            xb = torch.cat([cls, xb], dim=1)
            xb = net.encoder(xb)  # (N, seq+1, D)
            cls_out.append(xb[:, 0].cpu().float().numpy())
    raw["cls"] = np.concatenate(cls_out, 0)
    del net
    return {n: _srp(v, dim, seed) for n, v in raw.items()}


def extract_clip(X_flat, imsize, dim, seed, device, pretrained=True, batch=64):
    """CLIP ViT-B/32 via HuggingFace transformers. Taps: blocks 3,6,9,12 + pooled."""
    import logging
    from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
    # Suppress benign "UNEXPECTED key" warnings (CLIP checkpoint includes text weights;
    # CLIPVisionModel only loads the visual encoder and correctly ignores the rest).
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPImageProcessor.from_pretrained(model_id)
    clip_mean = processor.image_mean
    clip_std  = processor.image_std
    model = (CLIPVisionModel.from_pretrained(model_id) if pretrained else
             CLIPVisionModel(CLIPVisionConfig.from_pretrained(model_id)))
    model = model.eval().to(device)
    model.requires_grad_(False)

    Xt = _preprocess(X_flat, imsize, 224, clip_mean, clip_std, device, batch)
    # Probe with one sample to learn the number of hidden states
    with torch.no_grad():
        probe = model(pixel_values=Xt[:1].to(device), output_hidden_states=True)
    num_blocks = len(probe.hidden_states) - 1  # first element is embedding-layer output
    tap_indices = sorted({2, 5, 8, num_blocks - 1})

    layer_out = {f"block{i+1}": [] for i in tap_indices}
    pool_out  = []
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            xb = Xt[s:s+batch].to(device)
            out = model(pixel_values=xb, output_hidden_states=True)
            hs = out.hidden_states   # tuple of (N, seq, D)
            for i in tap_indices:
                feat = hs[i+1][:, 0].cpu().float().numpy()  # CLS token
                layer_out[f"block{i+1}"].append(feat)
            pool_out.append(out.pooler_output.cpu().float().numpy())  # (N, D)

    raw = {k: np.concatenate(v, 0) for k, v in layer_out.items()}
    raw["pooled"] = np.concatenate(pool_out, 0)
    del model
    return {n: _srp(v, dim, seed) for n, v in raw.items()}


def extract_dino(X_flat, imsize, dim, seed, device, pretrained=True, batch=64):
    """DINOv2 ViT-B/14. Taps: blocks 3,6,9,12 + final CLS embedding."""
    from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model
    model_id = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_id)
    net = (Dinov2Model.from_pretrained(model_id) if pretrained else
           Dinov2Model(Dinov2Config.from_pretrained(model_id)))
    net = net.eval().to(device); net.requires_grad_(False)
    Xt = _preprocess(X_flat, imsize, 224, processor.image_mean, processor.image_std, device, batch)
    num_blocks = net.config.num_hidden_layers
    tap_indices = sorted({2, 5, 8, num_blocks - 1})
    layer_out = {f"block{i+1}": [] for i in tap_indices}; emb_out = []
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            xb = Xt[s:s+batch].to(device)
            out = net(pixel_values=xb, output_hidden_states=True)
            for i in tap_indices:
                layer_out[f"block{i+1}"].append(out.hidden_states[i + 1][:, 0].cpu().float().numpy())
            emb_out.append(out.last_hidden_state[:, 0].cpu().float().numpy())

    raw = {k: np.concatenate(v, 0) for k, v in layer_out.items()}
    raw["embedding"] = np.concatenate(emb_out, 0)
    del net
    return {n: _srp(v, dim, seed) for n, v in raw.items()}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "alexnet":  extract_alexnet,
    "resnet50": extract_resnet50,
    "vit":      extract_vit,
    "clip":     extract_clip,
    "dino":     extract_dino,
}


# ---------------------------------------------------------------------------
# Best-layer selection + scoring
# ---------------------------------------------------------------------------

def score_model(model_name, weights, Xtr, Xte, Ytr, Yte, roi_masks, nc,
                DATASET, R, device, dim, seed):
    """Extract all taps, pick the best layer PER brain region on a val split, then report each
    region's chosen-layer test score and the voxel-weighted overall (brain-score protocol)."""
    extractor = MODEL_REGISTRY[model_name]
    t0 = time.time()
    label = f"{model_name}-{weights}"
    print(f"\n--- {label} | extracting features ---")
    torch.manual_seed(seed)
    feats_tr = extractor(Xtr, DATASET.imsize, dim, seed, device, weights == "pretrained")
    torch.manual_seed(seed)
    feats_te = extractor(Xte, DATASET.imsize, dim, seed, device, weights == "pretrained")
    print(f"    forward done in {time.time()-t0:.1f}s  layers={list(feats_tr)}")

    from harness.ridge import ridge_predict

    # Reward region = all loaded columns for every vision preset (reward_roi is a whole-region tag),
    # so roi_masks align directly to the scored columns.
    mask  = R.reward_region_mask(roi_masks, DATASET.reward_roi)
    Yv    = R._to_np(R.select_targets(Ytr, mask))
    Yte_v = R._to_np(R.select_targets(Yte, mask))
    nc_m  = nc[mask] if (nc is not None and mask is not None) else nc
    roi_m = {k: (np.asarray(v, bool)[mask] if mask is not None else np.asarray(v, bool))
             for k, v in roi_masks.items()}

    dataset = os.environ.get("NEURO_DATASET", "nsd")
    groups = SELECT_GROUPS.get(dataset) or list(roi_m)
    groups = [g for g in groups if g in roi_m and roi_m[g].any()]

    def per_voxel_corr(A_fit, A_pred, Y_fit, Y_true):
        """Ridge fit A_fit->Y_fit (train-standardized), predict A_pred; per-voxel r²/NC²."""
        Af, Ap = _standardize(A_fit, A_pred)
        pred = ridge_predict(torch.from_numpy(Af.astype(np.float32)).to(device),
                             torch.from_numpy(Ap.astype(np.float32)).to(device),
                             torch.from_numpy(Y_fit).to(device), device=device)[1]
        return R._noise_normalize(R._corr_per_column(pred, Y_true), nc_m)

    # 1) val pass: per-voxel score for every layer on an 80/20 split of train (no test leakage).
    rng = np.random.default_rng(seed)
    idx = rng.permutation(Yv.shape[0]); cut = int(0.8 * Yv.shape[0])
    i_fit, i_val = idx[:cut], idx[cut:]
    val_corr = {}
    for name, A in feats_tr.items():
        val_corr[name] = per_voxel_corr(A[i_fit], A[i_val], Yv[i_fit], Yv[i_val])
        print(f"      val  {name:>12}: {val_corr[name].mean():.4f}")

    # 2) each region picks the layer with the best mean val score inside it.
    best = {g: max(val_corr, key=lambda L: float(val_corr[L][roi_m[g]].mean())) for g in groups}

    # 3) test pass: per-voxel test score for each distinct winning layer. When the preset sets
    #    cv_folds>1 (nnn/ieeg), score the winner under the reward's OWN rotating K-fold CV over the
    #    FULL image set -- no feature standardization, exactly as harness.reward does -- so the
    #    number is apples-to-apples with the search DB; otherwise the single fit-on-train pass.
    #    (Layer selection above still uses the train-only 80/20 val split, so no test leakage.)
    cv = getattr(DATASET, "cv_folds", 0)
    winners = set(best.values())
    if cv > 1:
        Y_full = np.concatenate([Yv, Yte_v], 0).astype(np.float32)

        def cv_test(L):
            A_full = np.concatenate([feats_tr[L], feats_te[L]], 0).astype(np.float32)
            return R._cv_column_r(torch.from_numpy(A_full).to(device),
                                  torch.from_numpy(Y_full).to(device), cv, device, nc_m)[1]
        test_corr = {L: cv_test(L) for L in winners}
    else:
        test_corr = {L: per_voxel_corr(feats_tr[L], feats_te[L], Yv, Yte_v) for L in winners}

    # 4) assemble: each region's voxels take its own layer's test score; overall = mean over them.
    per_voxel = np.full(Yv.shape[1], np.nan)
    for g in groups:
        per_voxel[roi_m[g]] = test_corr[best[g]][roi_m[g]]
    overall = float(per_voxel[~np.isnan(per_voxel)].mean())

    region_best = {g: {"layer": best[g], "test": round(float(per_voxel[roi_m[g]].mean()), 4)}
                   for g in groups}
    roi_corr = {k: round(float(per_voxel[m].mean()), 4) for k, m in roi_m.items() if m.any()}
    print("    per-region best layer:")
    for g in groups:
        print(f"      {g:>10}: {region_best[g]['layer']:>12}  test={region_best[g]['test']:.4f}")
    print(f"    -> overall (voxel-weighted) test_corr={overall:.4f}")

    return {
        "test_corr":        round(overall, 4),
        "select_groups":    groups,
        "region_best":      region_best,
        "roi_corr":         roi_corr,
        "layer_val_scores": {k: round(float(v.mean()), 4) for k, v in val_corr.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _score_all(models, DATASET, R, device, dim, seed):
    """Load ``DATASET``'s split and run every model through best-layer-per-region scoring; return
    ``{model_name: record}`` (or ``{"error": ...}`` for a model that raised). Prints each row."""
    from harness.loaders import load_split
    t_load = time.time()
    Xtr, Xte, Ytr, Yte, roi_masks, nc = load_split(DATASET)
    if torch.is_tensor(Xtr):
        Xtr, Xte = Xtr.to(device), Xte.to(device)
    print(f"  data loaded in {time.time()-t_load:.1f}s  "
          f"n_train={Ytr.shape[0]} n_test={Yte.shape[0]} n_targets={Ytr.shape[1]}")
    records = {}
    for mname, weights in models:
        label = f"{mname}-{weights}"
        try:
            rec = score_model(mname, weights, Xtr, Xte, Ytr, Yte, roi_masks, nc,
                              DATASET, R, device, dim, seed)
            rec["weights"] = weights
            records[label] = rec
            regions = " ".join(f"{g}:{rec['region_best'][g]['layer']}"
                               f"({rec['region_best'][g]['test']:.3f})"
                               for g in rec["select_groups"])
            print(f"\n  [{label}] overall r²/NC² test={rec['test_corr']:.4f}  |  {regions}")
            rois = list(rec["roi_corr"])
            print("  " + "".join(f"{r:>9}" for r in rois))
            print("  r²/NC² " + "".join(f"{rec['roi_corr'][r]:>9.4f}" for r in rois))
        except Exception as e:
            print(f"\n  [{label}] FAILED: {e}")
            records[label] = {"error": str(e), "weights": weights}
    return records


def _mean_over_subjects(models, per_subject):
    """Average each model's per-subject records with EQUAL weight per subject (mirrors
    ``harness.reward.evaluate_subjects``): mean ``test_corr`` + mean ``roi_corr`` over the ROIs present
    for every subject. The per-subject records (incl. each subject's chosen layers) are kept under
    ``per_subject`` for inspection. A model that errored on ANY subject stays an error."""
    from subjects import common_roi_keys, mean_scalar
    out = {}
    for model, weights in models:
        m = f"{model}-{weights}"
        recs = {s: per_subject[s][m] for s in per_subject}
        errs = {s: r for s, r in recs.items() if "test_corr" not in r}
        if errs:
            out[m] = {"error": f"failed on subjects {sorted(errs)}", "per_subject": recs}
            continue
        roi_keys = common_roi_keys([r["roi_corr"] for r in recs.values()])
        out[m] = {
            "test_corr": mean_scalar(r["test_corr"] for r in recs.values()),
            "roi_corr": {k: mean_scalar(recs[s]["roi_corr"][k] for s in recs) for k in roi_keys},
            "subjects": list(per_subject),
            "per_subject": recs,
        }
    return out


def main(argv=None):
    """Run each requested vision model through best-layer-per-region scoring and print + save the
    per-model ``test_corr`` / per-ROI table (directly comparable to search-DB champions). A
    comma ``--subject`` list scores each subject separately and averages (equal weight per subject),
    matching the multi-subject search reward."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",   default="all",
                   choices=list(MODEL_REGISTRY) + ["all"],
                   help="which model to run (default: all)")
    p.add_argument("--weights", choices=("pretrained", "untrained", "both"), default="both",
                   help="score learned weights, matched random initialization, or both (default)")
    p.add_argument("--dataset", default="nsd",  help="NEURO_DATASET (default nsd)")
    p.add_argument("--subject", default=None,
                   help="NEURO_SUBJECT; a comma list (e.g. 1,3,4) is scored per-subject and averaged "
                        "(equal weight), like the search reward. Default: dataset default subject.")
    p.add_argument("--dim",     type=int, default=2048, help="SRP target dim (default 2048)")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output", help="write JSON to this exact path")
    a = p.parse_args(argv)

    os.environ["NEURO_DATASET"] = a.dataset
    if a.subject is not None:
        os.environ["NEURO_SUBJECT"] = str(a.subject)

    from subjects import subject_tokens
    from harness.datasets import DATASET, with_subject
    from harness import reward as R

    subjects = subject_tokens(a.subject if a.subject is not None else os.environ.get("NEURO_SUBJECT"))
    subj_label = ",".join(subjects) if subjects else "default"
    names = list(MODEL_REGISTRY) if a.model == "all" else [a.model]
    weights = ("pretrained", "untrained") if a.weights == "both" else (a.weights,)
    models = [(name, w) for name in names for w in weights]
    device = R._readout_device()
    print(f"=== vision_baselines | dataset={a.dataset} subj={subj_label} "
          f"reward_roi={DATASET.reward_roi} dim={a.dim} models={models} ===")

    if len(subjects) <= 1:
        # Single subject (or the dataset default): DATASET is already bound to it at import.
        all_records = _score_all(models, DATASET, R, device, a.dim, a.seed)
    else:
        # Per-subject then equal-weight mean -- the search reward's multi-subject protocol.
        per_subject = {}
        for s in subjects:
            dset = with_subject(DATASET, s)
            print(f"\n----- subject {s} ({dset.study}) -----")
            per_subject[s] = _score_all(models, dset, R, device, a.dim, a.seed)
        all_records = _mean_over_subjects(models, per_subject)

    # Summary table
    print("\n=== summary ===")
    for mname, rec in all_records.items():
        if "test_corr" not in rec:
            continue
        if "select_groups" in rec:
            regions = "  ".join(f"{g}={rec['region_best'][g]['layer']}" for g in rec["select_groups"])
            print(f"  {mname:>10}: test_corr={rec['test_corr']:.4f}   [{regions}]")
        else:                                          # multi-subject mean (layers differ per subject)
            print(f"  {mname:>10}: test_corr={rec['test_corr']:.4f}   [mean over subjects {subj_label}]")

    if not a.no_save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = a.output or os.path.join(HERE, "results",
                           f"vision_baselines_{a.model}_{a.dataset}_subj{subj_label}_{stamp}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"dataset": a.dataset, "subject": subj_label, "reward_roi": DATASET.reward_roi,
                       "srp_dim": a.dim, "seed": a.seed, "records": all_records}, f, indent=2)
        print(f"  saved -> {out}")
    return all_records


if __name__ == "__main__":
    main()
