"""Panel C data: the LAION search champion erjpu against interpretable baselines on NSD.

The search never used NSD, so NSD tests whether erjpu generalizes to a new dataset. Every model is
scored with the final encoding readout of the visual protocol: features centred and globally scaled
(`neuro_eval.encoding.design`, features="center"), per-voxel ridge with alpha chosen by 5-fold CV on
the training images, per-voxel Pearson r on the test images, mean over a region's voxels, then the
equal mean over subjects.

Cohort and images
  Subjects 01, 02, 05 and 07, the four who completed all 40 sessions and saw all 1,000 shared images.
  Voxels are those of the NSD final evaluation: ncsnr > 0.2 (official ncsnr volumes), early and
  ventral stream blocks pooled without reweighting. Regions: the pooled block and each stream.
  Train: the subject's own unique images. Test: the 1,000 shared images minus those in LAION-fMRI.
  LAION-fMRI shows 240 NSD images (names `NSD_special1000_<k>` and `NSD_special100_<k>`), but its
  stimulus IDs are rows of its own stimulus table, not NSD IDs. Each of the 240 is therefore matched
  to the NSD 73k image set by pixels: both images are area-downsampled to 32 x 32 RGB and compared by
  Pearson r over all 73,000 NSD images; a match must have r >= 0.99, beat the next image by >= 0.03 and
  be unique. Matched images leave the test set and, unless --keep-laion-train, the training sets too.

Features are computed once per model and subject and cached under <out stem>/features/; scores are
cached under <out stem>/scores/. A rerun resumes and rewrites the JSON from every cached score that
matches the current image split.

Panel C of the fMRI figure (figure/README.md).

Usage: python figure/c_nsd.py [--out PATH] [--only erjpu,gabor] [--device cuda] [--keep-laion-train]
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
FIGURE = Path(__file__).resolve().parent
OUT = FIGURE / "output/data/c.json"
FROZEN = FIGURE / "model.py"
SUBJECTS = ["01", "02", "05", "07"]
POOLED = "early and ventral visual streams"
REGIONS = [POOLED, "early visual stream", "ventral visual stream"]
# AlexNet layer fixed before any NSD score: best LAION pooled visual encoding score
# (artifacts/final/baselines/laion__visual__encoding.md, conv4 0.503; the same layer is listed for the
# pooled ROI in artifacts/final/alexnet_comparison_visual_encoding_2026-11-05/alexnet_roi_layer_selection_laion.json).
ALEXNET = "alexnet_conv4"
MODELS = {  # name: (label, kind)
    "erjpu": ("Searched model", "searched"),
    "pixels": ("Pixels", "baseline"),
    "gabor": ("Gabor", "baseline"),
    "gabor_norm": ("Gabor + normalization", "baseline"),
    "gist": ("GIST", "baseline"),
    "hmax": ("HMAX", "baseline"),
    "scattering": ("Scattering", "baseline"),
    "texture": ("Texture statistics", "baseline"),
    "curvature": ("Curvature", "baseline"),
    ALEXNET: ("AlexNet conv4", "reference"),
}
MATCH_SIZE, MATCH_MIN_R, MATCH_MIN_MARGIN = 32, 0.99, 0.03


def load_model(path):
    """Import a frozen file for standalone comparison after the static source scan.

    This executes code in-process; search and frozen final evaluation continue
    to use IsolatedModel.
    """
    from neuro_eval.model import scan_source
    import importlib.util
    path = Path(path).resolve()
    if reason := scan_source(str(path)):
        raise ValueError(f"{path} rejected: {reason}")
    spec = importlib.util.spec_from_file_location(f"model_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL


def sha256_text(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_npy(path, array):
    with open(path, "wb") as file:
        np.save(file, array, allow_pickle=False)


def thumbnails(images, device):
    """(n, H, W, 3) uint8 -> (n, 3 * 32 * 32) area-downsampled RGB in [0, 1]."""
    x = torch.as_tensor(np.array(images)).to(device).permute(0, 3, 1, 2).float() / 255
    return torch.nn.functional.interpolate(x, size=(MATCH_SIZE, MATCH_SIZE), mode="area").flatten(1)


def standardize(x):
    x = x - x.mean(1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True)


def laion_nsd_matches(cache, device):
    """{LAION stimulus id: NSD 73k id (0-based)} for every NSD-source LAION image, by pixel matching."""
    import h5py
    from neuro_eval import readers
    laion_pkl = Path(readers.get_env_var("LAION_DATA_DIR")) / "laion_data.pkl"
    nsd_h5 = readers.nsd_stimuli_path()
    laion_h5 = readers.laion_stimuli_path()
    sources = {"laion_responses": str(laion_pkl), "laion_stimuli": laion_h5, "nsd_stimuli": nsd_h5}
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["sources"] == sources:
            return data
    table = readers.load_pickle(laion_pkl)["stimuli"]
    ids = [str(i) for i in np.flatnonzero(table["dataset"].values == "NSD")]
    names = table["image_name"].values[[int(i) for i in ids]].tolist()
    laion = readers.LazyEncodedHdf5Dict(laion_h5, ids)  # the evaluator's own decoding
    query = standardize(torch.cat([thumbnails(laion[i][None], device) for i in ids]))
    t0 = time.time()
    with h5py.File(nsd_h5, "r") as file:
        brick = file["imgBrick"]
        scores = torch.cat([query @ standardize(thumbnails(brick[k:k + 1000], device)).T
                            for k in range(0, len(brick), 1000)], 1)
    top = scores.topk(2, dim=1)
    best, r1, r2 = top.indices[:, 0].tolist(), top.values[:, 0].tolist(), top.values[:, 1].tolist()
    print(f"matched {len(ids)} LAION NSD-source images against {scores.shape[1]} NSD images "
          f"({time.time() - t0:.0f}s): min r {min(r1):.4f}, max runner-up {max(r2):.4f}", flush=True)
    bad = [(n, a, b) for n, a, b in zip(names, r1, r2) if a < MATCH_MIN_R or a - b < MATCH_MIN_MARGIN]
    if bad or len(set(best)) != len(best):
        raise RuntimeError(f"Ambiguous LAION-to-NSD pixel matches: {bad[:5]} (unique {len(set(best))}/{len(best)})")
    # Name check: special1000_<k> images are NSD image k (special100 numbering is not an NSD id).
    named = [(int(n.rsplit("_", 1)[1].split(".")[0]), b) for n, b in zip(names, best) if "special1000_" in n]
    data = {"sources": sources, "method": f"area-downsampled {MATCH_SIZE}x{MATCH_SIZE} RGB, Pearson r over all "
            f"NSD images; accept r >= {MATCH_MIN_R}, margin >= {MATCH_MIN_MARGIN}, unique",
            "matches": {i: {"name": n, "nsd_id": int(b), "r": a, "runner_up_r": c}
                        for i, n, b, a, c in zip(ids, names, best, r1, r2)},
            "min_r": min(r1), "max_runner_up_r": max(r2),
            "special1000_name_agrees": f"{sum(k == b for k, b in named)}/{len(named)}"}
    atomic_write(cache, lambda p: p.write_text(json.dumps(data, indent=1)))
    return data


def subject_split(d, shared, laion_ids, keep_laion_train):
    """Responses, region masks and image IDs of one subject, using the final NSD loaders and filters."""
    from neuro_eval.datasets import _load_native, _pack_responses, _union_masks
    native, subject = _load_native(d)  # final stage: early + ventral blocks, ncsnr > 0.2
    first = native["neural"][native["regions"][0]][subject]
    train, test = set(first["train"]), set(first["test"])
    for region in native["regions"]:
        split = native["neural"][region][subject]
        if set(split["train"]) != train or set(split["test"]) != test:
            raise ValueError(f"Stimulus IDs differ across regions for {region}")
    if test != shared or train & shared:
        raise ValueError(f"Subject {d.subject} did not see exactly the 1,000 shared images")
    n_laion_train = len(train & laion_ids)
    if not keep_laion_train:
        train -= laion_ids
    train_ids, test_ids = sorted(train, key=int), sorted(test - laion_ids, key=int)
    Ytr, Yte, masks = _pack_responses(native, subject, train_ids, test_ids)
    masks = _union_masks(d, native, subject, masks)
    if list(masks) != REGIONS:
        raise ValueError(f"Unexpected regions {list(masks)}")
    info = {"n_train": len(train_ids), "n_laion_in_unique": n_laion_train,
            "n_units": {r: int(m.sum()) for r, m in masks.items()},
            "voxel_counts": {r: native["voxel_counts"][r][subject] for r in native["regions"]}}
    return native, train_ids, test_ids, Ytr, Yte, masks, info


@contextmanager
def _model_environment(name, trusted):
    """Allow fixed baselines to exceed the candidate parameter cap."""
    previous = os.environ.get("NEURO_MAX_PARAMS")
    try:
        if name in trusted:
            os.environ["NEURO_MAX_PARAMS"] = str(10**9)
        yield
    finally:
        if previous is None:
            os.environ.pop("NEURO_MAX_PARAMS", None)
        else:
            os.environ["NEURO_MAX_PARAMS"] = previous


def features(name, model, X, ids_hash, path, device, trusted):
    """Cached (n_images, D) float32 features in train-then-test order."""
    from neuro_eval._evaluation import encoding_forward
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text())["ids_sha256"] == ids_hash:
        return torch.from_numpy(np.load(path))
    t0 = time.time()
    with _model_environment(name, trusted):
        A = encoding_forward(model, X(), torch.device(device), f"{name} features").cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda p: save_npy(p, A.numpy()))
    meta_path.write_text(json.dumps({"ids_sha256": ids_hash, "shape": list(A.shape),
                                     "seconds": round(time.time() - t0, 1)}))
    print(f"  {name}: features {tuple(A.shape)} in {time.time() - t0:.0f}s", flush=True)
    return A


def code_revision():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=REPO, capture_output=True, text=True).stdout
        return head + ("+dirty" if dirty.strip() else "")
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", help=f"comma-separated subset of {','.join(MODELS)}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-laion-train", action="store_true",
                        help="keep LAION-matched NSD images in the training sets (always removed from test)")
    args = parser.parse_args()
    names = args.only.split(",") if args.only else list(MODELS)
    if unknown := set(names) - set(MODELS):
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    out, work = args.out.resolve(), args.out.resolve().with_suffix("")
    work.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO))
    os.environ["NEURO_NO_CACHE"] = "1"  # no hidden multi-GB pixel cache; features are cached here instead
    os.environ.pop("NEURO_IMSIZE", None)
    from neuro_eval import readers
    from neuro_eval.benchmarks import FILTERS, PROTOCOL_VERSION
    from neuro_eval.datasets import get_dataset, with_subject, dataset_source_paths, _pixels, _file_identity
    from neuro_eval.encoding import encoding_scores, ENCODING_FEATURES, N_FOLDS, ALPHAS

    if sha256_file(FROZEN) != "cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633":
        raise ValueError(f"{FROZEN} does not match its frozen sha256")
    models = {}
    if "erjpu" in names:
        models["erjpu"] = load_model(FROZEN)
    from baselines import load
    if baseline_names := [n for n in names if n != "erjpu"]:
        models.update(load(baseline_names))
    trusted = set(models) - {"erjpu"}

    match = laion_nsd_matches(work / "laion_nsd_matches.json", args.device)
    laion_ids = {str(m["nsd_id"]) for m in match["matches"].values()}
    shared = {str(i) for i in readers.load_nsd_pickle(os.environ.get("NSD_DATA_FILE", "nsd_data.pkl"))["shared_ids"]}
    base = get_dataset("nsd", region="visual", search_metric="encoding")  # final NSD dataset, visual protocol
    assert base.reward_roi == POOLED and base.imsize == 128 and base.stage == "final"

    subjects, test_reference = {}, None
    for s in SUBJECTS:
        d = with_subject(base, s)
        t0 = time.time()
        native, train_ids, test_ids, Ytr, Yte, masks, info = subject_split(d, shared, laion_ids, args.keep_laion_train)
        if test_reference is None:
            test_reference = test_ids
        elif test_ids != test_reference:
            raise ValueError("Test images differ across subjects")
        ids = train_ids + test_ids
        ids_hash = sha256_text([train_ids, test_ids])
        print(f"subject {s}: {len(train_ids)} train, {len(test_ids)} test images, units {info['n_units']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        pixels = {}

        def X():
            if "X" not in pixels:
                pixels["X"] = _pixels(native["stimuli"], ids, d.imsize)
            return pixels["X"]

        for name in names:
            score_path = work / "scores" / f"{name}__{s}.json"
            if score_path.exists() and json.loads(score_path.read_text())["ids_sha256"] == ids_hash:
                continue
            A = features(name, models[name], X, ids_hash, work / "features" / name / f"{s}.npy", args.device, trusted)
            t1 = time.time()
            scored = encoding_scores(A[:len(train_ids)], A[len(train_ids):], Ytr, Yte, masks, torch.device(args.device))
            row = {"ids_sha256": ids_hash, "subject": s, "n_features": int(A.shape[1]),
                   "scores": {r: scored["roi_corr"][r]["test"] for r in REGIONS},
                   "train_scores": {r: scored["roi_corr"][r]["train"] for r in REGIONS},
                   "alpha_range": [scored["encoding_alpha_min"], scored["encoding_alpha_max"]],
                   "seconds": round(time.time() - t1, 1)}
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps(row, indent=1))
            print(f"  {name}: pooled r={row['scores'][POOLED]:.4f} ({row['seconds']:.0f}s)", flush=True)
            del A
            torch.cuda.empty_cache()
        subjects[s] = {**info, "ids_sha256": ids_hash,
                       "files": {k: _file_identity(p) for k, p in dataset_source_paths(d).items()}}
        del native, Ytr, Yte, pixels

    # Panel data from every cached score of the current split (models from earlier --only runs included).
    result = {}
    for name, (label, kind) in MODELS.items():
        rows = {}
        for s in SUBJECTS:
            p = work / "scores" / f"{name}__{s}.json"
            if p.exists() and (row := json.loads(p.read_text()))["ids_sha256"] == subjects[s]["ids_sha256"]:
                rows[s] = row
        if len(rows) == len(SUBJECTS):
            result[name] = {"label": label, "kind": kind, "scores": {s: rows[s]["scores"] for s in SUBJECTS},
                            "mean": {r: float(np.mean([rows[s]["scores"][r] for s in SUBJECTS])) for r in REGIONS}}
    descriptions = {n: getattr(m, "description", "") for n, m in load([n for n in MODELS if n != "erjpu"]).items()}
    overlap_shared = sorted(m["nsd_id"] for m in match["matches"].values() if str(m["nsd_id"]) in shared)
    data = {
        "subjects": SUBJECTS, "n_test_images": len(test_reference),
        "test_image_ids_sha256": sha256_text(test_reference),
        "regions": REGIONS, "models": result,
        "provenance": {
            "script": "figure/c_nsd.py", "code_revision": code_revision(),
            "dataset": "nsd", "subjects": SUBJECTS,
            "subject_rule": "NSD subjects who completed all 40 sessions and saw all 1,000 shared images",
            "units": f"final NSD visual protocol: ncsnr > {FILTERS['nsd']['ncsnr_threshold']} (official ncsnr volumes), "
                     "early and ventral stream blocks pooled without reweighting",
            "filters": FILTERS["nsd"], "neuro_eval_protocol_version": PROTOCOL_VERSION,
            "readout": {"design": f"neuro_eval.encoding.design(features={ENCODING_FEATURES!r})",
                        "ridge": f"per-voxel alpha by {N_FOLDS}-fold CV on training images, "
                                 f"{len(ALPHAS)} alphas logspace(-10, 10)",
                        "score": "per-voxel Pearson r on test images, region mean, then equal subject mean",
                        "function": "neuro_eval.encoding.encoding_scores"},
            "images": {"train": "each subject's own unique images"
                                + ("" if args.keep_laion_train else ", minus images in LAION-fMRI"),
                       "test": "the 1,000 NSD shared images minus images in LAION-fMRI",
                       "pixels": "rgb-pil-bilinear-hwc-v1, 128 x 128", "test_image_ids": test_reference},
            "laion_exclusion": {"method": match["method"], "n_laion_nsd_source_images": len(match["matches"]),
                                "min_match_r": match["min_r"], "max_runner_up_r": match["max_runner_up_r"],
                                "special1000_name_agrees": match["special1000_name_agrees"],
                                "shared_images_in_laion": overlap_shared,
                                "match_file": str(work / "laion_nsd_matches.json")},
            "per_subject": subjects,
            "models": {
                "erjpu": {"path": str(FROZEN.relative_to(REPO)), "sha256": sha256_file(FROZEN),
                          "search_run": "artifacts/search/laion__visual__encoding__2026-11-05"},
                **{n: {"module": f"baselines ({n})", "settings": "defaults of baselines/",
                       "description": descriptions.get(n, "")} for n in MODELS if n != "erjpu"},
                ALEXNET: {"module": "baselines.alexnet", "weights": "ImageNet alexnet-owt-7be5be79.pth",
                          "description": descriptions.get(ALEXNET, ""),
                          "layer_choice": "best LAION pooled visual encoding score (search split): "
                                          "artifacts/final/baselines/laion__visual__encoding.md (conv4 0.503), "
                                          "agrees with alexnet_comparison_visual_encoding_2026-11-05/"
                                          "alexnet_roi_layer_selection_laion.json; chosen before any NSD score"},
            },
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda p: p.write_text(json.dumps(data, indent=2)))
    print(f"wrote {out} ({len(result)}/{len(MODELS)} models complete)", flush=True)


if __name__ == "__main__":
    main()
