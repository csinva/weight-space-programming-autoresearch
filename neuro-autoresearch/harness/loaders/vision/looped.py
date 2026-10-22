"""Adapt vendored looped-networks responses to the fixed harness tensor contract.

Search datasets return an empty training split and all protocol-selected images
in the test split. RSA is evaluated on that complete fixed image set. Final
datasets retain the source train/test split for encoding. The two primary
streams are scored separately and receive equal weight.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from harness.datasets import Dataset, dataset_key, with_subject
from harness.protocols import SOURCE_REVISION, protocol_for
from harness.vendor.looped import utils


REGIONS = {name: ("early visual stream", "ventral visual stream")
           for name in ("nsd", "laion", "triplen", "tvsd")}
FILTERS = {
    "nsd": {"ncsnr_threshold": 0.2},
    "laion": {"ncsnr_threshold": 0.2},
    "triplen": {"reliability_threshold": 0.4},
    "tvsd": {"oracle_threshold": 0.4},
}

_PROVENANCE: dict[tuple, dict] = {}
_PIXEL_MEMO: dict[str, torch.Tensor] = {}


class VisionSplit(tuple):
    """Six-field compatibility tuple with evaluator-only stimulus group metadata."""

    def __new__(cls, values, *, source_groups=None):
        result = super().__new__(cls, values)
        result.source_groups = source_groups or {}
        return result


def _source_groups(d: Dataset, native: dict, ids: list[str]) -> dict[str, list[int]]:
    # An explicit allowlist prevents NSD and the overlapping no_ood grouping
    # from entering feedback. Intersect with the already filtered fixed cohort.
    if dataset_key(d) != "laion":
        return {}
    return {name: [i for i, sid in enumerate(ids) if sid in members]
            for name in ("laion", "thingsplus", "ood")
            for members in [set(native["test_subsets"].get(name, []))]}


def _file_identity(path) -> dict:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def dataset_source_paths(d: Dataset) -> dict[str, str]:
    """Storage locations, without importing or loading a sibling checkout."""
    name = dataset_key(d)
    # Load this repository's .env before resolving any optional stimulus path.
    data_dir = utils.get_env_var(f"{name.upper()}_DATA_DIR")
    if name in ("nsd", "triplen"):
        from harness.vendor.looped.data.nsd import nsd_stimuli_path
        image_path = nsd_stimuli_path()
    elif name == "laion":
        from harness.vendor.looped.data.laion import laion_stimuli_path
        image_path = laion_stimuli_path()
    else:
        from harness.vendor.looped.data.tvsd import tvsd_stimuli_path
        image_path = tvsd_stimuli_path()
    filename = os.environ.get("NSD_DATA_FILE", "nsd_data.pkl") if name == "nsd" else f"{name}_data.pkl"
    paths = {"responses": str(Path(data_dir) / filename),
             "stimuli": image_path}
    if name == "nsd":
        from harness.vendor.looped.data.nsd import _NSD_REGION_MAP, voxel_ncsnr_path
        source = utils.load_pickle(paths["responses"])
        subject = int(d.subject) - 1
        volumes = {region: voxel_ncsnr_path(source["data"][_NSD_REGION_MAP[region]][subject], subject)
                   for region in REGIONS[name]}
        if len(set(volumes.values())) == 1:
            paths["ncsnr"] = next(iter(volumes.values()))
        else:
            paths.update({f"ncsnr:{region}": path for region, path in volumes.items()})
    return paths


def dataset_cache_key(d: Dataset) -> tuple:
    """The scientific input file identities underlying one cached tensor split."""
    files = tuple((key, tuple(_file_identity(path).values()))
                  for key, path in dataset_source_paths(d).items())
    return d, files


def _load_native(d: Dataset):
    name = dataset_key(d)
    d = with_subject(d, d.subject)
    regions = list(REGIONS[name])
    cfg = dict(FILTERS[name])
    if name == "nsd":
        from harness.vendor.looped.data.nsd import load_all_nsd_data
        subject = int(d.subject) - 1
        data = load_all_nsd_data(cfg, subjects=[subject], regions=regions)
    elif name == "laion":
        from harness.vendor.looped.data.laion import SUBJECTS, load_all_laion_data
        subject = SUBJECTS.index(d.subject)
        data = load_all_laion_data(cfg, subjects=[subject], regions=regions)
    elif name == "triplen":
        from harness.vendor.looped.data.triplen import (
            SUBJECTS, _TRIPLEN_REGION_MAP, load_all_triplen_data, load_triplen_pickle,
        )
        subject = SUBJECTS.index(d.subject)
        source = load_triplen_pickle(dataset_source_paths(d)["responses"])
        regions = [r for r in regions if subject in source["data"].get(_TRIPLEN_REGION_MAP[r], {})]
        if not regions:
            raise ValueError(f"No Triple-N regions available for {d.subject}")
        data = load_all_triplen_data(cfg, subjects=[subject], regions=regions, fold=0)
    elif name == "tvsd":
        from harness.vendor.looped.data.tvsd import SUBJECTS, load_all_tvsd_data
        subject = SUBJECTS.index(d.subject)
        data = load_all_tvsd_data(cfg, subjects=[subject], regions=regions)
    else:
        raise ValueError(f"Not a looped vision dataset: {name}")
    return data, subject


def _selected_ids(d: Dataset, native: dict, subject: int):
    """Sort IDs numerically and enforce identical image alignment across regions."""
    name = dataset_key(d)
    first = native["neural"][native["regions"][0]][subject]
    train_ids, test_ids = set(first["train"]), set(first["test"])
    if train_ids & test_ids:
        raise ValueError("Source train/test stimulus IDs overlap")
    for region in native["regions"]:
        split = native["neural"][region][subject]
        if set(split["train"]) != train_ids or set(split["test"]) != test_ids:
            raise ValueError(f"Stimulus IDs differ across regions for {region}")
    if name == "laion":
        shared = set(native["shared_test_ids"])
        # Exclude source-labelled NSD images before exposing pixels or responses
        # to search. The remaining non-NSD source images are the fixed reward set.
        test_ids = shared - set(native["test_subsets"]["nsd"])
        train_ids = set()
    elif name == "triplen":
        test_ids |= train_ids
        train_ids = set()
    if not test_ids:
        raise ValueError(f"No evaluation stimuli remain for {name}/{d.subject}")
    return sorted(train_ids, key=int), sorted(test_ids, key=int)


def _pixel_cache_dir() -> Path:
    return Path(os.environ.get("NEURO_SPLIT_CACHE", Path.home() / ".cache/neuro-autoresearch/splits"))


def _pixels(stimuli, ids: list[str], imsize: int) -> torch.Tensor:
    """HWC flattened [0,1] RGB, keyed by exact IDs and stimulus-file identity."""
    identity = _file_identity(stimuli._hdf5_path)
    key = hashlib.sha256(json.dumps({"source": identity, "ids": ids, "imsize": imsize,
                                    "pipeline": "rgb-pil-bilinear-hwc-v1"}, sort_keys=True).encode()).hexdigest()
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    if use_cache and key in _PIXEL_MEMO:
        return _PIXEL_MEMO[key]
    cache_path = _pixel_cache_dir() / f"looped_pixels_{key}.npy"
    if use_cache and cache_path.exists():
        # Copy-on-write mapping lets model input cloning remain inexpensive and
        # prevents a tensor mutation from changing the persisted stimulus cache.
        pixels = np.load(cache_path, mmap_mode="c", allow_pickle=False)
        if pixels.shape != (len(ids), imsize * imsize * 3) or pixels.dtype != np.float32:
            raise ValueError(f"Invalid stimulus cache {cache_path}; remove it and retry")
    else:
        pixels = np.empty((len(ids), imsize * imsize * 3), dtype=np.float32)
        for row, sid in enumerate(ids):
            im = Image.fromarray(np.asarray(stimuli[sid], dtype=np.uint8)).convert("RGB")
            im = im.resize((imsize, imsize), resample=Image.Resampling.BILINEAR)
            pixels[row] = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="looped_pixels_", suffix=".npy", dir=cache_path.parent)
            try:
                with os.fdopen(fd, "wb") as file:
                    np.save(file, pixels, allow_pickle=False)
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    result = torch.from_numpy(pixels)
    if use_cache:
        _PIXEL_MEMO[key] = result
    return result


def _pack_responses(native: dict, subject: int, train_ids: list[str], test_ids: list[str]):
    blocks, counts = [], {}
    ordered_ids = train_ids + test_ids
    for region in native["regions"]:
        source = native["neural"][region][subject]
        merged = {**source["train"], **source["test"]}
        values = np.asarray([merged[sid] for sid in ordered_ids], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid or non-finite responses in region {region}")
        blocks.append(values)
        counts[region] = values.shape[1]
    values = torch.from_numpy(np.concatenate(blocks, axis=1))
    masks, offset = {}, 0
    for region, count in counts.items():
        mask = np.zeros(values.shape[1], dtype=bool)
        mask[offset:offset + count] = True
        masks[region] = mask
        offset += count
    return values[:len(train_ids)], values[len(train_ids):], masks


def load_looped(d: Dataset):
    """Return Xtrain, Xtest, Ytrain, Ytest, region masks, None.

    Noise ceilings are deliberately not returned: source alignment uses raw RSA
    and raw encoding Pearson r, with the source reliability filters applied.
    """
    native, subject = _load_native(d)
    train_ids, test_ids = _selected_ids(d, native, subject)
    X = _pixels(native["stimuli"], train_ids + test_ids, d.imsize)
    Ytr, Yte, masks = _pack_responses(native, subject, train_ids, test_ids)
    groups = _source_groups(d, native, train_ids + test_ids)
    protocol = protocol_for(dataset_key(d))
    _PROVENANCE[dataset_cache_key(d)] = {
        "source_revision": SOURCE_REVISION, "dataset": dataset_key(d), "subject": d.subject,
        "protocol": protocol.to_dict(), "metric": d.metric, "stage": d.stage,
        "files": {key: _file_identity(path) for key, path in dataset_source_paths(d).items()},
        "filters": FILTERS[dataset_key(d)].copy(), "regions": list(masks),
        "region_units": {name: int(mask.sum()) for name, mask in masks.items()},
        "train_ids": train_ids, "test_ids": test_ids,
        "source_group_counts": {name: len(indices) for name, indices in groups.items()},
        "stimulus_ids_sha256": hashlib.sha256(json.dumps([train_ids, test_ids]).encode()).hexdigest(),
        "pixel_pipeline": "rgb-pil-bilinear-hwc-v1", "imsize": d.imsize,
    }
    return VisionSplit((X[:len(train_ids)], X[len(train_ids):], Ytr, Yte, masks, None),
                       source_groups=groups)


def dataset_provenance(d: Dataset) -> dict:
    """Exact stimulus IDs, files and filtering of a loaded split (loads on first call)."""
    from harness.loaders import load_split
    load_split(d)
    return json.loads(json.dumps(_PROVENANCE[dataset_cache_key(d)]))
