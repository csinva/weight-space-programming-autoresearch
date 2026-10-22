"""Image benchmark presets, source adapters, split caches and provenance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import os
import tempfile

from PIL import Image
import numpy as np
import torch

from neuro_eval.benchmarks import canonical_dataset, protocol_for, SOURCE_REVISION, REGIONS, FILTERS
from neuro_eval import readers


@dataclass(frozen=True)
class Dataset:
    """Fixed image benchmark and public subject identifier."""
    study: str = "Unnamed"
    neural: str = "fmri"
    stimulus: str = "image"
    imsize: int = 128
    reward_roi: str = "all"
    metric: str = "rsa"
    stage: str = "search"
    loader: str = "laion"
    subject: str = "sub-01"

# Anatomical region coverage and thresholds are fixed in benchmarks.py.
PRESETS: dict[str, Dataset] = {
    "laion": Dataset(study="LaionFmriSub01", loader="laion", neural="fmri",
                     reward_roi="all", subject="sub-01", metric="rsa", stage="search"),
    "triplen": Dataset(study="TripleNM1", loader="triplen", neural="spikes",
                       reward_roi="all", subject="M1", metric="rsa", stage="search"),
    "nsd": Dataset(study="NsdBetasSubj01", loader="nsd", neural="fmri",
                   reward_roi="all", subject="01", metric="encoding", stage="final"),
    "tvsd": Dataset(study="TvsdMonkeyF", loader="tvsd", neural="spikes",
                    reward_roi="all", subject="F", metric="encoding", stage="final"),
}
# Accepted public spellings resolve to the same fixed benchmark.
PRESETS["laion-fmri"] = PRESETS["laion"]
PRESETS["laion_fmri"] = PRESETS["laion"]
PRESETS["triple-n"] = PRESETS["triplen"]

MIN_IMSIZE = 128


def with_subject(dataset: Dataset, subject: str) -> Dataset:
    """Bind a public subject identifier, validating it without loading neural data."""
    if dataset.loader == "laion":
        sid = subject.strip().lower()
        if not sid.startswith("sub-"):
            sid = f"sub-{int(sid):02d}"
        if sid not in protocol_for("laion").search_subjects:
            raise ValueError(f"Unknown LAION subject {subject!r}; expected sub-01,03,05,06,07")
        return replace(dataset, subject=sid, study=f"LaionFmri{sid}")
    if dataset.loader == "nsd":
        sid = int(subject)
        if f"{sid:02d}" not in protocol_for("nsd").final_subjects:
            raise ValueError("NSD subject must be 1 through 8")
        return replace(dataset, subject=f"{sid:02d}", study=f"NsdBetasSubj{sid:02d}")
    if dataset.loader == "tvsd":
        sid = subject.strip().upper()
        if sid not in protocol_for("tvsd").final_subjects:
            raise ValueError("TVSD subject must be F or N")
        return replace(dataset, subject=sid, study=f"TvsdMonkey{sid}")
    if dataset.loader == "triplen":
        sid = subject.strip().upper()
        if not sid.startswith("M"):
            sid = f"M{int(sid)}"
        if sid not in protocol_for("triplen").search_subjects:
            raise ValueError("Triple-N subject must be M1 through M5")
        return replace(dataset, subject=sid, study=f"TripleN{sid}")
    raise ValueError(f"Unknown image dataset {dataset.loader!r}")


def dataset_key(dataset: Dataset) -> str:
    """Canonical registry name for a dataset variant."""
    return canonical_dataset(dataset.loader)


def get_dataset(name: str, subject: str | None = None) -> Dataset:
    """Resolve a preset at call time, including the current input resolution override.

    Explicit subject wins; otherwise a single default binding may use the first
    NEURO_SUBJECT entry. Protocol evaluators validate the complete fixed cohort.
    """
    dataset = PRESETS[canonical_dataset(name)]
    imsize = os.environ.get("NEURO_IMSIZE")
    if imsize:
        dataset = replace(dataset, imsize=int(imsize))
    if dataset.stimulus == "image" and dataset.imsize < MIN_IMSIZE:
        raise ValueError(f"imsize={dataset.imsize} < MIN_IMSIZE={MIN_IMSIZE}; images require at least 128px")
    if subject is None:
        subjects = [s.strip() for s in os.environ.get("NEURO_SUBJECT", "").split(",") if s.strip()]
        subject = subjects[0] if subjects else None
    return with_subject(dataset, subject) if subject else dataset


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
    data_dir = readers.get_env_var(f"{name.upper()}_DATA_DIR")
    if name in ("nsd", "triplen"):
        image_path = readers.nsd_stimuli_path()
    elif name == "laion":
        image_path = readers.laion_stimuli_path()
    else:
        image_path = readers.tvsd_stimuli_path()
    filename = os.environ.get("NSD_DATA_FILE", "nsd_data.pkl") if name == "nsd" else f"{name}_data.pkl"
    paths = {"responses": str(Path(data_dir) / filename),
             "stimuli": image_path}
    if name == "nsd":
        source = readers.load_pickle(paths["responses"])
        subject = int(d.subject) - 1
        volumes = {region: readers.voxel_ncsnr_path(source["data"][readers._NSD_REGION_MAP[region]][subject], subject)
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
        subject = int(d.subject) - 1
        data = readers.load_all_nsd_data(cfg, subjects=[subject], regions=regions)
    elif name == "laion":
        subject = readers.LAION_SUBJECTS.index(d.subject)
        data = readers.load_all_laion_data(cfg, subjects=[subject], regions=regions)
    elif name == "triplen":
        subject = readers.TRIPLEN_SUBJECTS.index(d.subject)
        source = readers.load_triplen_pickle(dataset_source_paths(d)["responses"])
        regions = [r for r in regions if subject in source["data"].get(readers._TRIPLEN_REGION_MAP[r], {})]
        if not regions:
            raise ValueError(f"No Triple-N regions available for {d.subject}")
        data = readers.load_all_triplen_data(cfg, subjects=[subject], regions=regions, fold=0)
    elif name == "tvsd":
        subject = readers.TVSD_SUBJECTS.index(d.subject)
        data = readers.load_all_tvsd_data(cfg, subjects=[subject], regions=regions)
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
    load_split(d)
    return json.loads(json.dumps(_PROVENANCE[dataset_cache_key(d)]))

_SPLIT_MEMO: dict = {}
_SPLIT_ENV_KEYS = ("NEURO_NO_CACHE", "NEURO_SPLIT_CACHE")


def load_split(dataset: Dataset):
    """Load one immutable split, invalidating cache when source files change."""
    key = (dataset_cache_key(dataset), tuple(os.environ.get(k) for k in _SPLIT_ENV_KEYS))
    if key not in _SPLIT_MEMO:
        _SPLIT_MEMO[key] = load_looped(dataset)
    return _SPLIT_MEMO[key]
