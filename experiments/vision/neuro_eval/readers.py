"""Prepared response and lazy stimulus readers from looped-networks.

Source revision: 3ee575baeceb1ad784102162a051b7444182fdaa (imported 2026-10-21).
Consolidates data/{stimuli,nsd,laion,triplen,tvsd}.py and the local path/pickle
helpers. Source filters, response ordering, splits and image decoding are
preserved. Dataset-specific names distinguish formerly module-local helpers;
unused NSD stream/ROI convenience exports are omitted. Environment loading is
anchored to this repository, including after relocation from reference/.
The experiment's cohort and image-policy adaptations live in datasets.py.
"""

import io
import os
import pickle
from functools import lru_cache
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from PIL import Image


# Repository paths and source-identity pickle cache

DEFAULT_DATA_DIRS = {
    "NSD_DATA_DIR": "/data/shared/datasets/nsd_processed",
    "LAION_DATA_DIR": "/data/shared/datasets/laion_fmri/processed",
    "TRIPLEN_DATA_DIR": "/data/shared/datasets/triplen/processed",
    "TVSD_DATA_DIR": "/data/shared/datasets/papale2025.tvsd/processed",
}


def get_env_var(key: str) -> str:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    value = os.environ.get(key, DEFAULT_DATA_DIRS.get(key))
    if not value:
        raise RuntimeError(f"Missing required env var: {key} (check this repository's .env)")
    return value


def load_pickle(file_path):
    path = Path(file_path).expanduser().resolve()
    stat = path.stat()
    return _load_pickle(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _load_pickle(path, size_bytes, mtime_ns):
    """Cache immutable source objects, invalidating when a prepared file changes."""
    with open(path, "rb") as file:
        return pickle.load(file)


# Lazy HDF5 stimulus decoding

class LazyHdf5Dict:
    """Read-only dict {str(stimulus id): image} that reads each image from the HDF5 file on demand."""

    def __init__(self, hdf5_path: str, dataset_name: str, indices):
        self._hdf5_path = hdf5_path
        self._dataset_name = dataset_name
        self._index_map = {str(idx): int(idx) for idx in indices}
        self._keys_sorted = sorted(self._index_map.keys(), key=lambda x: int(x))
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        return self._file

    def __contains__(self, key):
        return str(key) in self._index_map

    def __len__(self):
        return len(self._index_map)

    def keys(self):
        return self._keys_sorted

    def __getitem__(self, key):
        key_str = str(key)
        if key_str not in self._index_map:
            raise KeyError(key)
        return self._open()[self._dataset_name][self._index_map[key_str]]

    def __del__(self):
        if self._file is not None:
            self._file.close()


class LazyEncodedHdf5Dict:
    """Read-only dict {str(stimulus id): uint8 RGB image} decoding each encoded image of the HDF5 file on demand."""

    def __init__(self, hdf5_path, indices):
        self._hdf5_path = hdf5_path
        self._keys_sorted = sorted((str(int(i)) for i in indices), key=int)
        self._keys = set(self._keys_sorted)
        self._file = None

    def __contains__(self, key):
        return str(key) in self._keys

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys_sorted

    def __getitem__(self, key):
        if str(key) not in self._keys:
            raise KeyError(key)
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        im = Image.open(io.BytesIO(bytes(self._file["images"][int(key)])))
        if im.mode in ("RGBA", "LA", "P"):  # 131 OOD PNGs: shown alpha-composited over PsychoPy's mid-grey
            im = Image.alpha_composite(Image.new("RGBA", im.size, (128, 128, 128, 255)), im.convert("RGBA"))
        return np.asarray(im.convert("RGB"))

    def __getstate__(self):  # never hand an open h5 handle to a DataLoader worker
        return {**self.__dict__, "_file": None}

    def __del__(self):
        if self._file is not None:
            self._file.close()


# NSD responses and coordinate-based reliability

# Full region name -> key in the NSD pickle.
_NSD_REGION_MAP = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "hV4": "hV4",
    "FFA": "FFA",
    "PPA": "PPA",
    # Only in the extended pickle (NSD_DATA_FILE=nsd_data_ext.pkl): Kastner2015 atlas ROIs continuing the
    # ventral (VO, PHC) and lateral (LO, TO) chains, plus the remaining NSD "streams" parcels.
    **{r: r for r in ("VO1", "VO2", "PHC1", "PHC2", "LO1", "LO2", "TO1", "TO2", "V3A", "V3B",
                      "midventral", "midlateral", "lateral")},
}



_NSD_SUBJECTS = list(range(8))


def nsd_root() -> str:
    """NSD release root: $NSD_ROOT (env or .env), else the gpu2 shared copy."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return os.environ.get("NSD_ROOT", "/data/shared/datasets/allen2021.natural_scenes")


def nsd_stimuli_path() -> str:
    return os.environ.get("NSD_STIMULI_HDF5_PATH", f"{nsd_root()}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")


def nsd_betas_dir() -> str:
    return os.environ.get("NSD_BETAS_DIR", f"{nsd_root()}/nsddata_betas")


def load_nsd_pickle(name):
    """A pickle from NSD_DATA_DIR, e.g. 'nsd_data.pkl' or 'nsd_data_ext.pkl'."""
    return load_pickle(os.path.join(get_env_var("NSD_DATA_DIR"), name))


def voxel_ncsnr_path(fmri_xr, subject: int) -> str:
    """Official reliability volume selected by response metadata (zero-based subject)."""
    resolution = fmri_xr.attrs.get("resolution", "1pt8mm")
    preprocessing = fmri_xr.attrs.get("preprocessing", "fithrf_GLMdenoise_RR")
    return os.path.join(
        nsd_betas_dir(), "ppdata", f"subj{subject + 1:02d}",
        f"func{resolution}", f"betas_{preprocessing}", "ncsnr.nii.gz",
    )


def load_voxel_ncsnr(fmri_xr, subject: int) -> np.ndarray:
    """Official NSD ncsnr of each voxel of an fMRI DataArray, read at its x/y/z coordinates."""
    missing_coordinates = [axis for axis in ("x", "y", "z") if axis not in fmri_xr.coords]
    if missing_coordinates:
        raise ValueError(
            "Cannot apply the NSD ncsnr filter because the response data lack "
            f"voxel coordinate(s) {missing_coordinates}. Rebuild nsd_data.pkl with "
            "voxel coordinates, or explicitly set ncsnr_threshold=null for a legacy run."
        )
    path = voxel_ncsnr_path(fmri_xr, subject)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NSD ncsnr volume not found: {path}. Set NSD_BETAS_DIR to the "
            "directory containing the official NSD beta derivatives."
        )

    volume = np.asanyarray(nib.load(path).dataobj)
    coordinates = np.column_stack([
        np.asarray(fmri_xr.coords[axis].values, dtype=np.int64)
        for axis in ("x", "y", "z")
    ])
    if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(volume.shape[:3])):
        raise ValueError(f"Voxel coordinates fall outside ncsnr volume {path} with shape {volume.shape}")

    values = np.asarray(volume[tuple(coordinates.T)], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite ncsnr values found for subject {subject}")
    return values


def load_all_nsd_data(cfg, subjects=None, regions=None) -> dict:
    """NSD fMRI responses of the requested subjects and regions.

    Voxels are kept when their ncsnr is strictly greater than cfg['ncsnr_threshold'] (default 0.2;
    None keeps every voxel), in their original order. NSD_DATA_FILE picks the pickle in NSD_DATA_DIR
    (default nsd_data.pkl; nsd_data_ext.pkl adds the extended ROIs).

    Args:
        subjects: subject indices (default: all 8).
        regions: full region names (default: every region in _NSD_REGION_MAP).

    Returns:
        dict with keys:
            - "regions", "subjects": what was loaded
            - "neural": {region: {subj: {"train": {sid: resp}, "test": {sid: resp}}}}
            - "stimuli": lazy {str(stim_id): uint8 image} over every loaded stimulus
            - "shared_test_ids": test ids shared by all subjects, sorted by int
            - "ncsnr_threshold"
            - "voxel_counts": {region: {subj: {"total", "retained", "excluded"}}}
            - "ncsnr": {region: {subj: ncsnr of the retained voxels, or None}}
            - "units": {region: {subj: (n_voxels, 3) x, y, z of the retained voxels (func1pt8mm, 0-based)}}
    """
    subjects = subjects if subjects is not None else _NSD_SUBJECTS
    region_pairs = [(pkl_key, name) for name, pkl_key in _NSD_REGION_MAP.items()
                    if regions is None or name in regions]

    nsd = load_nsd_pickle(os.environ.get("NSD_DATA_FILE", "nsd_data.pkl"))
    missing = [name for _, name in region_pairs if _NSD_REGION_MAP[name] not in nsd["data"]]
    if missing:
        raise KeyError(f"Regions {missing} not in this NSD pickle (set NSD_DATA_FILE=nsd_data_ext.pkl?)")
    shared_ids = nsd["shared_ids"]
    ncsnr_threshold = cfg.get("ncsnr_threshold", 0.2)

    neural, voxel_counts, voxel_ncsnr, units = {}, {}, {}, {}
    all_stimulus_ids = set()
    per_subject_test_ids = []

    for region_key, region_full in region_pairs:
        neural[region_full], voxel_counts[region_full], voxel_ncsnr[region_full], units[region_full] = {}, {}, {}, {}
        for subj in subjects:
            fmri_xr = nsd["data"][region_key][subj]
            total_voxels = int(fmri_xr.sizes["neuroid"])
            ncsnr = None
            if ncsnr_threshold is not None:
                ncsnr = load_voxel_ncsnr(fmri_xr, subj)
                keep = ncsnr > float(ncsnr_threshold)  # strictly greater, the published NSD rule
                fmri_xr = fmri_xr.isel(neuroid=np.flatnonzero(keep))
                ncsnr = ncsnr[keep]
            elif "ncsnr" in fmri_xr.coords:
                ncsnr = np.asarray(fmri_xr.coords["ncsnr"].values, dtype=np.float32)
            voxel_ncsnr[region_full][subj] = ncsnr
            units[region_full][subj] = np.stack([fmri_xr[a].values for a in "xyz"], 1).astype(np.int64)
            retained_voxels = int(fmri_xr.sizes["neuroid"])
            if retained_voxels == 0:
                raise ValueError(
                    f"ncsnr_threshold={ncsnr_threshold} removed every voxel for "
                    f"subject {subj}, region {region_full}"
                )
            voxel_counts[region_full][subj] = {
                "total": total_voxels,
                "retained": retained_voxels,
                "excluded": total_voxels - retained_voxels,
            }
            stimulus_ids = [int(i) for i in fmri_xr.coords["stimulus"].values]
            all_stimulus_ids.update(stimulus_ids)

            train_ids = [str(i) for i in stimulus_ids if i not in shared_ids]
            test_ids = [str(i) for i in stimulus_ids if i in shared_ids]
            neural[region_full][subj] = {
                "train": {i: fmri_xr.sel(stimulus=int(i)).values for i in train_ids},
                "test": {i: fmri_xr.sel(stimulus=int(i)).values for i in test_ids},
            }
            if region_key == region_pairs[0][0]:
                per_subject_test_ids.append(set(test_ids))

    return {
        "regions": [name for _, name in region_pairs],
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": LazyHdf5Dict(nsd_stimuli_path(), "imgBrick", all_stimulus_ids),
        "shared_test_ids": sorted(set.intersection(*per_subject_test_ids), key=int),
        "ncsnr_threshold": ncsnr_threshold,
        "voxel_counts": voxel_counts,
        "ncsnr": voxel_ncsnr,
        "units": units,
    }


# LAION-fMRI responses

LAION_SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
LAION_DEFAULT_RAW_DIR = "/data/shared/datasets/laion_fmri"

# Full region name -> key in the LAION pickle (same full names as the NSD loader where they exist).
_LAION_REGION_MAP = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
    "lateral": "lateral",
    "dorsal": "dorsal",
    **{r: r for r in ("V1", "V2", "V3", "hV4", "FFA", "PPA")},
}


def laion_stimuli_path() -> str:
    raw = os.environ.get("LAION_FMRI_DIR", LAION_DEFAULT_RAW_DIR)
    return os.path.join(raw, "stimuli/images/task-images_stimuli.h5")


def load_all_laion_data(cfg, subjects=None, regions=None) -> dict:
    """LAION-fMRI responses of the requested subjects (indices into LAION_SUBJECTS) and regions (full names).

    Same keys as load_all_nsd_data ("units": voxel x, y, z). Train = each subject's own images, test = the 1492 images every subject saw.
    cfg['ncsnr_threshold'] may only raise the 0.2 used to build the pickle.
    """
    subjects = subjects if subjects is not None else list(range(len(LAION_SUBJECTS)))
    region_pairs = [(key, name) for name, key in _LAION_REGION_MAP.items() if regions is None or name in regions]
    laion = load_pickle(os.path.join(get_env_var("LAION_DATA_DIR"), "laion_data.pkl"))
    shared_ids = laion["shared_ids"]
    threshold = cfg.get("ncsnr_threshold", laion["ncsnr_threshold"])
    if threshold is None or threshold < laion["ncsnr_threshold"]:
        raise ValueError(f"laion_data.pkl keeps only voxels with ncsnr > {laion['ncsnr_threshold']}")

    neural, voxel_counts, voxel_ncsnr, units = {}, {}, {}, {}
    all_stimulus_ids, per_subject_test_ids = set(), []
    for region_key, region_full in region_pairs:
        neural[region_full], voxel_counts[region_full], voxel_ncsnr[region_full], units[region_full] = {}, {}, {}, {}
        for subj in subjects:
            xr = laion["data"][region_key][subj]
            total = laion["voxel_counts"][region_key][subj][1]
            xr = xr.isel(neuroid=np.flatnonzero(xr.ncsnr.values > threshold))
            voxel_ncsnr[region_full][subj] = xr.ncsnr.values
            units[region_full][subj] = np.stack([xr[a].values for a in "xyz"], 1).astype(np.int64)
            voxel_counts[region_full][subj] = {"total": total, "retained": xr.sizes["neuroid"],
                                               "excluded": total - xr.sizes["neuroid"]}
            ids = [int(i) for i in xr.stimulus.values]
            all_stimulus_ids.update(ids)
            values = xr.values
            neural[region_full][subj] = {
                "train": {str(i): values[k] for k, i in enumerate(ids) if i not in shared_ids},
                "test": {str(i): values[k] for k, i in enumerate(ids) if i in shared_ids},
            }
            if region_key == region_pairs[0][0]:
                per_subject_test_ids.append(set(neural[region_full][subj]["test"]))

    test_ids = sorted(set.intersection(*per_subject_test_ids), key=int)
    source = laion["stimuli"]["dataset"]
    return {
        "regions": [name for _, name in region_pairs],
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": LazyEncodedHdf5Dict(laion_stimuli_path(), all_stimulus_ids),
        "shared_test_ids": test_ids,
        "nsd_source_ids": [str(i) for i in all_stimulus_ids if source.iat[int(i)] == "NSD"],
        "test_subsets": {"no_ood": [i for i in test_ids if source.iat[int(i)] != "OOD"],
                         **{src.lower(): [i for i in test_ids if source.iat[int(i)] == src]
                            for src in ("NSD", "LAION", "THINGSplus", "OOD")}},
        "ncsnr_threshold": threshold,
        "voxel_counts": voxel_counts,
        "ncsnr": voxel_ncsnr,
        "units": units,
    }


def shared_image_sources(name: str) -> dict[str, str]:
    """{stimulus id: source} of the shared images seen by every subject of LAION or NSD.

    NSD: 907 of shared1000, since subjects 3, 4, 6 and 8 did not finish all sessions.
    """
    if name == "nsd":
        source = load_nsd_pickle(os.environ.get("NSD_DATA_FILE", "nsd_data.pkl"))
    else:
        source = load_pickle(os.path.join(get_env_var("LAION_DATA_DIR"), "laion_data.pkl"))
    block = source["data"]["early"]
    ids = set(source["shared_ids"]).intersection(
        *({int(i) for i in block[s].coords["stimulus"].values} for s in sorted(block)))
    return {str(i): "nsd" if name == "nsd" else source["stimuli"]["dataset"].iat[i] for i in sorted(ids)}


# Triple-N responses and source folds

TRIPLEN_SUBJECTS = ["M1", "M2", "M3", "M4", "M5"]
N_FOLDS, FOLD_SEED = 10, 42

# Full region name -> key in the Triple-N pickle (same full names as the NSD loader where they exist).
_TRIPLEN_REGION_MAP = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
    "V1": "V1",
    "V2": "V2",
    "V4": "V4",
    "IT": "IT",
}


def load_triplen_pickle(path=None):
    """The preprocessed pickle (cached, so looping over folds reads it once)."""
    return load_pickle(path or os.path.join(get_env_var("TRIPLEN_DATA_DIR"), "triplen_data.pkl"))


def triplen_subjects_with(regions=None) -> list[int]:
    """Monkeys (indices into TRIPLEN_SUBJECTS) with units in every requested region (early areas: M1 and M4 only)."""
    data = load_triplen_pickle()["data"]
    keys = [k for name, k in _TRIPLEN_REGION_MAP.items() if regions is None or name in regions]
    return [s for s in range(len(TRIPLEN_SUBJECTS)) if all(s in data[k] for k in keys)]


def load_all_triplen_data(cfg, subjects=None, regions=None, fold: int | str = 0) -> dict:
    """Triple-N responses of the requested monkeys (indices into TRIPLEN_SUBJECTS) and regions (full names) for one fold.

    Same keys as load_all_nsd_data; "ncsnr" holds each unit's split-half reliability (its noise-ceiling analogue),
    "voxel_counts" counts units and "units" holds each unit's (session, unit) id. Test = the 100 images of `fold`, train = the other 900, for every monkey;
    fold="half": test = folds 0-4, train = folds 5-9 (500 / 500).
    cfg['reliability_threshold'] may only raise the 0.4 used to build the pickle; cfg['unit_types'] optionally keeps
    only those BombCell types (1 single unit, 2 multi-unit, 3 non-somatic). subjects=None loads every monkey with
    units in all requested regions.
    """
    if fold != "half" and not 0 <= fold < N_FOLDS:
        raise ValueError(f"fold must be in [0, {N_FOLDS}) or 'half'")
    tn = load_triplen_pickle()
    region_pairs = [(key, name) for name, key in _TRIPLEN_REGION_MAP.items() if regions is None or name in regions]
    subjects = subjects if subjects is not None else triplen_subjects_with(regions)
    threshold = cfg.get("reliability_threshold", tn["reliability_threshold"])
    if threshold is None or threshold < tn["reliability_threshold"]:
        raise ValueError(f"triplen_data.pkl keeps only units with reliability > {tn['reliability_threshold']}")
    unit_types = cfg.get("unit_types")

    neural, unit_counts, reliability, units = {}, {}, {}, {}
    test_ids = None
    for region_key, region_full in region_pairs:
        neural[region_full], unit_counts[region_full], reliability[region_full], units[region_full] = {}, {}, {}, {}
        for subj in subjects:
            if subj not in tn["data"][region_key]:
                raise KeyError(f"no {region_full} units for monkey {TRIPLEN_SUBJECTS[subj]} (have: "
                               f"{[TRIPLEN_SUBJECTS[s] for s in tn['data'][region_key]]})")
            xr = tn["data"][region_key][subj]
            total = tn["unit_counts"][region_key][subj][1]
            keep = xr.reliability.values > threshold
            if unit_types is not None:
                keep &= np.isin(xr.unit_type.values, list(unit_types))
            xr = xr.isel(neuroid=np.flatnonzero(keep))
            if xr.sizes["neuroid"] == 0:
                raise ValueError(f"no units left for monkey {TRIPLEN_SUBJECTS[subj]}, region {region_full}")
            reliability[region_full][subj] = xr.reliability.values
            units[region_full][subj] = np.stack([xr.session.values, xr.unit.values], 1).astype(np.int64)
            unit_counts[region_full][subj] = {"total": total, "retained": xr.sizes["neuroid"],
                                              "excluded": total - xr.sizes["neuroid"]}
            in_test = xr.fold.values < N_FOLDS // 2 if fold == "half" else xr.fold.values == fold
            ids, values = xr.stimulus.values, xr.values
            neural[region_full][subj] = {
                "train": {str(i): values[k] for k, i in enumerate(ids) if not in_test[k]},
                "test": {str(i): values[k] for k, i in enumerate(ids) if in_test[k]},
            }
            test_ids = sorted(neural[region_full][subj]["test"], key=int)

    all_ids = tn["stimuli"]["nsdId"].tolist()
    return {
        "regions": [name for _, name in region_pairs],
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": LazyHdf5Dict(nsd_stimuli_path(), "imgBrick", all_ids),
        "shared_test_ids": test_ids,
        "fold": fold,
        "ncsnr_threshold": threshold,
        "voxel_counts": unit_counts,
        "ncsnr": reliability,
        "units": units,
    }


# TVSD responses and official split

TVSD_SUBJECTS = ["F", "N"]
TVSD_DEFAULT_DATA_DIR = "/data/shared/datasets/papale2025.tvsd/processed"

# Full region name -> key in the TVSD pickle (same full names as the NSD / Triple-N loaders where they exist).
_TVSD_REGION_MAP = {
    "early visual stream": "early",  # V1 + V4
    "ventral visual stream": "ventral",  # IT
    "V1": "V1",
    "V4": "V4",
    "IT": "IT",
}


def tvsd_data_dir() -> str:
    try:
        return get_env_var("TVSD_DATA_DIR")
    except RuntimeError:
        return TVSD_DEFAULT_DATA_DIR


def tvsd_stimuli_path() -> str:
    return os.path.join(tvsd_data_dir(), "tvsd_stimuli.h5")


def load_tvsd_pickle(path=None):
    """The preprocessed pickle (cached)."""
    return load_pickle(path or os.path.join(tvsd_data_dir(), "tvsd_data.pkl"))


def tvsd_subjects_with(regions=None) -> list[int]:
    """Monkeys (indices into TVSD_SUBJECTS) with units in every requested region (both, for every region)."""
    data = load_tvsd_pickle()["data"]
    keys = [k for name, k in _TVSD_REGION_MAP.items() if regions is None or name in regions]
    return [s for s in range(len(TVSD_SUBJECTS)) if all(s in data[k] for k in keys)]


def load_all_tvsd_data(cfg, subjects=None, regions=None) -> dict:
    """TVSD responses of the requested monkeys (indices into TVSD_SUBJECTS) and regions (full names).

    Same keys as load_all_triplen_data (minus "fold"); "ncsnr" holds each electrode's oracle correlation (its
    noise-ceiling analogue), "voxel_counts" counts electrodes and "units" holds each electrode's index (0-1023). Train = the 22,248 single-presentation images,
    test = the 100 repeated images, identical for both monkeys. cfg['oracle_threshold'] may only raise the value used
    to build the pickle (default 0.4). subjects=None loads every monkey with units in all requested regions.
    """
    tv = load_tvsd_pickle()
    region_pairs = [(key, name) for name, key in _TVSD_REGION_MAP.items() if regions is None or name in regions]
    subjects = subjects if subjects is not None else tvsd_subjects_with(regions)
    threshold = cfg.get("oracle_threshold", tv["oracle_threshold"])
    if threshold is None or threshold < tv["oracle_threshold"]:
        raise ValueError(f"tvsd_data.pkl keeps only electrodes with oracle > {tv['oracle_threshold']}")

    neural, unit_counts, oracle, units = {}, {}, {}, {}
    test_ids = None
    for region_key, region_full in region_pairs:
        neural[region_full], unit_counts[region_full], oracle[region_full], units[region_full] = {}, {}, {}, {}
        for subj in subjects:
            if subj not in tv["data"][region_key]:
                raise KeyError(f"no {region_full} units for monkey {TVSD_SUBJECTS[subj]} (have: "
                               f"{[TVSD_SUBJECTS[s] for s in tv['data'][region_key]]})")
            xr = tv["data"][region_key][subj]
            total = tv["unit_counts"][region_key][subj][1]
            xr = xr.isel(neuroid=np.flatnonzero(xr.oracle.values > threshold))
            if xr.sizes["neuroid"] == 0:
                raise ValueError(f"no electrodes left for monkey {TVSD_SUBJECTS[subj]}, region {region_full}")
            oracle[region_full][subj] = xr.oracle.values
            units[region_full][subj] = xr.electrode.values[:, None].astype(np.int64)
            unit_counts[region_full][subj] = {"total": total, "retained": xr.sizes["neuroid"],
                                              "excluded": total - xr.sizes["neuroid"]}
            ids, in_test, values = xr.stimulus.values, xr.split.values == "test", xr.values
            neural[region_full][subj] = {
                "train": {str(i): values[k] for k, i in enumerate(ids) if not in_test[k]},
                "test": {str(i): values[k] for k, i in enumerate(ids) if in_test[k]},
            }
            test_ids = sorted(neural[region_full][subj]["test"], key=int)

    return {
        "regions": [name for _, name in region_pairs],
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": LazyEncodedHdf5Dict(tvsd_stimuli_path(), tv["stimuli"]["stimulus_id"].tolist()),
        "shared_test_ids": test_ids,
        "ncsnr_threshold": threshold,
        "voxel_counts": unit_counts,
        "ncsnr": oracle,
        "units": units,
    }
