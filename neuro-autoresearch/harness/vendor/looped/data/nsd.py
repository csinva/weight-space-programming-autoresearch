"""NSD (Natural Scenes Dataset) fMRI responses and stimuli."""
import os

import nibabel as nib
import numpy as np
from dotenv import load_dotenv

from harness.vendor.looped import utils
from harness.vendor.looped.data.stimuli import LazyHdf5Dict

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

# Every ROI of the extended pickle, as full region names.
ALL_ROIS = ["V1", "V2", "V3", "hV4", "VO1", "VO2", "PHC1", "PHC2", "LO1", "LO2", "TO1", "TO2", "V3A", "V3B",
            "FFA", "PPA", "early visual stream", "midventral", "ventral visual stream", "midlateral", "lateral"]

# The five NSD "streams" parcels (pickle keys), which together tile visual cortex.
STREAMS = ["early", "midventral", "ventral", "midlateral", "lateral"]

_NSD_SUBJECTS = list(range(8))


def nsd_root() -> str:
    """NSD release root: $NSD_ROOT (env or .env), else the gpu2 shared copy."""
    load_dotenv()
    return os.environ.get("NSD_ROOT", "/data/shared/datasets/allen2021.natural_scenes")


def nsd_stimuli_path() -> str:
    return os.environ.get("NSD_STIMULI_HDF5_PATH", f"{nsd_root()}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")


def nsd_betas_dir() -> str:
    return os.environ.get("NSD_BETAS_DIR", f"{nsd_root()}/nsddata_betas")


def load_nsd_pickle(name):
    """A pickle from NSD_DATA_DIR, e.g. 'nsd_data.pkl' or 'nsd_data_ext.pkl'."""
    return utils.load_pickle(os.path.join(utils.get_env_var("NSD_DATA_DIR"), name))


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


def load_stream_voxels(nsd, subject, ncsnr_threshold):
    """Responses (train ids then test ids) of every reliable stream voxel, its coordinates and stream index."""
    parts = []
    for k, region in enumerate(STREAMS):
        xr = nsd["data"][region][subject]
        xr = xr.isel(neuroid=np.flatnonzero(load_voxel_ncsnr(xr, subject) > ncsnr_threshold))
        parts.append((k, xr))
    ids = [int(i) for i in parts[0][1].stimulus.values]
    tr_ids = [i for i in ids if i not in nsd["shared_ids"]]
    te_ids = [i for i in ids if i in nsd["shared_ids"]]
    Y = np.concatenate([xr.sel(stimulus=tr_ids + te_ids).values for _, xr in parts], axis=1)
    vox = {a: np.concatenate([xr[a].values.astype(np.int16) for _, xr in parts]) for a in "xyz"}
    vox["stream"] = np.concatenate([np.full(xr.sizes["neuroid"], k, np.int8) for k, xr in parts])
    return [str(i) for i in tr_ids + te_ids], len(tr_ids), Y, vox
