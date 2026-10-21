"""LAION-fMRI responses and stimuli, in the same layout as looped.data.nsd.load_all_nsd_data.

Responses come from $LAION_DATA_DIR/laion_data.pkl (prepare_data/laion_fmri.py): per subject, each image's
betas averaged over repeats, voxels already kept at ncsnr > 0.2. Stimuli are the encoded images in
$LAION_FMRI_DIR/stimuli/images/task-images_stimuli.h5 (row i = row i of task-images_metadata.csv = stimulus id i),
decoded as displayed in the scanner (transparent PNGs over mid-grey).
"test_subsets": "no_ood" = the 1121 shared images that are not out-of-distribution (metadata dataset != "OOD"), and
one subset per image source (metadata dataset): "nsd" (240), "laion" (641), "thingsplus" (240), "ood" (371).
"""
import os

import numpy as np

from harness.vendor.looped import utils
from harness.vendor.looped.data.stimuli import LazyEncodedHdf5Dict

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
DEFAULT_RAW_DIR = "/data/shared/datasets/laion_fmri"

# Full region name -> key in the LAION pickle (same full names as the NSD loader where they exist).
_LAION_REGION_MAP = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
    "lateral": "lateral",
    "dorsal": "dorsal",
    **{r: r for r in ("V1", "V2", "V3", "hV4", "FFA", "PPA")},
}


def laion_stimuli_path() -> str:
    raw = os.environ.get("LAION_FMRI_DIR", DEFAULT_RAW_DIR)
    return os.path.join(raw, "stimuli/images/task-images_stimuli.h5")


def load_all_laion_data(cfg, subjects=None, regions=None) -> dict:
    """LAION-fMRI responses of the requested subjects (indices into SUBJECTS) and regions (full names).

    Same keys as load_all_nsd_data ("units": voxel x, y, z). Train = each subject's own images, test = the 1492 images every subject saw.
    cfg['ncsnr_threshold'] may only raise the 0.2 used to build the pickle.
    """
    subjects = subjects if subjects is not None else list(range(len(SUBJECTS)))
    region_pairs = [(key, name) for name, key in _LAION_REGION_MAP.items() if regions is None or name in regions]
    laion = utils.load_pickle(os.path.join(utils.get_env_var("LAION_DATA_DIR"), "laion_data.pkl"))
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
        "test_subsets": {"no_ood": [i for i in test_ids if source.iat[int(i)] != "OOD"],
                         **{src.lower(): [i for i in test_ids if source.iat[int(i)] == src]
                            for src in ("NSD", "LAION", "THINGSplus", "OOD")}},
        "ncsnr_threshold": threshold,
        "voxel_counts": voxel_counts,
        "ncsnr": voxel_ncsnr,
        "units": units,
    }
