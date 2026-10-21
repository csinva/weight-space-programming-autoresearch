"""Triple-N macaque Neuropixels responses and stimuli, in the same layout as looped.data.nsd.load_all_nsd_data.

Responses come from $TRIPLEN_DATA_DIR/triplen_data.pkl (prepare_data/triplen.py): per monkey and region, each
unit's trial-averaged rate to the 1000 NSD shared1000 images, z-scored over images, units already kept at split-half
reliability > 0.4. Stimuli are the NSD images themselves (nsd_stimuli.hdf5, indexed by nsdId).

Every image is NSD-shared, so train/test is an image-level 10-fold split stored in the pickle
(KFold(10, shuffle=True, random_state=42)): fold k tests its 100 images and trains on the other 900. fold="half"
is a 50/50 split of the same assignment: test = folds 0-4 (500 images), train = folds 5-9 (500).
"""
import functools
import os

import numpy as np

from harness.vendor.looped import utils
from harness.vendor.looped.data.nsd import nsd_stimuli_path
from harness.vendor.looped.data.stimuli import LazyHdf5Dict

SUBJECTS = ["M1", "M2", "M3", "M4", "M5"]
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
    return utils.load_pickle(path or os.path.join(utils.get_env_var("TRIPLEN_DATA_DIR"), "triplen_data.pkl"))


def subjects_with(regions=None) -> list[int]:
    """Monkeys (indices into SUBJECTS) with units in every requested region (early areas: M1 and M4 only)."""
    data = load_triplen_pickle()["data"]
    keys = [k for name, k in _TRIPLEN_REGION_MAP.items() if regions is None or name in regions]
    return [s for s in range(len(SUBJECTS)) if all(s in data[k] for k in keys)]


def load_all_triplen_data(cfg, subjects=None, regions=None, fold: int | str = 0) -> dict:
    """Triple-N responses of the requested monkeys (indices into SUBJECTS) and regions (full names) for one fold.

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
    subjects = subjects if subjects is not None else subjects_with(regions)
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
                raise KeyError(f"no {region_full} units for monkey {SUBJECTS[subj]} (have: "
                               f"{[SUBJECTS[s] for s in tn['data'][region_key]]})")
            xr = tn["data"][region_key][subj]
            total = tn["unit_counts"][region_key][subj][1]
            keep = xr.reliability.values > threshold
            if unit_types is not None:
                keep &= np.isin(xr.unit_type.values, list(unit_types))
            xr = xr.isel(neuroid=np.flatnonzero(keep))
            if xr.sizes["neuroid"] == 0:
                raise ValueError(f"no units left for monkey {SUBJECTS[subj]}, region {region_full}")
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
