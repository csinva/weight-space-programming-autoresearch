"""TVSD (THINGS Ventral Stream Spiking Dataset, Papale et al. 2025) responses and stimuli, in the layout of
looped.data.triplen.load_all_triplen_data but with the dataset's official fixed split and no folds.

Responses come from $TVSD_DATA_DIR/tvsd_data.pkl (prepare_data/tvsd.py): per monkey (F, N) and region, each
electrode's normalised multi-unit activity to 22,248 training images (one presentation each) and 100 test images
(mean of 30 repeats), z-scored with the training mean / sd, electrodes already kept at oracle > 0.4. Stimuli are the
THINGS images packed in $TVSD_DATA_DIR/tvsd_stimuli.h5 (row i = stimulus id i); ids 0..22247 are the training
images, 22248..22347 the test images. $TVSD_DATA_DIR defaults to /data/shared/datasets/papale2025.tvsd/processed.
"""
import functools
import os

import numpy as np

from neuro_eval.reference import utils
from neuro_eval.reference.data.stimuli import LazyEncodedHdf5Dict

SUBJECTS = ["F", "N"]
DEFAULT_DATA_DIR = "/data/shared/datasets/papale2025.tvsd/processed"

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
        return utils.get_env_var("TVSD_DATA_DIR")
    except RuntimeError:
        return DEFAULT_DATA_DIR


def tvsd_stimuli_path() -> str:
    return os.path.join(tvsd_data_dir(), "tvsd_stimuli.h5")


def load_tvsd_pickle(path=None):
    """The preprocessed pickle (cached)."""
    return utils.load_pickle(path or os.path.join(tvsd_data_dir(), "tvsd_data.pkl"))


def subjects_with(regions=None) -> list[int]:
    """Monkeys (indices into SUBJECTS) with units in every requested region (both, for every region)."""
    data = load_tvsd_pickle()["data"]
    keys = [k for name, k in _TVSD_REGION_MAP.items() if regions is None or name in regions]
    return [s for s in range(len(SUBJECTS)) if all(s in data[k] for k in keys)]


def load_all_tvsd_data(cfg, subjects=None, regions=None) -> dict:
    """TVSD responses of the requested monkeys (indices into SUBJECTS) and regions (full names).

    Same keys as load_all_triplen_data (minus "fold"); "ncsnr" holds each electrode's oracle correlation (its
    noise-ceiling analogue), "voxel_counts" counts electrodes and "units" holds each electrode's index (0-1023). Train = the 22,248 single-presentation images,
    test = the 100 repeated images, identical for both monkeys. cfg['oracle_threshold'] may only raise the value used
    to build the pickle (default 0.4). subjects=None loads every monkey with units in all requested regions.
    """
    tv = load_tvsd_pickle()
    region_pairs = [(key, name) for name, key in _TVSD_REGION_MAP.items() if regions is None or name in regions]
    subjects = subjects if subjects is not None else subjects_with(regions)
    threshold = cfg.get("oracle_threshold", tv["oracle_threshold"])
    if threshold is None or threshold < tv["oracle_threshold"]:
        raise ValueError(f"tvsd_data.pkl keeps only electrodes with oracle > {tv['oracle_threshold']}")

    neural, unit_counts, oracle, units = {}, {}, {}, {}
    test_ids = None
    for region_key, region_full in region_pairs:
        neural[region_full], unit_counts[region_full], oracle[region_full], units[region_full] = {}, {}, {}, {}
        for subj in subjects:
            if subj not in tv["data"][region_key]:
                raise KeyError(f"no {region_full} units for monkey {SUBJECTS[subj]} (have: "
                               f"{[SUBJECTS[s] for s in tv['data'][region_key]]})")
            xr = tv["data"][region_key][subj]
            total = tv["unit_counts"][region_key][subj][1]
            xr = xr.isel(neuroid=np.flatnonzero(xr.oracle.values > threshold))
            if xr.sizes["neuroid"] == 0:
                raise ValueError(f"no electrodes left for monkey {SUBJECTS[subj]}, region {region_full}")
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
