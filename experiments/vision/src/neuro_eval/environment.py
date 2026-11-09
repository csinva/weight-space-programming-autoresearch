"""Scoped environment for held-out evaluation under the frozen search protocol."""
import contextlib
import os

from neuro_eval.benchmarks import protocol_for


@contextlib.contextmanager
def preserved_environment():
    """Restore the process environment after a scoped, nonconcurrent operation."""
    previous = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


@contextlib.contextmanager
def final_environment(selection: dict):
    protocol = protocol_for(selection["search_dataset"], region=selection["search_region"],
                            metric=selection["search_metric"] or "rsa")
    from neuro_eval.model import EVAL_ENV_KEYS
    previous = os.environ.copy()
    try:
        saved = selection["environment"]
        storage_keys = {"nsd": ("NSD_DATA_DIR", "NSD_DATA_FILE", "NSD_ROOT", "NSD_STIMULI_HDF5_PATH", "NSD_BETAS_DIR"),
                        "laion": ("LAION_DATA_DIR", "LAION_FMRI_DIR"),
                        "tvsd": ("TVSD_DATA_DIR", "TVSD_STIMULI_HDF5_PATH")}[protocol.final_dataset]
        storage_overrides = {key: previous[key] for key in storage_keys if key in previous}
        if saved:
            for key in EVAL_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ.update({k: str(saved[k]) for k in EVAL_ENV_KEYS if k in saved})
        os.environ.update(storage_overrides)
        os.environ.update(NEURO_DATASET=protocol.final_dataset, NEURO_STAGE="final",
                          NEURO_METRIC="encoding", NEURO_REGION=protocol.search_region)
        os.environ.pop("NEURO_SUBJECT", None)  # each cell scores its complete protocol cohort
        os.environ.pop("LAB_RUN", None)
        os.environ.pop("LAB_PROPOSER", None)
        yield protocol
    finally:
        os.environ.clear()
        os.environ.update(previous)
