"""Data loaders, grouped by the two paradigms a dataset can take. ``load_split`` is the
single entry point.

- **timeline** (iEEG) -- continuous-timeline studies aligned on the fly via the
  NeuralSet ``Segmenter``: each trigger event cuts a stimulus + neural segment.
- **vision** (laion/triplen/nsd/tvsd) -- prepared response tables read by the
  vendored looped-networks readers. RSA search has an empty train split; final
  encoding retains the source train/test split.

A model is read out by the harness against ``load_split``'s (X_train, X_test, Y_train,
Y_test); the agent should NOT need to edit anything here.
"""
from __future__ import annotations

import os

from harness.datasets import Dataset
from harness.protocols import VISION_DATASETS


_SPLIT_MEMO: dict = {}

# Storage and legacy-loader settings enter the memo key; vision scientific
# thresholds are fixed. Resolution and subject already live on the Dataset.
_SPLIT_ENV_KEYS = (
    "NSD_DATA_DIR", "NSD_DATA_FILE", "NSD_ROOT", "NSD_BETAS_DIR", "NSD_STIMULI_HDF5_PATH",
    "LAION_DATA_DIR", "LAION_FMRI_DIR", "TRIPLEN_DATA_DIR", "TVSD_DATA_DIR",
    "NEURO_HUTH_NUM_TRAIN", "NEURO_NO_CACHE", "NEURO_SPLIT_CACHE",
)


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test, roi_masks, nc). Memoized per-process by the frozen
    ``Dataset`` plus the split-shaping env knobs: the search evaluator scores many models against the
    SAME split, so the load is paid ONCE, not once per model. The split is treated read-only
    downstream (the reward copies/clones X per eval), so sharing it across models is safe."""
    source_key = d
    if d.loader in VISION_DATASETS:
        from harness.loaders.vision.looped import dataset_cache_key
        source_key = dataset_cache_key(d)
    key = (source_key, tuple(os.environ.get(k) for k in _SPLIT_ENV_KEYS))
    if key not in _SPLIT_MEMO:
        _SPLIT_MEMO[key] = _dispatch_split(d)
    return _SPLIT_MEMO[key]


def _dispatch_split(d: Dataset):
    """Load (X_train, X_test, Y_train, Y_test, roi_masks, nc) -- 2-D float tensors, a (possibly
    empty) ``{roi_name: bool mask}`` over targets for the per-region breakdown, and
    ``nc`` (None for canonical raw RSA/encoding metrics). Imports are lazy so a
    score-only caller never pulls a loader's heavy deps."""
    if d.loader == "timeline":
        if d.timeline is None:
            raise ValueError(f"{d.study}: loader='timeline' requires a timeline=Timeline(...) block")
        from harness.loaders.timeline import load_timeline
        return (*load_timeline(d), {}, None)       # timeline studies (iEEG): no ROI map / ceiling
    if d.loader in VISION_DATASETS:
        from harness.loaders.vision.looped import load_looped
        return load_looped(d)
    if d.loader == "huth":
        from harness.loaders.huth import load_huth
        return load_huth(d)                         # Huth language fMRI: baked blob, no ceiling
    raise ValueError(f"unknown loader {d.loader!r}; expected timeline|laion|nsd|triplen|tvsd|huth")


def apply_feature_pipeline(d: Dataset, A, which: str):
    """Post-forward feature transform hook used by ``harness.reward`` (lazy so score-only callers
    don't import a loader's deps). Dispatches on ``d.feature_pipeline``; currently only the Huth
    fMRI-language ``"lanczos_fir"`` pipeline (Lanczos word->TR + z-score + FIR delays)."""
    from harness.loaders.huth import apply_feature_pipeline as _huth
    return _huth(d, A, which)
