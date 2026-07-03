"""Data loaders, grouped by the two paradigms a dataset can take. ``load_split`` is the
single entry point.

- **timeline** (text/meg/ieeg) -- continuous-timeline studies aligned on the fly via the
  NeuralSet ``Segmenter``: each trigger event cuts a stimulus + neural segment.
- **vision** (nsd/tvsd/nnn) -- pre-aligned response tables (prepared betas / spiking MUA),
  read directly by a self-contained ``loaders.vision`` reader; no Segmenter.

A model is read out by the harness against ``load_split``'s (X_train, X_test, Y_train,
Y_test); the agent should NOT need to edit anything here.
"""
from __future__ import annotations

from harness.datasets import Dataset


_SPLIT_MEMO: dict = {}


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test, roi_masks, nc). Memoized per-process by the (frozen,
    hashable) ``Dataset``: the search evaluator scores many models against the SAME split, so the load
    is paid ONCE, not once per model. The split is treated read-only downstream (the reward copies X
    to the GPU per eval), so sharing it across models is safe."""
    if d not in _SPLIT_MEMO:
        _SPLIT_MEMO[d] = _dispatch_split(d)
    return _SPLIT_MEMO[d]


def _dispatch_split(d: Dataset):
    """Load (X_train, X_test, Y_train, Y_test, roi_masks, nc) -- 2-D float tensors, a (possibly
    empty) ``{roi_name: bool mask}`` over the scored targets for the per-ROI breakdown, and ``nc``
    (per-voxel R^2 noise ceiling aligned to the scored targets, for noise-normalizing the reward, or
    None when the dataset has no repeat-based ceiling). Dispatches on d.loader; imports are lazy so a
    score-only caller never pulls a loader's heavy deps."""
    if d.loader == "timeline":
        if d.timeline is None:
            raise ValueError(f"{d.study}: loader='timeline' requires a timeline=Timeline(...) block")
        from harness.loaders.timeline import load_timeline
        return (*load_timeline(d), {}, None)       # timeline studies (text/meg/ieeg): no ROI map / ceiling
    if d.loader == "nsd":
        from harness.loaders.vision.nsd import load_nsd
        return load_nsd(d)                         # nsd provides the per-voxel ceiling
    if d.loader == "tvsd":
        from harness.loaders.vision.tvsd import load_tvsd
        return load_tvsd(d)                        # tvsd provides the per-electrode oracle ceiling
    if d.loader == "nnn":
        from harness.loaders.vision.nnn import load_nnn
        return (*load_nnn(d), None)
    raise ValueError(f"unknown loader {d.loader!r}; expected timeline|nsd|tvsd|nnn")
