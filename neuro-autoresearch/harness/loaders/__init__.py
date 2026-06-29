"""Data loaders, grouped by the two paradigms a dataset can take. ``load_split`` is the
single entry point.

- **neuralset** (text/meg/ieeg) -- continuous-timeline studies aligned on the fly via the
  NeuralSet ``Segmenter``: each trigger event cuts a stimulus + neural segment.
- **standalone** (nsd/tvsd/nnn) -- pre-aligned response tables (prepared betas / spiking
  MUA), read directly by a self-contained reader; no Segmenter.

A model is read out by the harness against ``load_split``'s (X_train, X_test, Y_train,
Y_test); the agent should NOT need to edit anything here.
"""
from __future__ import annotations

from harness.datasets import Dataset


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test, roi_masks, nc) -- 2-D float tensors, a (possibly
    empty) ``{roi_name: bool mask}`` over the scored targets for the per-ROI breakdown, and ``nc``
    (per-voxel R^2 noise ceiling aligned to the scored targets, for noise-normalizing the reward, or
    None when the dataset has no repeat-based ceiling). Dispatches on d.loader; imports are lazy so a
    score-only caller never pulls a loader's heavy deps."""
    if d.loader == "timeline":
        from harness.loaders.timeline import load_neuralset
        return (*load_neuralset(d), {}, None)      # timeline studies (text/meg/ieeg): no ROI map / ceiling
    if d.loader == "nsd":
        from harness.loaders.vision.nsd import load_nsd
        return load_nsd(d)                         # nsd provides the per-voxel ceiling
    if d.loader == "tvsd":
        from harness.loaders.vision.tvsd import load_tvsd
        return load_tvsd(d)                        # tvsd provides the per-electrode oracle ceiling
    if d.loader == "nnn":
        from harness.loaders.vision.nnn import load_nnn
        return (*load_nnn(d), None)
    raise ValueError(f"unknown loader {d.loader!r}; expected neuralset|nsd|tvsd|nnn")
