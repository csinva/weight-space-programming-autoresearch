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
    """Return (X_train, X_test, Y_train, Y_test, roi_masks) -- 2-D float tensors plus a (possibly
    empty) ``{roi_name: bool mask}`` over the scored targets for the per-ROI breakdown. Dispatches on
    d.loader; imports are lazy so a score-only caller never pulls a loader's heavy deps."""
    if d.loader == "neuralset":
        from harness.loaders.neuralset import load_neuralset
        return (*load_neuralset(d), {})            # timeline studies (text/meg/ieeg): no ROI map
    if d.loader == "nsd":
        from harness.loaders.standalone.nsd import load_nsd
        return load_nsd(d)
    if d.loader == "tvsd":
        from harness.loaders.standalone.tvsd import load_tvsd
        return load_tvsd(d)
    if d.loader == "nnn":
        from harness.loaders.standalone.nnn import load_nnn
        return load_nnn(d)
    raise ValueError(f"unknown loader {d.loader!r}; expected neuralset|nsd|tvsd|nnn")
