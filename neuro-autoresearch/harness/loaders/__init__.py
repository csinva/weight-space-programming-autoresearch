"""Data loaders, grouped by dataset family. ``load_split`` is the single entry point.

- ``core``      -- shared machinery (RawPixels, build cache, ROI selection, repeat-collapse/split)
- ``neuralset`` -- the NeuralSet/Segmenter timeline path (nsd, text, meg, ieeg)
- ``tvsd``      -- THINGS macaque-spiking pre-averaged table
- ``nnn``       -- Triple-N macaque-spiking pre-averaged table (NSD-1000)

A model is read out by the harness against ``load_split``'s (X_train, X_test, Y_train,
Y_test); the agent should NOT need to edit anything here.
"""
from __future__ import annotations

from harness.datasets import Dataset
from harness.loaders.neuralset import load_neuralset
from harness.loaders.tvsd import load_tvsd
from harness.loaders.nnn import load_nnn


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test) as 2-D float tensors.

    Dispatches by the dataset's loader: pre-averaged tables (``tvsd``/``nnn``) take their
    self-contained readers; everything else goes through the NeuralSet/Segmenter timeline
    path (``neuralset``). See those modules for the per-family split logic.
    """
    if d.loader == "nsd_betas":
        from harness.loaders.nsd_betas import load_nsd_betas
        return load_nsd_betas(d)
    if d.loader == "table":
        return load_nnn(d) if d.study.startswith("Nnn") else load_tvsd(d)
    return load_neuralset(d)
