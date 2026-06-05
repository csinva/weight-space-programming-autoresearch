"""Fixed data layer: stimulus + neural -> aligned, split (X, Y) tensors.

The autoresearch agent should NOT need to edit this file.
"""
from __future__ import annotations

import numpy as np
import torch
import neuralset as ns
from neuralset.extractors.image import BaseClassicImageExtractor

from config import Dataset


class RawPixels(BaseClassicImageExtractor):
    """Flattened, [0,1]-normalised RGB pixels of each stimulus image."""

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        return (image.astype("float32") / 255.0).reshape(-1)


_NEURAL = {
    "meg": lambda: ns.extractors.MegExtractor(frequency=100.0),
    "eeg": lambda: ns.extractors.EegExtractor(frequency=100.0),
    "fmri": lambda: ns.extractors.FmriExtractor(),
}


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test) as 2-D float tensors."""
    path = d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)
    study = ns.Study(name=d.study, path=path)
    if d.download:
        study.download()
    try:
        events = study.run()
    except FileNotFoundError:           # cold demo cache: self-provision once
        study.download()
        events = study.run()

    seg = ns.Segmenter(
        extractors=dict(stim=RawPixels(imsize=d.imsize), neural=_NEURAL[d.neural]()),
        trigger_query=d.trigger, start=d.start, duration=d.duration,
        drop_incomplete=True,
    )
    batch = seg.apply(events).load_all()
    X = batch.data["stim"].float()
    Y = batch.data["neural"].float().reshape(batch.data["neural"].shape[0], -1)

    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(d.seed))
    n_test = max(1, int(round(n * d.test_frac)))
    te, tr = perm[:n_test], perm[n_test:]
    return X[tr], X[te], Y[tr], Y[te]
