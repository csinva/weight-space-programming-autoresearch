"""Cross-paradigm loader primitives shared by every dataset loader: the stimulus
pixel extractor (RawPixels), the on-disk split-cache dir, the NeuralSet study-path
resolver, and the train-constant-column drop. The autoresearch agent should NOT edit
this file.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import neuralset as ns
from neuralset.extractors.image import BaseClassicImageExtractor

from harness.datasets import Dataset


class RawPixels(BaseClassicImageExtractor):
    """Flattened, [0,1]-normalised RGB pixels of each stimulus image."""

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        return (image.astype("float32") / 255.0).reshape(-1)


def _study_path(d):
    """Resolve the NeuralSet study path: explicit ``d.path`` if set, else ``None`` to
    download-from-scratch when ``d.download``, else the shared ``ns.CACHE_FOLDER``."""
    return d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)


def _cache_dir() -> str:
    return os.environ.get(
        "NEURO_SPLIT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "neuro-autoresearch", "splits"),
    )


def drop_constant_cols(Y_train, *others):
    """Drop target columns that are constant on the TRAIN split (no signal; they only
    dilute the mean-r and waste compute). Mask uses train variance only (no test peeking).
    Returns (Y_train, *others) each column-subset by the same keep mask."""
    keep = Y_train.std(0) > 0
    return tuple(Y[:, keep] for Y in (Y_train, *others))
