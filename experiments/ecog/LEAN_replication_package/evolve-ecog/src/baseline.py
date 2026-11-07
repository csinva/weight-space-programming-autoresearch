"""Linear ridge baseline embedder (melspec -> ECoG high-gamma).

This is the reference point the hand-built interpretable transformer in
`interpretable_transformer.py` tries to beat. It is run once by `setup_run.py`
to populate the baseline row of `results/overall_results.csv`.

The embedder is the **identity**: it returns the raw mel-spectrogram frames
unchanged, so the downstream pipeline (FIR delays + bootstrapped ridge) is just a
linear, temporally-delayed ridge regression from the 50 mel bins to the 166
electrodes. There are no model parameters to train and none are trained — exactly
the linear baseline the autoresearch loop aims to improve on with structured,
interpretable audio features.

It exposes the same embedder interface as every attempt:
`__call__(melspec (T, 50)) -> np.ndarray (T, 50)`.
"""
from __future__ import annotations

import numpy as np


class MelspecIdentityEmbedder:
    """Returns the melspec frames as-is — the linear-ridge-on-melspec baseline."""

    n_params = 0  # no model; purely the ridge readout the pipeline fits

    def __call__(self, melspec: np.ndarray) -> np.ndarray:
        return np.asarray(melspec, dtype=np.float32)  # (T, 50)
