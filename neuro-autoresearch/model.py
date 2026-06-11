"""THE editable artifact: the model the autoresearch loop evolves.

A model maps a stimulus batch ``X (N, D_in)`` to activations ``(N, D_out)``. Its
weights are written DIRECTLY in code -- it is never trained (no optimizer, no
backprop). Only the ridge readout in ``harness/reward.py`` is fit, and that fit
*is* the evaluation. The reward is the held-out ``test_corr`` it produces.

The loop is: edit this one model, run ``python -m harness.run``, read the reward,
edit again. You start from the identity below and refine it to push ``test_corr``
up. Do NOT edit anything under ``harness/`` -- that fixedness is what makes the
reward trustworthy. Run-config (which slices to log) lives in ``experiment.py``;
everything you change about the *model* lives here.
"""
from __future__ import annotations

import torch


class Model:
    """The current proposal. Seeded as the identity baseline -- evolve it.

    ``name``/``idea``/``summary`` are logged verbatim with each score. ``idea`` is
    your forward-looking hypothesis for this iteration; ``summary`` is what the model
    actually does. Keep each to AT MOST two extremely dense, information-heavy
    sentences. ``n_params`` is the count of weights you wrote in code (0 for a
    parameter-free transform).
    """

    name = "identity"
    idea = "Raw stimulus features straight into the ridge readout -- the baseline to beat."
    summary = "Identity transform: returns the stimulus features unchanged, so the ridge readout fits directly on raw inputs."
    n_params = 0

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X


# The single model instance the loop evaluates.
MODEL = Model()
