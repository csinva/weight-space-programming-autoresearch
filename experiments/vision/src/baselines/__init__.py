"""Fixed reference models of visual cortex, scored under the exact search/final protocols.

Each module exports ``MODELS = {name: model}``; a model follows the candidate
contract (flattened HWC RGB images in [0, 1] -> (images, features), < 10,000
features) but runs in-process as trusted code, so pretrained weights are allowed.
"""
MODULES = ("pixels", "gabor", "scattering", "texture", "curvature", "gist", "hmax", "alexnet")


def load(names=None):
    """Return {name: model} for all baselines, or only ``names``."""
    import importlib
    models = {}
    for module in MODULES:
        models.update(importlib.import_module(f"baselines.{module}").MODELS)
    unknown = set(names or ()) - set(models)
    if unknown:
        raise KeyError(f"unknown baselines: {sorted(unknown)}")
    return {k: v for k, v in models.items() if not names or k in names}
