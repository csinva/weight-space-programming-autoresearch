"""Compatibility entry point for the vendored looped-networks laion reader."""
from harness.loaders.vision.looped import load_looped


def load_laion(dataset):
    return load_looped(dataset)
