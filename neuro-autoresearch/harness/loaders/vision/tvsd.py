"""Compatibility entry point for the vendored looped-networks tvsd reader."""
from harness.loaders.vision.looped import load_looped


def load_tvsd(dataset):
    return load_looped(dataset)
