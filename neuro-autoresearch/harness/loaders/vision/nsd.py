"""Compatibility entry point for the vendored looped-networks nsd reader."""
from harness.loaders.vision.looped import load_looped


def load_nsd(dataset):
    return load_looped(dataset)
