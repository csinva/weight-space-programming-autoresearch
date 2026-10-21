"""Compatibility entry point for the vendored looped-networks triplen reader."""
from harness.loaders.vision.looped import load_looped


def load_triplen(dataset):
    return load_looped(dataset)
