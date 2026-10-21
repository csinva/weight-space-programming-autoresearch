"""Compatibility entry point for the vendored looped-networks nnn reader."""
from harness.loaders.vision.looped import load_looped


def load_nnn(dataset):
    return load_looped(dataset)
