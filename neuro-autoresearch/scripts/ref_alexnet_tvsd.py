#!/usr/bin/env python3
"""Trained-AlexNet reference baselines for TVSD, read out with the SAME ridge as the search.

Sweeps AlexNet layers conv1..conv5 + fc6 + fc7 (each conv map spatially pooled to stay under
the ridge SVD feature cap) so a cortical area is compared to AlexNet's BEST-matching layer,
not just the semantic fc7 -- fairer for mid-level areas like V4. Uses the cached ImageNet
checkpoint (offline). Run from the repo root::

    python scripts/ref_alexnet_tvsd.py V4         # sweep all layers for V4
    python scripts/ref_alexnet_tvsd.py IT V4      # both areas
"""
from __future__ import annotations
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

from harness.datasets import PRESETS  # noqa: E402
from harness.data import load_split  # noqa: E402
from harness.reward import evaluate_on_split, _readout_device  # noqa: E402

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_FEAT_CUT = {"conv1": 3, "conv2": 6, "conv3": 8, "conv4": 10, "conv5": 13}  # net.features[:cut]
_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
_CAP = 30000  # keep n_train(22248) * D_out under the cuSOLVER SVD limit (~7.3e8; p=40000 FAILS, p=32768 OK)


def _pool_to_cap(f: torch.Tensor) -> torch.Tensor:
    """Flatten a conv map; if over the feature cap, adaptive-avg-pool spatially to fit."""
    n, c, h, w = f.shape
    if c * h * w <= _CAP:
        return f.flatten(1)
    g = max(1, int(math.isqrt(_CAP // c)))
    return torch.nn.functional.adaptive_avg_pool2d(f, g).flatten(1)


class AlexNetLayer:
    """Frozen ImageNet AlexNet at one layer, as a harness model: X (N, side*side*3) -> (N, D)."""
    n_params = 0
    summary = "frozen pretrained AlexNet, single layer, read out with the harness ridge"

    def __init__(self, net, device, layer: str):
        self.features, self.avgpool = net.features, net.avgpool
        self.fc6 = torch.nn.Sequential(*list(net.classifier.children())[:3])  # -> 4096
        self.fc7 = torch.nn.Sequential(*list(net.classifier.children())[:6])  # -> 4096
        self.device, self.layer = device, layer
        self.name = f"alexnet_{layer}"
        self.idea = f"trained ImageNet AlexNet {layer} reference"
        self.mean, self.std = _MEAN.to(device), _STD.to(device)

    @torch.no_grad()
    def __call__(self, X):
        X = X if torch.is_tensor(X) else torch.as_tensor(np.asarray(X))
        X = X.to(self.device, torch.float32)
        side = int(round((X.shape[1] / 3) ** 0.5))
        out = []
        for i in range(0, X.shape[0], 256):
            b = X[i:i + 256].reshape(-1, side, side, 3).permute(0, 3, 1, 2)
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear", align_corners=False)
            b = (b - self.mean) / self.std
            if self.layer in _FEAT_CUT:
                out.append(_pool_to_cap(self.features[:_FEAT_CUT[self.layer]](b)))
            else:
                f = torch.flatten(self.avgpool(self.features(b)), 1)
                out.append(self.fc6(f) if self.layer == "fc6" else self.fc7(f))
        return torch.cat(out, 0)


def main() -> None:
    rois = sys.argv[1:] or ["V4"]
    dev = _readout_device()
    net = torchvision.models.alexnet(weights="IMAGENET1K_V1").to(dev).eval()
    for roi in rois:
        d = dataclasses.replace(PRESETS["tvsd"], imsize=128, reward_roi=roi)
        split = load_split(d)
        print(f"\nTrained-AlexNet layer sweep on TVSD monkey F / {roi} (128px):")
        best = (None, -1.0)
        for layer in _LAYERS:
            row = evaluate_on_split(AlexNetLayer(net, dev, layer), split, d)
            tc = row["test_corr"]
            print(f"  {layer:6s}: test_corr={tc:.4f}  train={row['train_corr']:.4f}  ({row['seconds']}s)")
            if tc > best[1]:
                best = (layer, tc)
        print(f"  -> best AlexNet layer for {roi}: {best[0]} = {best[1]:.4f}")


if __name__ == "__main__":
    main()
