#!/usr/bin/env python3
"""Trained-AlexNet (fc7) reference baseline for TVSD, read out with the SAME ridge as the
open-ended search, so the untrained champions have a trained bar to compare against.

A frozen ImageNet-pretrained AlexNet truncated to fc7 (4096-d) is wrapped as a harness
"model" (X -> activations) and scored by ``harness.reward.evaluate_on_split`` on the TVSD
split -- identical readout, split, and stimuli to the search. Uses the cached ImageNet
checkpoint (offline). Run from the repo root::

    python scripts/ref_alexnet_tvsd.py IT V4          # default IT V4, monkey F, 128px
"""
from __future__ import annotations
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import numpy as np
import torch
import torchvision

from harness.datasets import PRESETS  # noqa: E402
from harness.data import load_split
from harness.reward import evaluate_on_split, _readout_device

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class AlexNetFc7:
    """Frozen ImageNet AlexNet fc7 (4096-d) as a harness model: X (N, side*side*3) -> (N, 4096)."""
    name = "alexnet_fc7"
    idea = "trained ImageNet AlexNet (fc7, 4096-d) reference baseline"
    summary = "frozen pretrained AlexNet read out with the harness ridge"
    n_params = 0

    def __init__(self, device):
        net = torchvision.models.alexnet(weights="IMAGENET1K_V1").to(device).eval()
        self.features, self.avgpool = net.features, net.avgpool
        self.fc7 = torch.nn.Sequential(*list(net.classifier.children())[:6])  # up to the 4096-d ReLU
        self.device = device
        self.mean, self.std = _MEAN.to(device), _STD.to(device)

    @torch.no_grad()
    def __call__(self, X):
        X = X if torch.is_tensor(X) else torch.as_tensor(np.asarray(X))
        X = X.to(self.device, torch.float32)
        side = int(round((X.shape[1] / 3) ** 0.5))
        out = []
        for i in range(0, X.shape[0], 256):
            b = X[i:i + 256].reshape(-1, side, side, 3).permute(0, 3, 1, 2)          # NCHW, [0,1]
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear", align_corners=False)
            b = (b - self.mean) / self.std                                            # ImageNet norm
            f = torch.flatten(self.avgpool(self.features(b)), 1)
            out.append(self.fc7(f))
        return torch.cat(out, 0)


def main() -> None:
    rois = sys.argv[1:] or ["IT", "V4"]
    dev = _readout_device()
    model = AlexNetFc7(dev)
    print(f"Trained-AlexNet fc7 reference on TVSD monkey F (128px), device={dev}:")
    for roi in rois:
        d = dataclasses.replace(PRESETS["tvsd"], imsize=128, reward_roi=roi)
        row = evaluate_on_split(model, load_split(d), d)
        print(f"  tvsd/{roi:3s}: test_corr={row['test_corr']:.4f}  train={row['train_corr']:.4f}  "
              f"electrodes={row['n_targets']}  ({row['seconds']}s)")


if __name__ == "__main__":
    main()
