"""AlexNet reference: ImageNet-trained and untrained (random-init) layers, each pooled to < 10,000 features."""
import functools
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.filters import chunked, images

WEIGHTS = Path(os.environ.get("NEURO_ALEXNET_WEIGHTS",
                              Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth")).expanduser()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# (layer, module group, module index, pooled spatial size) -> features < 10,000
LAYERS = [("conv1", "features", 1, 12), ("conv2", "features", 4, 7), ("conv3", "features", 7, 5),
          ("conv4", "features", 9, 6), ("conv5", "features", 11, 6),
          ("fc6", "classifier", 2, None), ("fc7", "classifier", 5, None), ("fc8", "classifier", 6, None)]


@functools.cache
def network(trained):
    """torchvision's AlexNet architecture, built once per weight type (ImageNet or seed-0 init)."""
    if trained and not Path(WEIGHTS).is_file():
        raise FileNotFoundError(
            f"AlexNet checkpoint missing: {WEIGHTS}. Set NEURO_ALEXNET_WEIGHTS to the "
            "ImageNet alexnet-owt-7be5be79.pth checkpoint, or select an untrained baseline.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = alexnet()
    if trained:
        net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return net.eval()


def alexnet():
    net = nn.Module()
    net.features = nn.Sequential(
        nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2))
    net.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    net.classifier = nn.Sequential(
        nn.Dropout(), nn.Linear(256 * 36, 4096), nn.ReLU(),
        nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000))
    return net


class AlexNetLayer:
    def __init__(self, layer, group, index, pool, trained):
        kind = "ImageNet-trained" if trained else "untrained (seed-0 random init)"
        self.name = f"alexnet_{layer}" if trained else f"alexnet_random_{layer}"
        self.description = f"{kind} AlexNet {layer}, 224px input" + (f", pooled to {pool}x{pool}." if pool else ".")
        self.group, self.index, self.pool, self.trained = group, index, pool, trained

    @property
    def n_params(self):
        return sum(p.numel() for p in network(self.trained).parameters())

    @torch.no_grad()
    def __call__(self, X):
        net = network(self.trained).to(X.device)
        mean, std = MEAN.to(X.device), STD.to(X.device)

        def forward(chunk):
            z = F.interpolate(images(chunk), size=224, mode="bilinear", align_corners=False)
            z = (z - mean) / std
            if self.group == "features":
                z = net.features[:self.index + 1](z)
                return F.adaptive_avg_pool2d(z, self.pool).flatten(1)
            z = net.avgpool(net.features(z)).flatten(1)
            return net.classifier[:self.index + 1](z)

        return chunked(forward, X)


MODELS = {m.name: m for trained in (True, False)
          for m in (AlexNetLayer(*layer, trained) for layer in LAYERS)}
