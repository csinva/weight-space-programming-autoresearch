#!/usr/bin/env python3
"""Minimal proof-of-concept: train the NSD search CHAMPION on ImageNet.

The champion (`runs/nsd__2026-06-27/models/qq145.py`) is a hand-written, never-trained HMAX-style
vision model. Its only free weights are the random He conv layers `W2`/`W3` (the "ventral" branch);
the Gabor V1 front-end stays fixed. Here we make `W2`/`W3` trainable, bolt a linear 1000-way head on
the champion's representation, and train it on ImageNet with plain backprop for a few epochs.

    IMAGENET_DATA_ROOT=/datadisk/imagenet/ilsvrc2012/data \
        python scripts/experiments/train_champion_imagenet.py --epochs 5

Nothing under harness/ or the champion file is modified (the champion source is loaded and its
`@torch.no_grad()` guards stripped in-memory so gradients can flow).
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAMPION = os.path.join(REPO, "runs", "nsd__2026-06-27", "models", "qq145.py")
IMSIZE = 128   # train at NSD's imsize so the trained model is directly re-scorable on NSD later


def load_trainable_champion(device):
    """Load the champion, strip its no_grad guards, move to device, make W2/W3 trainable Parameters."""
    import types
    src = open(CHAMPION).read().replace("@torch.no_grad()", "")   # let gradients flow
    mod = types.ModuleType("champion_trainable")
    exec(compile(src, CHAMPION, "exec"), mod.__dict__)
    champ = mod.Model()
    champ._to(device)                       # move all buffers/weights (plain tensors) to device, set _dev
    champ.W2 = nn.Parameter(champ.W2)       # the two random conv layers become the trained weights
    champ.W3 = nn.Parameter(champ.W3)       # (_dev == device now, so _to() never reassigns them)
    return champ


class TrainableChampion(nn.Module):
    """Champion feature extractor (W2/W3 trainable) + a linear ImageNet head."""

    def __init__(self, device, n_classes=1000):
        super().__init__()
        self.champ = load_trainable_champion(device)
        self.trained = nn.ParameterList([self.champ.W2, self.champ.W3])   # register for the optimizer
        with torch.no_grad():
            d_out = self.champ(torch.zeros(2, IMSIZE * IMSIZE * 3, device=device)).shape[1]
        self.head = nn.Sequential(nn.LayerNorm(d_out), nn.Linear(d_out, n_classes)).to(device)
        print(f"  champion D_out={d_out}  trainable params="
              f"{sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x_flat):
        return self.head(self.champ(x_flat))


class ParquetImageNet(torch.utils.data.IterableDataset):
    """Streams (flat HWC [0,1] pixels, label) from HuggingFace ImageNet parquet shards."""

    def __init__(self, shards):
        self.shards = shards

    def __iter__(self):
        import pyarrow.parquet as pq
        info = torch.utils.data.get_worker_info()
        shards = self.shards[info.id::info.num_workers] if info else self.shards
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["image", "label"]):
                for row in batch.to_pylist():
                    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").resize((IMSIZE, IMSIZE))
                    x = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0   # HWC flat, [0,1]
                    yield torch.from_numpy(x), int(row["label"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "champion_imagenet.pt"))
    a = p.parse_args()

    root = os.environ.get("IMAGENET_DATA_ROOT", "/datadisk/imagenet/ilsvrc2012/data")
    shards = sorted(glob.glob(os.path.join(root, "train-*.parquet")))
    if not shards:
        raise SystemExit(f"no ImageNet train shards under {root!r} (set IMAGENET_DATA_ROOT)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== train champion on ImageNet | {len(shards)} shards | {a.epochs} epochs | device={device} ===")

    model = TrainableChampion(device)
    loader = torch.utils.data.DataLoader(ParquetImageNet(shards), batch_size=a.batch,
                                         num_workers=a.workers, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    for epoch in range(a.epochs):
        model.train()
        seen = correct = 0
        run_loss = 0.0
        t0 = time.time()
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)                       # fp32 — the champion's sqrt/pow energy ops overflow in fp16
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):            # a rare image blows up one of the champion's
                continue                            # nonlinear branches; skip it so it can't poison W2/W3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
            if step % 50 == 0:
                print(f"  e{epoch} step {step:>5}  loss={run_loss/(step+1):.3f}  "
                      f"train_acc={correct/max(seen,1):.3f}  ({seen} imgs, {time.time()-t0:.0f}s)")
        print(f"[epoch {epoch}] loss={run_loss/(step+1):.3f}  train_acc={correct/max(seen,1):.3f}  "
              f"{seen} imgs in {time.time()-t0:.0f}s")
        torch.save({"W2": model.champ.W2.detach().cpu(), "W3": model.champ.W3.detach().cpu(),
                    "head": model.head.state_dict(), "epoch": epoch}, a.out)
        print(f"  saved -> {a.out}")


if __name__ == "__main__":
    main()
