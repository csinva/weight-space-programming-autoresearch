"""Process TVSD into the in-repo data/tvsd/ tree (both monkeys F and N).

Removes the gated 12GB THINGS dependency + the 200MB .mat on the target machine. Stimuli are
SHARED across the two monkeys (identical images, identical order), so they are staged ONCE:
  data/tvsd/stimuli/{train,test}/<idx>.jpg  -- native-resolution THINGS images (raw JPEG bytes,
      NOT resized: imsize stays a load-time knob). One copy, used by both monkeys.
  data/tvsd/neural/monkey{F,N}.pt -- train_MUA, test_MUA, area labels, oracle (per monkey).

Run (full, both monkeys):  python -m scripts.process_dataset.tvsd
Sample (a few images each):  python -m scripts.process_dataset.tvsd --n 5
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import torch

from harness.loaders.shared import processed_dir
from harness.loaders.vision.tvsd import (
    _read_mat, _tvsd_area_labels, _tvsd_image_relpaths, _tvsd_images_root, _tvsd_oracle,
    _tvsd_targets,
)

RAW = "/home/t-ymehta/Research/neuralset/data/tvsd_normmua"


def process_neural(subject: str, n: int | None):
    """Stage one monkey's neural targets (per-monkey: MUA + area labels + oracle)."""
    mdir = os.path.join(RAW, f"monkey{subject}")
    out = processed_dir("tvsd")
    mat = _read_mat(os.path.join(mdir, "THINGS_normMUA.mat"))
    Y_train = torch.from_numpy(_tvsd_targets(mat["train_MUA"]))
    Y_test = torch.from_numpy(_tvsd_targets(mat["test_MUA"]))
    oracle = torch.from_numpy(_tvsd_oracle(mat["oracle"]))   # per-electrode noise ceiling
    if n is not None:  # smoke subset: keep neural rows aligned to the shared stimuli below.
        Y_train, Y_test = Y_train[:n], Y_test[:n]
    labels = _tvsd_area_labels(subject)
    ndir = os.path.join(out, "neural")
    os.makedirs(ndir, exist_ok=True)
    torch.save({"train_MUA": Y_train, "test_MUA": Y_test, "areas": labels, "oracle": oracle},
               os.path.join(ndir, f"monkey{subject}.pt"))
    print(f"  monkey{subject} neural: train {tuple(Y_train.shape)} test {tuple(Y_test.shape)} "
          f"oracle med {float(oracle.median()):.3f}")


def process_stimuli(n: int | None):
    """Stage the THINGS JPEGs ONCE (shared by both monkeys -- identical images, identical order)."""
    out = processed_dir("tvsd")
    things = _tvsd_images_root()
    imgs_mat = os.path.join(RAW, "things_imgs.mat")          # shared copy at the dataset root
    if not os.path.exists(imgs_mat):
        imgs_mat = os.path.join(RAW, "monkeyF", "things_imgs.mat")  # pre-dedup fallback
    for split in ("train", "test"):
        rels = _tvsd_image_relpaths(imgs_mat, f"{split}_imgs")
        if n is not None:
            rels = rels[:n]
        sdir = os.path.join(out, "stimuli", split)
        os.makedirs(sdir, exist_ok=True)
        for i, r in enumerate(rels):
            shutil.copyfile(os.path.join(things, r), os.path.join(sdir, f"{i}.jpg"))
        print(f"  stimuli/{split}: {len(rels)} images -> {sdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="only first N images per split (smoke test); default all")
    a = ap.parse_args()
    for subject in ("F", "N"):
        process_neural(subject, a.n)
    process_stimuli(a.n)                                     # once, shared


if __name__ == "__main__":
    main()
