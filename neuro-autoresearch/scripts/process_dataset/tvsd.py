"""Process TVSD into the in-repo data/tvsd/ tree (both monkeys F and N).

Removes the gated 12GB THINGS dependency + the 200MB .mat on the target machine:
  data/tvsd/stimuli/monkey{F,N}/{train,test}/<idx>.jpg  -- native-resolution THINGS images
      this monkey uses (raw JPEG bytes, NOT resized: imsize stays a load-time knob).
  data/tvsd/neural/monkey{F,N}.pt -- train_MUA, test_MUA, area labels.

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
from harness.loaders.standalone.tvsd import (
    _read_mat, _tvsd_area_labels, _tvsd_image_relpaths, _tvsd_images_root, _tvsd_targets,
)

RAW = "/home/t-ymehta/Research/neuralset/data/tvsd_normmua"


def process_monkey(subject: str, n: int | None):
    mdir = os.path.join(RAW, f"monkey{subject}")
    out = processed_dir("tvsd")
    things = _tvsd_images_root()

    # neural: train/test MUA + area labels (small).
    mat = _read_mat(os.path.join(mdir, "THINGS_normMUA.mat"))
    Y_train = torch.from_numpy(_tvsd_targets(mat["train_MUA"]))
    Y_test = torch.from_numpy(_tvsd_targets(mat["test_MUA"]))
    labels = _tvsd_area_labels(subject)
    ndir = os.path.join(out, "neural")
    os.makedirs(ndir, exist_ok=True)
    torch.save({"train_MUA": Y_train, "test_MUA": Y_test, "areas": labels},
               os.path.join(ndir, f"monkey{subject}.pt"))
    print(f"  monkey{subject} neural: train {tuple(Y_train.shape)} test {tuple(Y_test.shape)}")

    # stimuli: copy the native THINGS JPEGs this monkey uses, keyed by split+index.
    imgs_mat = os.path.join(mdir, "things_imgs.mat")
    for split in ("train", "test"):
        rels = _tvsd_image_relpaths(imgs_mat, f"{split}_imgs")
        if n is not None:
            rels = rels[:n]
        sdir = os.path.join(out, "stimuli", f"monkey{subject}", split)
        os.makedirs(sdir, exist_ok=True)
        for i, r in enumerate(rels):
            shutil.copyfile(os.path.join(things, r), os.path.join(sdir, f"{i}.jpg"))
        print(f"  monkey{subject} stimuli/{split}: {len(rels)} images -> {sdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="only first N images per split (smoke test); default all")
    a = ap.parse_args()
    for subject in ("F", "N"):
        process_monkey(subject, a.n)


if __name__ == "__main__":
    main()
