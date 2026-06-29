"""Bake the `text` (Le Petit Prince fMRI) timeline into data/text/<study>.pt (in-repo runtime source).

`text` is the one NeuralSet timeline whose fMRI volumes don't reduce to a re-segmentable continuous
.fif the way meg/ieeg do, so instead of a slim tree we bake the FINAL (X, Y, onsets, stim_ids): run
the raw NeuralSet study once via the loader's `_build_xy` (with the preset's fixed stimulus/neural
windows) and save the blob. Runtime then reads ONLY data/text/; the OpenNeuro dirs below are the only
raw reader. The window/stimulus params are fixed at build time -- re-run this if you change them.

Run (on the box where the raw data lives):
  python -m scripts.process_dataset.text                  # default subject EN057
  python -m scripts.process_dataset.text --subject EN058  # a per-subject sibling
"""
from __future__ import annotations

import argparse
import os

import torch
import neuralset as ns

from harness.datasets import PRESETS, with_subject
from harness.loaders import studies  # noqa: F401  (registers Li2022PetitSubj*)
from harness.loaders.timeline import _build_xy
from harness.loaders.shared import processed_dir

# Raw OpenNeuro source dirs (the offline build reads here once; runtime reads only data/text/).
# EN057 has its own staged dir; every other English subject lives under li2022petit_en_all.
RAW_EN057 = "/home/t-ymehta/Research/neuralset/data/li2022petit_en057"
RAW_ALL = "/home/t-ymehta/Research/neuralset/data/li2022petit_en_all"


def _raw_path(subject: str) -> str:
    return RAW_EN057 if subject in ("", "EN057") else RAW_ALL


def build(subject: str) -> None:
    d = PRESETS["text"]
    if subject:
        d = with_subject(d, subject)
    study = ns.Study(name=d.study, path=_raw_path(subject))
    events = study.run()
    X, Y, onsets, stim_ids = _build_xy(d, events)

    out = processed_dir("text")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{d.study}.pt")
    torch.save({"X": X, "Y": Y, "onsets": onsets, "stim_ids": stim_ids}, path, pickle_protocol=4)
    print(f"[text] {d.study}: wrote X {tuple(X.shape)} / Y {tuple(Y.shape)} "
          f"({X.shape[0]} word-segments) -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="",
                    help="English subject id (e.g. EN058); default (empty) builds EN057")
    build(ap.parse_args().subject)
