"""Triple-N (NNN): macaque EVC+IT spiking to the NSD shared-1000, pre-averaged table loader.

Triple-N (Li et al., ScienceDB 10.57760/sciencedb.33556). Each session's Processed_*.mat
holds ``response_best`` [units x 1072] (mean firing rate per image; cols 1..1000 are the NSD
images, 1001..1072 are face/body/object localizer extras), plus per-unit ``reliability_best``,
``pos`` and ``snr``. Like TVSD this is a PRE-AVERAGED table read directly here (loader="nnn"),
NOT via NeuralSet: Y = reliable units pooled across all 90 sessions; X = the 1000 NSD images
straight from img_pool.mat; ``reward_roi`` picks a cortical area as a column subset of the units
(the readout then fits & scores exactly that area), the analog of TVSD's V1/V4/IT and NSD's
region menu. ALL raw .mat/.xls I/O for this dataset lives in this module.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from harness.datasets import Dataset
from harness.loaders.shared import RawPixels, _cache_dir, drop_constant_cols_with_rois, processed_dir

_NNN_N_NSD_IMAGES = 1000


def _nnn_min_reliability() -> float:
    """Per-unit reliability cut for a usable target (env NNN_MIN_RELIABILITY).

    The release's own good-unit threshold is 0.4 (split-half reliability across the
    repeated presentations); units below it are noise and only dilute the mean-r.
    """
    return float(os.environ.get("NNN_MIN_RELIABILITY", "0.4"))


def _nnn_area_table(root: str) -> dict:
    """exclude_area.xls -> {session: [(y1, y2, AREALABEL, Area), ...]}.

    Each row is one shank y-range of one session annotated with a functional patch
    label (MB/AB/MF/AF/MO/AO/LPP/PITP/CLC/AMC/PF/V1/V2/V4/Unknown) and the coarse
    stream Area (EVC | IT). A unit's area is the row whose [y1, y2) contains its ``pos``.
    """
    import pandas as pd

    df = pd.read_excel(os.path.join(root, "exclude_area.xls"))
    table: dict = {}
    for _, r in df.iterrows():
        table.setdefault(int(r["SesIdx"]), []).append(
            (float(r["y1"]), float(r["y2"]), str(r["AREALABEL"]), str(r["Area"])))
    return table


def _nnn_unit_area(table: dict, ses: int, pos: float) -> tuple[str, str]:
    for y1, y2, label, area in table.get(ses, []):
        if y1 <= pos < y2:
            return label, area
    return "Unknown", "Unknown"


def _nnn_build_X(d: Dataset, img_pool) -> np.ndarray:
    """RawPixels features of the 1000 NSD images, read straight from img_pool.mat."""
    from PIL import Image

    pix = RawPixels(imsize=d.imsize)
    feats = []
    for im in img_pool[:_NNN_N_NSD_IMAGES]:
        a = np.asarray(im)
        if a.ndim == 2:                       # the few grayscale entries -> RGB
            a = np.stack([a] * 3, -1)
        a = np.asarray(Image.fromarray(a.astype("uint8")).resize((d.imsize, d.imsize)))
        feats.append(pix._get_image_features(a))
    return np.stack(feats).astype("float32")  # [1000, imsize*imsize*3]


def _load_nnn_processed(d: Dataset, proc: str):
    """Read X (native imgs resized at load), Y (rel-filtered), areas from data/nnn/."""
    imgs = np.load(os.path.join(proc, "stimuli", "img_pool.npy"))  # [1000, H, W, 3] uint8
    X = _nnn_build_X(d, imgs)                                       # resize to d.imsize at load
    blob = torch.load(os.path.join(proc, "neural", "all_units.pt"), weights_only=False)
    keep = np.asarray(blob["reliability"]).ravel() > _nnn_min_reliability()
    Y = np.asarray(blob["Y"])[:, keep].astype("float32")
    areas = np.asarray(blob["areas"])[keep]
    return X, Y, areas


def load_nnn(d: Dataset):
    """Triple-N: Y from the per-session Processed tables, X from the NSD images, reward = ALL
    reliable units (EVC+IT pooled). Returns (X_train, X_test, Y_train, Y_test, roi_masks); ``roi_masks``
    breaks the score down by stream area (EVC/IT). Y pools every reliable unit
    (``reliability_best`` > NNN_MIN_RELIABILITY) across all 90 sessions; X is the 1000 NSD images,
    split deterministically (every ``round(1/test_frac)``-th held out). Cached by (reliability, imsize).
    """
    import glob
    import re
    import scipy.io as sio

    root, min_rel = d.path, _nnn_min_reliability()
    use_cache = not os.environ.get("NEURO_NO_CACHE")

    # Prefer the in-repo processed tree when present (no raw .mat/.xls needed).
    proc = processed_dir("nnn")
    if os.path.isdir(proc):
        X, Y, areas = _load_nnn_processed(d, proc)
        roi_masks = {a: areas == a for a in ("EVC", "IT")}
        n_src = "data/nnn"
    else:
        X, Y, areas, n_src = _load_nnn_raw(d, root, min_rel, use_cache)
        roi_masks = {a: areas == a for a in ("EVC", "IT")}

    # Deterministic strided split of the 1000 images (every round(1/test_frac)-th -> test).
    step = max(2, int(round(1.0 / d.test_frac)))
    is_test = (np.arange(X.shape[0]) % step) == 0
    tr = torch.as_tensor(np.flatnonzero(~is_test), dtype=torch.long)
    te = torch.as_tensor(np.flatnonzero(is_test), dtype=torch.long)

    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    Xtr, Xte = Xt[tr], Xt[te]
    Ytr, Yte = Yt[tr], Yt[te]
    Ytr, Yte, roi_masks = drop_constant_cols_with_rois(Ytr, Yte, roi_masks)
    print(f"  NNN[{n_src}]: region=all (reliability>{min_rel}) -> "
          f"{Ytr.shape[1]} units; train/test = {Xtr.shape[0]}/{Xte.shape[0]} NSD images "
          f"({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte, roi_masks


def _load_nnn_raw(d: Dataset, root: str, min_rel: float, use_cache: bool):
    """The original raw path: pool reliable units across the 90 Processed_ses*.mat."""
    import glob
    import re
    import scipy.io as sio

    # X: the 1000 NSD images (cached by imsize).
    x_cache = os.path.join(_cache_dir(), f"nnn_X_im{d.imsize}.pt")
    if use_cache and os.path.exists(x_cache):
        X = torch.load(x_cache, weights_only=False)["X"]
    else:
        img_pool = sio.loadmat(os.path.join(root, "img_pool.mat"), simplify_cells=True)["img_pool"]
        X = _nnn_build_X(d, img_pool)
        if use_cache:
            os.makedirs(_cache_dir(), exist_ok=True)
            torch.save({"X": X}, x_cache)

    # Y: every reliable unit pooled across sessions, each tagged with its stream area (cached by rel/imsize).
    y_cache = os.path.join(_cache_dir(), f"nnn_Y_all_rel{min_rel}.pt")
    sessions = sorted(glob.glob(os.path.join(root, "Processed_ses*.mat")))
    if use_cache and os.path.exists(y_cache):
        blob = torch.load(y_cache, weights_only=False)
        Y, areas = blob["Y"], blob["areas"]
    else:
        table = _nnn_area_table(root)
        cols, areas = [], []
        for f in sessions:
            ses = int(re.search(r"ses(\d+)", os.path.basename(f)).group(1))
            m = sio.loadmat(f, simplify_cells=True)
            rb = np.asarray(m["response_best"], dtype="float32")[:, :_NNN_N_NSD_IMAGES]
            keep = np.asarray(m["reliability_best"]).ravel() > min_rel
            pos = np.asarray(m["pos"]).ravel()
            for i in np.flatnonzero(keep):
                areas.append(_nnn_unit_area(table, ses, float(pos[i]))[1] or "?")
            cols.append(rb[keep])                     # [kept_units, 1000]
        if not cols:
            raise RuntimeError(f"NNN: no units passed reliability>{min_rel}")
        Y = np.concatenate(cols, axis=0).T            # [1000 images, total_units]
        areas = np.asarray(areas)
        if use_cache:
            os.makedirs(_cache_dir(), exist_ok=True)
            torch.save({"Y": Y, "areas": areas}, y_cache)
    return X, Y, areas, f"{len(sessions)} sessions"
