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
from harness.loaders.shared import RawPixels, _cache_dir, drop_constant_cols

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


def _nnn_area_match(label: str, area: str, roi: str) -> bool:
    """Does a unit's (patch label, stream area) satisfy the requested ``reward_roi``?

    ``all``/``nsdgeneral`` keep everything; ``EVC``/``IT`` match the coarse stream;
    an alpha-only roi (e.g. ``MB``) matches the patch FAMILY (MB1/MB2/MB3); anything
    else is an exact label match (``V1`` matches V1 only, not V12).
    """
    roi_u = roi.upper()
    if roi in ("all", "nsdgeneral"):
        return True
    if roi_u in ("EVC", "IT"):
        return area.upper() == roi_u
    import re as _re

    family = _re.match(r"[A-Za-z]+", label or "")
    if roi.isalpha() and family is not None and family.group(0).upper() == roi_u:
        return True
    return label.upper() == roi_u


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


def load_nnn(d: Dataset):
    """Triple-N: Y from the per-session Processed tables, X from the NSD images, reward = one area.

    Returns (X_train, X_test, Y_train, Y_test) as 2-D float tensors -- the same contract as
    the Segmenter path. Y pools every reliable unit (``reliability_best`` > NNN_MIN_RELIABILITY)
    across all 90 sessions, optionally narrowed by ``reward_roi`` to a cortical area/patch; X is
    the 1000 NSD images. The 1000 images are split deterministically (every ``round(1/test_frac)``-th
    image held out), so train/test are fixed and image-disjoint. The assembled units are cached by
    (reliability, reward_roi, imsize); X by imsize. Bypass with NEURO_NO_CACHE.
    """
    import glob
    import re
    import scipy.io as sio

    root, roi, min_rel = d.path, d.reward_roi, _nnn_min_reliability()
    use_cache = not os.environ.get("NEURO_NO_CACHE")

    # X: the 1000 NSD images (cached by imsize, shared across every reward_roi).
    x_cache = os.path.join(_cache_dir(), f"nnn_X_im{d.imsize}.pt")
    if use_cache and os.path.exists(x_cache):
        X = torch.load(x_cache, weights_only=False)["X"]
    else:
        img_pool = sio.loadmat(os.path.join(root, "img_pool.mat"), simplify_cells=True)["img_pool"]
        X = _nnn_build_X(d, img_pool)
        if use_cache:
            os.makedirs(_cache_dir(), exist_ok=True)
            torch.save({"X": X}, x_cache)

    # Y: reliable units (optionally one area) pooled across sessions (cached by rel/roi/imsize).
    y_cache = os.path.join(_cache_dir(), f"nnn_Y_{roi}_rel{min_rel}.pt")
    sessions = sorted(glob.glob(os.path.join(root, "Processed_ses*.mat")))
    if use_cache and os.path.exists(y_cache):
        Y = torch.load(y_cache, weights_only=False)["Y"]
    else:
        table = _nnn_area_table(root) if roi not in ("all", "nsdgeneral") else None
        cols = []
        for f in sessions:
            ses = int(re.search(r"ses(\d+)", os.path.basename(f)).group(1))
            m = sio.loadmat(f, simplify_cells=True)
            rb = np.asarray(m["response_best"], dtype="float32")[:, :_NNN_N_NSD_IMAGES]
            keep = np.asarray(m["reliability_best"]).ravel() > min_rel
            if table is not None:
                pos = np.asarray(m["pos"]).ravel()
                for i in np.flatnonzero(keep):
                    lab, area = _nnn_unit_area(table, ses, float(pos[i]))
                    if not _nnn_area_match(lab, area, roi):
                        keep[i] = False
            if keep.any():
                cols.append(rb[keep])                 # [kept_units, 1000]
        if not cols:
            raise RuntimeError(
                f"NNN: no units passed reliability>{min_rel} for reward_roi={roi!r}")
        Y = np.concatenate(cols, axis=0).T            # [1000 images, total_units]
        if use_cache:
            os.makedirs(_cache_dir(), exist_ok=True)
            torch.save({"Y": Y}, y_cache)

    # Deterministic strided split of the 1000 images (every round(1/test_frac)-th -> test).
    step = max(2, int(round(1.0 / d.test_frac)))
    is_test = (np.arange(X.shape[0]) % step) == 0
    tr = torch.as_tensor(np.flatnonzero(~is_test), dtype=torch.long)
    te = torch.as_tensor(np.flatnonzero(is_test), dtype=torch.long)

    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    Xtr, Xte = Xt[tr], Xt[te]
    Ytr, Yte = Yt[tr], Yt[te]
    Ytr, Yte = drop_constant_cols(Ytr, Yte)
    print(f"  NNN: {len(sessions)} sessions; reward_roi={roi} (reliability>{min_rel}) -> "
          f"{Ytr.shape[1]} units; train/test = {Xtr.shape[0]}/{Xte.shape[0]} NSD images "
          f"({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte
