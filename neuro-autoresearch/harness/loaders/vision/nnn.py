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
from harness.loaders.shared import (
    RawPixels, _cache_dir, drop_constant_cols_with_rois, processed_dir, surviving_cols)

_NNN_N_NSD_IMAGES = 1000

# Absolute NC_r2 floor (NC_r ~0.1): a unit with near-zero explainable variance would blow up the
# r^2/NC_r2 reward, so it is dropped even if NNN_MIN_RELIABILITY is lowered. Mirrors nsd's _NC_R2_FLOOR.
_NNN_NC_R2_FLOOR = 0.01


def _nnn_min_reliability() -> float:
    """Per-unit reliability cut for a usable target (env NNN_MIN_RELIABILITY).

    The release's own good-unit threshold is 0.4 (split-half reliability across the
    repeated presentations); units below it are noise and only dilute the mean-r.
    """
    return float(os.environ.get("NNN_MIN_RELIABILITY", "0.4"))


def _nnn_unit_types() -> set[int] | None:
    """BombCell spike-isolation classes to keep (env NNN_UNIT_TYPE; unset = all).

    Every unit is tagged 1 = single unit, 2 = MUA, 3 = non-somatic. ``NNN_UNIT_TYPE=1``
    restricts the reward to single units (a robustness check on the ~47k MUA-dominated
    pool); a comma list (``1,2``) keeps several classes.
    """
    raw = os.environ.get("NNN_UNIT_TYPE", "").strip()
    types = set()
    for t in (x.strip() for x in raw.split(",") if x.strip()):
        if t not in ("1", "2", "3"):
            raise ValueError(f"NNN_UNIT_TYPE={raw!r}: unknown class {t!r} (1=single, 2=MUA, 3=non-somatic).")
        types.add(int(t))
    return types or None


_NNN_MONKEYS = (1, 2, 3, 4, 5)   # the 5 recording animals (M1..M5), staged from the raw session names


def _nnn_monkey_id(subject: str) -> int | None:
    """Map a ``NEURO_SUBJECT`` token to a monkey id (1..5), or None to pool ALL animals.

    Accepts a bare digit (``"3"``) or the release's ``"M3"`` form; empty/unset -> None (the
    historical behavior: every session pooled into one population). Unknown ids raise rather than
    silently pooling, so a typo can't quietly change the reward."""
    s = str(subject or "").strip().upper().lstrip("M")
    if not s:
        return None
    if not s.isdigit() or int(s) not in _NNN_MONKEYS:
        raise ValueError(f"NNN subject {subject!r} is not a valid monkey; valid ids are "
                         f"{list(_NNN_MONKEYS)} (M1..M5).")
    return int(s)


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
    """(label, area) of the shank range a unit's depth ``pos`` falls in for its session, or
    ``("Unknown", "Unknown")`` if it maps to no annotated range."""
    for y1, y2, label, area in table.get(ses, []):
        if y1 <= pos < y2:
            return label, area
    return "Unknown", "Unknown"


def _nnn_roi_masks(areas) -> dict:
    """Per-area advisory masks over EVERY area actually present, not just EVC/IT.

    Units whose ``pos`` falls outside any annotated shank range are tagged ``"Unknown"`` (~15% of
    reliable units); they are scored like the rest, so a hardcoded ``{EVC, IT}`` breakdown would
    silently drop them. Keying on the areas present keeps the advisory's coverage = the scored set."""
    areas = np.asarray(areas)
    order = [a for a in ("EVC", "IT") if a in areas]
    order += [a for a in np.unique(areas) if a not in order]
    return {a: areas == a for a in order}


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


def _load_nnn_processed(d: Dataset, proc: str, types: set[int] | None):
    """Read X (native imgs resized at load, cached by imsize), Y (rel/type-filtered), areas, nc."""
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    x_cache = os.path.join(_cache_dir(), f"nnn_X_im{d.imsize}.pt")   # same 1000 NSD images as raw
    if use_cache and os.path.exists(x_cache):
        X = torch.load(x_cache, weights_only=False)["X"]
    else:
        imgs = np.load(os.path.join(proc, "stimuli", "img_pool.npy"))  # [1000, H, W, 3] uint8
        X = _nnn_build_X(d, imgs)                                       # resize to d.imsize at load
        if use_cache:
            os.makedirs(_cache_dir(), exist_ok=True)
            torch.save({"X": X}, x_cache)
    blob = torch.load(os.path.join(proc, "neural", "all_units.pt"), weights_only=False)
    rel = np.asarray(blob["reliability"]).ravel()
    keep = (rel > _nnn_min_reliability()) & (rel ** 2 >= _NNN_NC_R2_FLOOR)
    mk = _nnn_monkey_id(d.subject)
    if mk is not None:                        # subset to ONE recording animal (multi-monkey reward
        if "monkey" not in blob:              # loops this per subject and averages -- see reward.py)
            raise RuntimeError(
                "NEURO_SUBJECT selects a monkey but this all_units.pt has no 'monkey' field -- "
                "(re)run `python -m scripts.process_dataset.nnn` to add it.")
        keep &= (np.asarray(blob["monkey"]).ravel() == mk)
    if types is not None:
        if "unit_type" not in blob:
            raise RuntimeError(
                "NNN_UNIT_TYPE is set but this all_units.pt has no 'unit_type' -- (re)run "
                "`python -m scripts.process_dataset.nnn` to add it.")
        keep &= np.isin(np.asarray(blob["unit_type"]).ravel(), list(types))
    Y = np.asarray(blob["Y"])[:, keep].astype("float32")
    areas = np.asarray(blob["areas"])[keep]
    # reliability is the per-unit split-half NC_r; its square is the R^2 ceiling the reward
    # normalizes by (Algonauts r^2/NC_r2), matching NSD's per-voxel ncsnr path.
    nc = (rel[keep] ** 2).astype("float32")
    return X, Y, areas, nc


def load_nnn(d: Dataset):
    """Triple-N: Y from the per-session Processed tables, X from the NSD images, reward = ALL
    reliable units (EVC+IT pooled). Returns (X_train, X_test, Y_train, Y_test, roi_masks, nc);
    ``roi_masks`` breaks the score down by stream area (EVC/IT), ``nc`` is the per-unit R^2 noise
    ceiling (reliability_best^2) the reward normalizes by. Y pools every reliable unit
    (``reliability_best`` > NNN_MIN_RELIABILITY) across all 90 sessions; X is the 1000 NSD images,
    split deterministically (every ``round(1/test_frac)``-th held out). Cached by (reliability, imsize).
    """
    min_rel = _nnn_min_reliability()

    proc = processed_dir("nnn")
    if not os.path.isdir(proc):
        raise FileNotFoundError(
            f"NNN not staged in-repo at {proc!r}. data/nnn/ is the only source -- "
            f"run `python -m scripts.process_dataset.nnn` to build it.")
    types = _nnn_unit_types()
    X, Y, areas, nc = _load_nnn_processed(d, proc, types)
    n_src = "data/nnn"
    roi_masks = _nnn_roi_masks(areas)

    if Y.shape[1] == 0:
        raise RuntimeError(
            f"NNN[{n_src}]: no units passed reliability>{min_rel} "
            f"(lower NNN_MIN_RELIABILITY).")

    # Deterministic strided split of the 1000 images (every round(1/test_frac)-th -> test).
    step = max(2, int(round(1.0 / d.test_frac)))
    is_test = (np.arange(X.shape[0]) % step) == 0
    tr = torch.as_tensor(np.flatnonzero(~is_test), dtype=torch.long)
    te = torch.as_tensor(np.flatnonzero(is_test), dtype=torch.long)

    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    Xtr, Xte = Xt[tr], Xt[te]
    Ytr, Yte = Yt[tr], Yt[te]
    keep = surviving_cols(Ytr)                          # one train-constant scan, shared with nc below
    nc = nc[keep.numpy()]
    Ytr, Yte, roi_masks = drop_constant_cols_with_rois(Ytr, Yte, roi_masks, keep=keep)
    sel = "all-classes" if types is None else "UnitType=" + ",".join(map(str, sorted(types)))
    mk = _nnn_monkey_id(d.subject)
    who = "all-monkeys" if mk is None else f"M{mk}"
    print(f"  NNN[{n_src}]: region=all ({who}, {sel}, reliability>{min_rel}, NC-normalized) -> "
          f"{Ytr.shape[1]} units; train/test = {Xtr.shape[0]}/{Xte.shape[0]} NSD images "
          f"({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte, roi_masks, nc
