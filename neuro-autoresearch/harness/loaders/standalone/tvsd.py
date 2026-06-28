"""TVSD: THINGS Ventral-stream Spiking Dataset (macaque), pre-averaged table loader.

TVSD ships a normalized-MUA table already aligned to its THINGS stimuli, so it bypasses the
timeline/Segmenter machinery entirely: Y is read straight from the .mat and X is built from
the THINGS images each column points at. ALL raw file I/O for this dataset (.mat + image
reads) is confined to this module -- one of the two places in the fixed harness that reads
outside NeuralSet, because NeuralSet has no TVSD study to express a pre-averaged table.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from harness.datasets import Dataset
from harness.loaders.shared import (
    RawPixels, _cache_dir, drop_constant_cols_with_rois, processed_dir, surviving_cols,
)

_TVSD_N_ELEC = 1024
# Absolute R^2-ceiling floor enforced even when TVSD_ORACLE_MIN=0: an electrode whose oracle noise
# ceiling is ~0 has essentially no explainable variance, so normalizing r^2/oracle^2 by it would blow
# up the reward. oracle^2 < this (~oracle < 0.1) is dropped regardless of the reliability knob.
_TVSD_NC_FLOOR = 0.01
# Per-monkey electrode -> cortical area, hardcoded from the release's _code/norm_MUA.m (the
# labels are NOT stored in the .mat). Tuples are (area, first_channel, last_channel), 1-based.
_TVSD_AREAS = {
    "F": (("V1", 1, 512), ("IT", 513, 832), ("V4", 833, 1024)),
    "N": (("V1", 1, 512), ("V4", 513, 768), ("IT", 769, 1024)),
}
_TVSD_REGIONS = ("V1", "V4", "IT")


def _tvsd_area_labels(subject: str) -> np.ndarray:
    """Length-1024 array of 'V1'/'V4'/'IT' per electrode, for the given monkey."""
    labels = np.empty(_TVSD_N_ELEC, dtype=object)
    for area, lo, hi in _TVSD_AREAS[subject]:
        labels[lo - 1:hi] = area
    return labels


def _read_mat(path: str) -> dict:
    """Load a .mat as a dict, handling both v5/v7 (scipy) and v7.3/HDF5 (h5py).

    TVSD ships monkey F as v5 and monkey N as v7.3, so both readers are needed. h5py reads
    HDF5 with axes reversed vs MATLAB; callers orient by the electrode axis (``_tvsd_targets``).
    """
    import scipy.io
    try:
        return scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(v) for k, v in f.items() if not k.startswith("#")}


def _tvsd_targets(a) -> np.ndarray:
    """A stored MUA matrix -> [stimuli x electrodes] float32 (electrode axis last)."""
    a = np.asarray(a, dtype=np.float32)
    return a if a.shape[1] == _TVSD_N_ELEC else a.T


def _tvsd_oracle(a) -> np.ndarray:
    """The stored per-electrode oracle noise ceiling -> length-1024 float32.

    ``oracle`` is the dataset authors' own per-electrode noise ceiling (the split-half/repeat
    consistency correlation the TVSD release publishes); it is the TVSD analog of NSD's per-voxel
    ``ncsnr``. Stored as a plain 1-D vector (orientation-trivial), so unlike the 2-D/3-D MUA
    matrices it needs no electrode-axis disambiguation."""
    o = np.asarray(a, dtype=np.float32).reshape(-1)
    if o.size != _TVSD_N_ELEC:
        raise ValueError(f"TVSD oracle has {o.size} entries, expected {_TVSD_N_ELEC} electrodes")
    return o


def _oracle_select(oracle, Ytr, Yte, roi_masks):
    """TVSD reliability cut + noise ceiling, mirroring NSD's ``_ncsnr_select``.

    ``oracle`` is the per-electrode noise-ceiling CORRELATION; its square is the R^2 ceiling the
    reward normalizes by (Algonauts r^2/NC_r2, here r^2/oracle^2 = the squared oracle-normalized
    predictivity, the TVSD-paper metric). Electrodes with ``oracle < TVSD_ORACLE_MIN`` (operator
    knob; default 0.2, mirroring NSD_NCSNR_MIN; set 0 to keep all reliable electrodes) are dropped
    from Y AND every roi_mask, so each breakdown scores only its signal electrodes. An absolute
    ``_TVSD_NC_FLOOR`` is enforced regardless of the knob so a ~zero-ceiling electrode never reaches
    the reward's r^2/oracle^2 divide. Returns ``(Ytr, Yte, roi_masks, nc)`` with ``nc`` (= oracle^2)
    aligned to the surviving electrode columns."""
    thresh = float(os.environ.get("TVSD_ORACLE_MIN", "0.2"))
    oracle = _tvsd_oracle(oracle)
    nc = np.clip(oracle, 0.0, None) ** 2                      # R^2 ceiling (anti-reliable -> 0)
    keep = oracle >= thresh                                   # all-True only if thresh <= min oracle
    keep &= nc >= _TVSD_NC_FLOOR                              # absolute floor even when thresh<=0
    if not keep.any():
        raise ValueError(f"TVSD_ORACLE_MIN={thresh} filtered out every electrode (no reliable targets)")
    print(f"  TVSD oracle select: oracle>={thresh} keeps {int(keep.sum())}/{keep.size} electrodes")
    keep_t = torch.as_tensor(keep)
    roi_masks = {k: np.asarray(v, dtype=bool)[keep] for k, v in roi_masks.items()}
    return Ytr[:, keep_t], Yte[:, keep_t], roi_masks, nc[keep]


def _tvsd_finalize(Xtr, Xte, Ytr, Yte, roi_masks, oracle, subject, tag):
    """Shared tail for both load paths: oracle reliability cut + noise ceiling, then the
    train-constant-column drop (carrying ``nc`` along), then the row print. Returns the
    ``(X_train, X_test, Y_train, Y_test, roi_masks, nc)`` 6-tuple the harness expects."""
    Ytr, Yte, roi_masks, nc = _oracle_select(oracle, Ytr, Yte, roi_masks)
    keep = surviving_cols(Ytr)                                # one train-constant scan, shared with nc
    nc = nc[keep.numpy() if torch.is_tensor(keep) else np.asarray(keep)]
    Ytr, Yte, roi_masks = drop_constant_cols_with_rois(Ytr, Yte, roi_masks, keep=keep)
    print(f"  TVSD monkey{subject}{tag}: {Xtr.shape[0]} train / {Xte.shape[0]} test images; "
          f"region=all -> {Ytr.shape[1]} reliable electrodes ({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte, roi_masks, nc


def _tvsd_images_root() -> str:
    """Dir holding the THINGS images as ``<concept>/<name>.jpg`` (gated, not bundled)."""
    root = os.environ.get("THINGS_IMAGES") or os.path.expanduser(
        "~/Research/neuralset/data/THINGS-images")
    if not os.path.isdir(root):
        raise RuntimeError(
            f"THINGS images not found at {root!r}. TVSD uses the gated THINGS image set, which "
            f"is NOT bundled with the spiking data. Stage it once (license + ~12GB) -- e.g. "
            f"neuralfetch.utils.download_things_images with NEURALFETCH_THINGS_PASSWORD -- or "
            f"point THINGS_IMAGES at a dir of <concept>/<name>.jpg files.")
    return root


def _tvsd_image_relpaths(things_imgs_mat: str, key: str) -> list[str]:
    """Per-stimulus THINGS paths ('<concept>/<name>.jpg') for train/test, in the MUA
    table's column order (the table is pre-sorted to match ``things_imgs.mat``)."""
    import h5py
    with h5py.File(things_imgs_mat, "r") as f:
        refs = np.asarray(f[key]["things_path"]).reshape(-1)

        def decode(ref):
            codes = np.asarray(f[ref][()]).reshape(-1)
            return "".join(chr(int(c)) for c in codes if int(c) > 0)

        return [decode(r).replace("\\", "/") for r in refs]


def _tvsd_build_X(d: Dataset, subject: str, things_imgs_mat: str):
    """(X_train, X_test) RawPixels features of the THINGS images, cached by (subject, imsize)."""
    cache = os.path.join(_cache_dir(), f"tvsd_X_{subject}_im{d.imsize}.pt")
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    if use_cache and os.path.exists(cache):
        blob = torch.load(cache, weights_only=False)   # cache holds numpy arrays
        return blob["X_train"], blob["X_test"]
    from PIL import Image
    pix, root = RawPixels(imsize=d.imsize), _tvsd_images_root()

    def build(key):
        rels = _tvsd_image_relpaths(things_imgs_mat, key)
        feats = [pix._get_image_features(
                     np.asarray(Image.open(os.path.join(root, r)).convert("RGB")
                                .resize((d.imsize, d.imsize))))
                 for r in rels]
        return np.stack(feats).astype("float32")

    X_train, X_test = build("train_imgs"), build("test_imgs")
    if use_cache:
        os.makedirs(_cache_dir(), exist_ok=True)
        # protocol 4: at imsize 128 X_train is ~4.4 GB, over the 4 GB limit of pickle proto <=3.
        torch.save({"X_train": X_train, "X_test": X_test}, cache, pickle_protocol=4)
    return X_train, X_test


def _tvsd_build_X_processed(d: Dataset, subject: str):
    """(X_train, X_test) RawPixels features from the stored native JPEGs in data/tvsd/.

    Cached by (subject, imsize) -- decoding the 22k+ training JPEGs is the load bottleneck, so the
    built float features are memoized to disk (as the raw-.mat path and NSD do). Subsequent runs and
    every concurrent search-eval process then load one tensor instead of re-decoding every image.
    """
    cache = os.path.join(_cache_dir(), f"tvsd_Xproc_{subject}_im{d.imsize}.pt")
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    if use_cache and os.path.exists(cache):
        blob = torch.load(cache, weights_only=False)
        return blob["X_train"], blob["X_test"]
    from PIL import Image
    pix = RawPixels(imsize=d.imsize)
    sroot = os.path.join(processed_dir("tvsd"), "stimuli", f"monkey{subject}")

    def build(split):
        sdir = os.path.join(sroot, split)
        n = len(os.listdir(sdir))
        feats = [pix._get_image_features(
                     np.asarray(Image.open(os.path.join(sdir, f"{i}.jpg")).convert("RGB")
                                .resize((d.imsize, d.imsize))))
                 for i in range(n)]
        return np.stack(feats).astype("float32")

    X_train, X_test = build("train"), build("test")
    if use_cache:
        os.makedirs(_cache_dir(), exist_ok=True)
        # protocol 4: at imsize 128 X_train is ~4.4 GB, over the 4 GB limit of pickle proto <=3.
        torch.save({"X_train": X_train, "X_test": X_test}, cache, pickle_protocol=4)
    return X_train, X_test


def load_tvsd(d: Dataset):
    """TVSD: Y from THINGS_normMUA.mat, X from the THINGS images, reward = ALL reliable electrodes.

    Returns (X_train, X_test, Y_train, Y_test, roi_masks, nc); ``roi_masks`` breaks the score down by
    cortical area (V1/V4/IT) and ``nc`` is the per-electrode R^2 noise ceiling (= ``oracle``^2) the
    reward normalizes by (r^2/oracle^2, the oracle-normalized predictivity). Electrodes with
    ``oracle < TVSD_ORACLE_MIN`` (default 0.2) are dropped from scoring -- the TVSD analog of NSD's
    ncsnr reliability cut. The split is the dataset's own (22248 train / 100 repeated test images).
    Prefers the in-repo processed tree (data/tvsd/) when present -- native JPEGs + a .pt of the MUA --
    so the gated THINGS images and the .mat are not needed; else falls back to the raw .mat + images.
    """
    subject = (d.subject or "F").upper()
    proc = processed_dir("tvsd")

    if os.path.isdir(proc):
        blob = torch.load(os.path.join(proc, "neural", f"monkey{subject}.pt"), weights_only=False)
        Y_train = np.asarray(blob["train_MUA"], dtype=np.float32)
        Y_test = np.asarray(blob["test_MUA"], dtype=np.float32)
        labels = blob["areas"]
        roi_masks = {a: labels == a for a in _TVSD_REGIONS}
        if "oracle" not in blob:
            raise RuntimeError(
                f"TVSD monkey{subject} [processed] blob at {proc!r} has no 'oracle' noise ceiling "
                f"-- re-run `python -m scripts.process_dataset.tvsd` to (re)stage it.")
        oracle = blob["oracle"]
        X_train, X_test = _tvsd_build_X_processed(d, subject)
        for split, nx, ny in (("train", len(X_train), len(Y_train)),
                              ("test", len(X_test), len(Y_test))):
            if nx != ny:
                raise RuntimeError(
                    f"TVSD monkey{subject} [processed]: {split} stimuli/neural count mismatch "
                    f"({nx} images vs {ny} MUA rows). The processed tree at {proc!r} is partial "
                    f"-- re-run `python -m scripts.process_dataset.tvsd` to stage all images, or "
                    f"`--n N` to stage a matched smoke subset of both.")
        Xtr, Xte = torch.as_tensor(X_train), torch.as_tensor(X_test)
        Ytr, Yte = torch.as_tensor(Y_train), torch.as_tensor(Y_test)
        return _tvsd_finalize(Xtr, Xte, Ytr, Yte, roi_masks, oracle, subject, " [processed]")

    mdir = os.path.join(d.path, f"monkey{subject}")
    mat = _read_mat(os.path.join(mdir, "THINGS_normMUA.mat"))
    Y_train = _tvsd_targets(mat["train_MUA"])           # [22248 x 1024]
    Y_test = _tvsd_targets(mat["test_MUA"])             # [100 x 1024]
    oracle = mat["oracle"]                              # [1024] per-electrode noise ceiling

    labels = _tvsd_area_labels(subject)                 # length-1024 V1/V4/IT per electrode
    roi_masks = {a: labels == a for a in _TVSD_REGIONS}

    X_train, X_test = _tvsd_build_X(d, subject, os.path.join(mdir, "things_imgs.mat"))

    Xtr, Xte = torch.as_tensor(X_train), torch.as_tensor(X_test)
    Ytr, Yte = torch.as_tensor(Y_train), torch.as_tensor(Y_test)
    return _tvsd_finalize(Xtr, Xte, Ytr, Yte, roi_masks, oracle, subject, "")
