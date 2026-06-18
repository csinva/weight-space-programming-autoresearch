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
from harness.loaders.core import RawPixels, _cache_dir

_TVSD_N_ELEC = 1024
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
        torch.save({"X_train": X_train, "X_test": X_test}, cache)
    return X_train, X_test


def load_tvsd(d: Dataset):
    """TVSD: Y from THINGS_normMUA.mat, X from the THINGS images, reward = one area.

    Returns (X_train, X_test, Y_train, Y_test) as 2-D float tensors -- the same contract as
    the Segmenter path. The split is the dataset's own (22248 train / 100 repeated test
    images); ``reward_roi`` (NEURO_ROI) picks the area V1/V4/IT as a column-subset of the
    1024 electrodes (the readout then fits and scores exactly that area's electrodes).
    """
    subject = (d.subject or "F").upper()
    mdir = os.path.join(d.path, f"monkey{subject}")
    mat = _read_mat(os.path.join(mdir, "THINGS_normMUA.mat"))
    Y_train = _tvsd_targets(mat["train_MUA"])           # [22248 x 1024]
    Y_test = _tvsd_targets(mat["test_MUA"])             # [100 x 1024]

    region = d.reward_roi
    known_region = region in _TVSD_REGIONS
    region_label = region if known_region else "all"
    cols = (_tvsd_area_labels(subject) == region) if known_region \
        else np.ones(_TVSD_N_ELEC, dtype=bool)          # unknown / "all" -> every electrode
    Y_train, Y_test = Y_train[:, cols], Y_test[:, cols]

    X_train, X_test = _tvsd_build_X(d, subject, os.path.join(mdir, "things_imgs.mat"))

    keep = Y_train.std(0) > 0                            # drop electrodes constant on train
    Xtr, Xte = torch.as_tensor(X_train), torch.as_tensor(X_test)
    Ytr, Yte = torch.as_tensor(Y_train[:, keep]), torch.as_tensor(Y_test[:, keep])
    print(f"  TVSD monkey{subject}: {Xtr.shape[0]} train / {Xte.shape[0]} test images; "
          f"region={region_label} -> "
          f"{Ytr.shape[1]} electrodes ({Xtr.shape[1]} stim dims)")
    return Xtr, Xte, Ytr, Yte
