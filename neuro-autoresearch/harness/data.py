"""Fixed data layer: stimulus + neural -> aligned, split (X, Y) tensors.

Modality-agnostic. Each ``Dataset`` declares a ``stimulus`` modality and a
``neural`` modality; this loader dispatches the right NeuralSet extractor for
each, aligns them per segment, and returns a train/test split. Datasets whose
stimuli repeat (e.g. NSD) get one sample per unique stimulus (repeats averaged)
and an image-disjoint canonical held-out test; others keep a contiguous temporal
split. The autoresearch agent should NOT need to edit this file.

Stimulus extractors are OFFLINE-friendly (no model-weight downloads):
  * image -> RawPixels (flattened RGB)
  * text  -> WordLength + WordFrequency (the trigger word's [length, zipf-freq])
  * audio -> MelSpectrum (time-averaged Mel bands)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess

import numpy as np
import torch
import neuralset as ns
from neuralset.extractors.image import BaseClassicImageExtractor

from harness.datasets import Dataset
from harness import studies  # noqa: F401  (registers NsdSubj01 / Li2022PetitSubj)


class RawPixels(BaseClassicImageExtractor):
    """Flattened, [0,1]-normalised RGB pixels of each stimulus image."""

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        return (image.astype("float32") / 255.0).reshape(-1)


def _study_path(d):
    """Resolve the NeuralSet study path: explicit ``d.path`` if set, else ``None`` to
    download-from-scratch when ``d.download``, else the shared ``ns.CACHE_FOLDER``."""
    return d.path if d.path is not None else (None if d.download else ns.CACHE_FOLDER)


_NEURAL = {
    # Bel2026 raw .fif is Elekta MaxShield -> allow_maxshield or it raises.
    "meg": lambda d: ns.extractors.MegExtractor(frequency=100.0, allow_maxshield=True),
    # iEEG/ECoG: like MEG, a fast electrophysiology signal. IeegExtractor picks the
    # ECOG/SEEG channels (dropping ECG/MISC) and read_raw_bids marks bads from channels.tsv.
    "ieeg": lambda d: ns.extractors.IeegExtractor(frequency=100.0),
    # fMRI: shift the window forward by `neural_offset` to align the delayed (HRF)
    # BOLD response and time-aggregate it (`neural_aggregation`); the default cleaning
    # (z-score + detrend) is applied by FmriExtractor itself.
    "fmri": lambda d: ns.extractors.FmriExtractor(
        aggregation=d.neural_aggregation, offset=d.neural_offset
    ),
}


def _stim_extractor(d: Dataset):
    """Build the stimulus extractor for the dataset's modality.

    Text/audio extractors are imported lazily so the image path needs none of
    their (optional) dependencies.
    """
    if d.stimulus == "image":
        return RawPixels(imsize=d.imsize)
    if d.stimulus in ("text", "word"):
        from neuralset.extractors.text import WordFrequency, WordLength
        from neuralset.extractors.meta import AggregatedExtractor

        # "trigger" aggregation => features of the triggering word only, so the
        # neural HRF offset (segment start) never leaks into the stimulus.
        return AggregatedExtractor(
            extractors=[
                WordLength(aggregation="trigger"),
                WordFrequency(aggregation="trigger"),
            ],
            extractor_aggregation="cat",
        )
    if d.stimulus == "audio":
        from neuralset.extractors.audio import MelSpectrum
        from neuralset.extractors.meta import TimeAggregatedExtractor

        return TimeAggregatedExtractor(extractor=MelSpectrum(), time_aggregation="mean")
    raise ValueError(f"unknown stimulus modality: {d.stimulus!r}")


def _load_workers() -> int:
    """DataLoader workers for the per-segment neural reads (env NEURO_LOAD_WORKERS).

    The per-segment fMRI/MEG reads are I/O-bound, so parallel workers cut the load
    by ~10x (e.g. the NSD subj01 load drops from ~25 min to a few min). Defaults to
    16; override via NEURO_LOAD_WORKERS.
    """
    return int(os.environ.get("NEURO_LOAD_WORKERS", "16"))


# --- on-disk build cache --------------------------------------------------------
# The slow thing is _build_xy: minutes of per-segment neural reads to assemble the
# full (X, Y) over ALL loaded voxels. That output is independent of the cheap,
# downstream choices -- the target region (``roi``, a column subset) and the split
# fraction -- so it is cached keyed by every field EXCEPT those. The first call
# builds and saves it; later calls (including a DIFFERENT roi) load it from disk in
# seconds and just re-subset/re-split, never re-reading the timeseries.
_CACHE_VERSION = "v2"

# Fields that only affect post-build (cheap) steps, so they are excluded from the
# build-cache key: ``roi`` subsets columns; ``test_frac`` only sizes a temporal tail;
# ``reward_roi`` only selects which targets the scalar reward averages over; ``run_tag``
# only names the output CSV. All region/run loops therefore share ONE build (the
# timeseries is read only once).
_BUILD_AGNOSTIC_FIELDS = ("roi", "test_frac", "reward_roi", "run_tag")


def _cache_dir() -> str:
    return os.environ.get(
        "NEURO_SPLIT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "neuro-autoresearch", "splits"),
    )


def _build_cache_key(d: Dataset) -> str:
    """A hash over the Dataset fields that affect _build_xy (all but roi/test_frac)."""
    fields = {k: v for k, v in dataclasses.asdict(d).items()
              if k not in _BUILD_AGNOSTIC_FIELDS}
    payload = json.dumps(fields, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{_CACHE_VERSION}:{payload}".encode()).hexdigest()[:16]
    return f"{d.study}_build_{digest}.pt"


def _build_cache_load(path: str):
    blob = torch.load(path, weights_only=False)
    return blob["X"], blob["Y"], blob["onsets"], blob["stim_ids"]


def _build_cache_save(path: str, X, Y, onsets, stim_ids) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save({"X": X, "Y": Y, "onsets": onsets, "stim_ids": stim_ids}, tmp)
    os.replace(tmp, path)


# --- segmentation ---------------------------------------------------------------
def _trigger_key(seg) -> object:
    """Stable identity of a segment's trigger, shared across separate segmenters.

    Two segmenters run on the SAME events (and thus the same triggers); the trigger
    DataFrame index ``_index`` is the natural shared key. Fall back to
    (timeline, onset) when an index is unavailable.
    """
    t = seg.trigger
    idx = getattr(t, "_index", None)
    return idx if idx is not None else (seg.timeline, round(t.start, 6))


def _seg_stim_ids(segments):
    """Per-segment stimulus identity from the trigger's ``filepath`` (or ``None``).

    Datasets whose stimuli repeat (e.g. NSD: each image is shown ~3x) expose a
    per-trigger stimulus id; the loader uses it to collapse repeats and to build an
    image-disjoint split. Modalities without a per-stimulus file (text/audio word
    triggers) have no ``filepath`` -> returns ``None`` and the loader keeps its
    contiguous-temporal split unchanged.
    """
    ids = [getattr(s.trigger, "filepath", None) for s in segments]
    return np.array(ids, dtype=object) if any(i is not None for i in ids) else None


def _segment_batch(d: Dataset, events, extractors: dict, start: float, duration: float):
    seg = ns.Segmenter(
        extractors=extractors,
        trigger_query=d.trigger, start=start, duration=duration,
        drop_incomplete=True,
    )
    return seg.apply(events).load_all(num_workers=_load_workers())


def _stim_window(d: Dataset) -> tuple[float, float]:
    """The stimulus (start, duration); defaults to the neural window."""
    start = d.start if d.stim_start is None else d.stim_start
    duration = d.duration if d.stim_duration is None else d.stim_duration
    return start, duration


def _neural_2d(neural, d) -> torch.Tensor:
    """Per-segment neural tensor -> 2-D (n_segments, n_targets).

    With ``neural_aggregation == "mean"`` the per-segment time (TR) axis is averaged
    into ONE value per channel -- a denoised, betas-like response that lifts the
    encoding score markedly (for NSD it takes a trained AlexNet from ~0.28 to ~0.32).
    ``"single"`` keeps every TR as a separate target (the default; unchanged).
    """
    neural = neural.float()
    if d.neural_aggregation == "mean" and neural.dim() >= 3:
        neural = neural.mean(dim=-1)        # collapse the trailing TR/time axis
    return neural.reshape(neural.shape[0], -1)


def _build_xy(d: Dataset, events):
    """Build row-aligned (X, Y, onsets, stim_ids): one row per trigger.

    ``onsets`` is the per-row trigger time (the temporal-split sort key); ``stim_ids``
    is the per-row stimulus identity (``None`` when the modality has no per-stimulus
    file). Fast path: when the stimulus window equals the neural window, a single
    Segmenter extracts both, already row-aligned. Decoupled path (e.g. audio on
    HRF-delayed fMRI): the stimulus and neural are segmented separately and aligned
    by trigger -- ``drop_incomplete`` may drop different triggers on each side, so
    the result is the intersection, ordered by trigger onset.
    """
    stim_start, stim_duration = _stim_window(d)
    stim_ext, neural_ext = _stim_extractor(d), _NEURAL[d.neural](d)

    if (stim_start, stim_duration) == (d.start, d.duration):
        batch = _segment_batch(
            d, events, dict(stim=stim_ext, neural=neural_ext), d.start, d.duration
        )
        X = batch.data["stim"].float()
        Y = _neural_2d(batch.data["neural"], d)
        onsets = np.array([s.start for s in batch.segments])
        return X, Y, onsets, _seg_stim_ids(batch.segments)

    neural_batch = _segment_batch(d, events, dict(neural=neural_ext), d.start, d.duration)
    stim_batch = _segment_batch(d, events, dict(stim=stim_ext), stim_start, stim_duration)

    stim_row = {_trigger_key(s): i for i, s in enumerate(stim_batch.segments)}
    paired = [
        (i, stim_row[k])
        for i, s in enumerate(neural_batch.segments)
        if (k := _trigger_key(s)) in stim_row
    ]
    if not paired:
        raise RuntimeError("no triggers shared between the stimulus and neural windows")
    n_idx = [i for i, _ in paired]
    s_idx = [j for _, j in paired]

    X = stim_batch.data["stim"].float()[s_idx]
    Y = _neural_2d(neural_batch.data["neural"], d)[n_idx]
    onsets = np.array([neural_batch.segments[i].trigger.start for i in n_idx])
    stim_ids = _seg_stim_ids([neural_batch.segments[i] for i in n_idx])
    return X, Y, onsets, stim_ids


# --- neural-target region (ROI) selection ---------------------------------------
# Volumetric studies (NSD) load a fixed set of voxels (the nsdgeneral mask) as the
# target Y. A sub-ROI is a COLUMN-SUBSET of those already-loaded voxels -- not a
# re-read of the 65 GB timeseries: we map a small ROI label volume onto the SAME
# voxel order the loader used (``label_volume[nsdgeneral_mask]``), then keep only
# the requested labels. Voxels outside nsdgeneral are not available (clipped).
#
# Each name maps to (NSD ROI label file, the label values within it to keep). The menu
# is the famous specific functional regions + the early/ventral visual streams; these
# are the reward targets (pick one per loop with NEURO_ROI) and the logged feedback panel.
_ROI_DEFS: dict[str, tuple[str, tuple[int, ...]]] = {
    "V1": ("prf-visualrois", (1, 2)),    # primary visual cortex (V1v + V1d)
    "V2": ("prf-visualrois", (3, 4)),    # secondary visual cortex (V2v + V2d)
    "V3": ("prf-visualrois", (5, 6)),    # third visual area (V3v + V3d)
    "hV4": ("prf-visualrois", (7,)),     # human V4
    "FFA": ("floc-faces", (2, 3)),       # fusiform face area (FFA-1 + FFA-2)
    "PPA": ("floc-places", (2,)),        # parahippocampal place area
    "EBA": ("floc-bodies", (1,)),        # extrastriate body area
    "VWFA": ("floc-words", (2, 3)),      # visual word form area (VWFA-1 + VWFA-2)
    "early": ("streams", (1,)),          # early visual stream
    "ventral": ("streams", (2, 3)),      # ventral visual stream (midventral + ventral)
}

_S3_NSD = "s3://natural-scenes-dataset"


def _nsd_subject(study) -> int:
    return min(type(study).SESSIONS_PER_SUBJECT)


def _ensure_roi_label_file(study, label_file: str):
    """Path to a small NSD ROI label volume, downloaded next to nsdgeneral if absent.

    The label .nii.gz lives in the study's func1pt8mm/roi dir (OUTSIDE the repo);
    it is tiny (~64 KB) and fetched once via the AWS CLI (anonymous).
    """
    subject = _nsd_subject(study)
    dst = study._roi_path(subject).parent / f"{label_file}.nii.gz"
    if not dst.exists():
        src = f"{_S3_NSD}/nsddata/ppdata/subj{subject:02}/func1pt8mm/roi/{label_file}.nii.gz"
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", src, str(dst)], check=True)
    return dst


def _roi_voxel_keep(study, roi: str):
    """(boolean keep-mask over the nsdgeneral voxels, n_nsdgeneral_voxels) for ``roi``.

    Returns ``(None, n_voxels)`` for the default full ``nsdgeneral`` region.
    """
    from neuralfetch.studies.allen2022massive import _get_roi_mask

    mask = _get_roi_mask(study._roi_path(_nsd_subject(study)))  # nsdgeneral 3-D bool
    n_voxels = int(mask.sum())
    if roi == "nsdgeneral":
        return None, n_voxels
    if roi not in _ROI_DEFS:
        raise ValueError(f"unknown roi {roi!r}; known: {['nsdgeneral', *_ROI_DEFS]}")
    import nibabel

    label_file, wanted = _ROI_DEFS[roi]
    labels = nibabel.load(_ensure_roi_label_file(study, label_file)).get_fdata()
    return np.isin(labels[mask], wanted), n_voxels


def _apply_roi(d: Dataset, study, Y):
    """Column-subset Y to the dataset's ``roi`` (no-op for the default nsdgeneral)."""
    if d.roi == "nsdgeneral":
        return Y
    if not hasattr(study, "_roi_path"):
        raise ValueError(f"roi={d.roi!r} requested but study {d.study!r} has no ROI masks")
    keep, n_voxels = _roi_voxel_keep(study, d.roi)
    if keep is None:
        return Y
    n_tr = Y.shape[1] // n_voxels  # voxels are flattened voxel-major: (voxel, tr)
    if Y.shape[1] != n_voxels * n_tr:
        raise RuntimeError(
            f"Y has {Y.shape[1]} columns, not a multiple of {n_voxels} nsdgeneral voxels"
        )
    col_keep = torch.as_tensor(np.repeat(keep, n_tr))
    return Y[:, col_keep]


# --- unique-stimulus collapse + canonical held-out split ------------------------
def _collapse_repeats(X, Y, onsets, stim_ids):
    """Average the neural target over the repeats of each unique stimulus.

    NSD shows each image ~3x; the stimulus X is identical across a repeat, so its
    group mean is unchanged, while averaging Y reduces trial noise. Returns one row
    per unique stimulus (X, Y, onsets, stim_ids), ordered by stimulus id.
    """
    uniq, first_idx, inv = np.unique(stim_ids, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    inv_t = torch.as_tensor(inv, dtype=torch.long)
    g = len(uniq)
    # Per-group sizes, computed ONCE; both averages divide by the clamped (>=1) count.
    counts_t = torch.bincount(inv_t, minlength=g).clamp(min=1)
    counts = counts_t.unsqueeze(1).float()
    counts_np = counts_t.numpy()

    # Y is averaged over repeats (this is what reduces trial noise).
    Y_g = torch.zeros(g, Y.shape[1], dtype=torch.float32).index_add_(0, inv_t, Y.float()) / counts
    # X is identical across a stimulus's repeats, so gather one row instead of
    # averaging the (wide) stimulus tensor.
    X_g = X[torch.as_tensor(first_idx, dtype=torch.long)]
    # Mean onset per group (used only by non-canonical splits).
    onsets_g = np.bincount(inv, weights=onsets, minlength=g) / counts_np
    return X_g, Y_g, onsets_g, uniq


def _image_id(filepath: str) -> int:
    """NSD 1-based image id from a stimulus path (``.../{id-1}.png``)."""
    return int(os.path.splitext(os.path.basename(filepath))[0]) + 1


def _canonical_test_ids(study) -> set[int]:
    """The NSD shared-1000 image ids (``sharedix``), via the study's data interface."""
    return {int(x) for x in np.asarray(study._get_test_image_ids()).reshape(-1).tolist()}


# --- TVSD: pre-averaged [electrodes x stimuli] table (no Segmenter) --------------
# TVSD ships a normalized-MUA table already aligned to its THINGS stimuli, so it bypasses
# the timeline/Segmenter machinery entirely: Y is read straight from the .mat and X is built
# from the THINGS images each column points at. ALL raw file I/O for this dataset (.mat +
# image reads) is confined to this section -- the one place in the fixed harness that reads
# outside NeuralSet, because NeuralSet has no TVSD study to express a pre-averaged table.
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


def _load_tvsd(d: Dataset):
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


def load_split(d: Dataset):
    """Return (X_train, X_test, Y_train, Y_test) as 2-D float tensors.

    Two split regimes, chosen by the data itself:

    * **Repeated stimuli with a canonical held-out set** (e.g. NSD: each image is
      shown ~3x and a fixed shared-1000 subset is the standard test). The repeats
      of each unique stimulus are collapsed to one sample (the neural target is
      averaged; the identical stimulus is unchanged), and the held-out test is the
      canonical shared set -- an **image-disjoint** split (asserted), no leakage.
    * **Otherwise** a contiguous temporal hold-out (latest ``test_frac`` of the
      triggers by onset): for time-continuous neural signals (fMRI/MEG/EEG) nearby
      triggers share samples, so a random split would leak near-duplicate rows.

    For volumetric studies the dataset's ``roi`` further column-subsets the neural
    target to a brain region (a fast subset of the already-loaded voxels).

    The slow ``_build_xy`` output is cached on disk keyed by every field except
    ``roi``/``test_frac`` (see ``_build_cache_key``), so a different ROI re-uses it
    and just re-subsets/re-splits in seconds -- never re-reading the timeseries.
    Bypass with NEURO_NO_CACHE, relocate with NEURO_SPLIT_CACHE.

    Table-loader datasets (e.g. TVSD) ship a pre-averaged [electrodes x stimuli] table and
    take a separate, self-contained path (``_load_tvsd``) that never touches the Segmenter.
    """
    if d.loader == "table":
        return _load_tvsd(d)
    use_cache = not os.environ.get("NEURO_NO_CACHE")
    build_path = os.path.join(_cache_dir(), _build_cache_key(d))

    study = ns.Study(name=d.study, path=_study_path(d))

    if use_cache and os.path.exists(build_path):
        X, Y, onsets, stim_ids = _build_cache_load(build_path)   # full voxels, all trials
    else:
        if d.download:
            study.download()
        try:
            events = study.run()
        except FileNotFoundError:           # cold demo cache: self-provision once
            study.download()
            events = study.run()
        X, Y, onsets, stim_ids = _build_xy(d, events)
        if use_cache:
            _build_cache_save(build_path, X, Y, onsets, stim_ids)

    n_trials = X.shape[0]
    Y = _apply_roi(d, study, Y)         # column-subset the neural target to d.roi

    # Collapse repeats of the same stimulus to one sample (no-op without repeats).
    repeated = stim_ids is not None and len(np.unique(stim_ids)) < len(stim_ids)
    if repeated:
        X, Y, onsets, stim_ids = _collapse_repeats(X, Y, onsets, stim_ids)
    n_unique = X.shape[0]

    canonical = stim_ids is not None and hasattr(study, "_get_test_image_ids")
    if canonical:                       # image-disjoint shared-1000 held-out test
        shared = _canonical_test_ids(study)
        is_test = np.array([_image_id(s) in shared for s in stim_ids])
        tr = torch.as_tensor(np.flatnonzero(~is_test), dtype=torch.long)
        te = torch.as_tensor(np.flatnonzero(is_test), dtype=torch.long)
        train_ids = set(stim_ids[np.flatnonzero(~is_test)].tolist())
        test_ids = set(stim_ids[np.flatnonzero(is_test)].tolist())
        assert train_ids.isdisjoint(test_ids), "train/test share images -> leakage"
        assert te.numel() and tr.numel(), "empty train or test split"
    else:                               # contiguous temporal split by trigger onset
        order = torch.as_tensor(np.argsort(onsets, kind="stable"), dtype=torch.long)
        n = order.numel()
        n_test = max(1, int(round(n * d.test_frac)))
        tr, te = order[: n - n_test], order[n - n_test:]
    X_train, X_test, Y_train, Y_test = X[tr], X[te], Y[tr], Y[te]

    regime = "shared-1000 held-out" if canonical else "temporal hold-out"
    roi_note = f"; roi={d.roi}" if d.roi != "nsdgeneral" else ""
    print(f"  {n_trials} trials -> {n_unique} unique stimuli; "
          f"train/test = {X_train.shape[0]}/{X_test.shape[0]} samples ({regime}{roi_note}); "
          f"{Y_train.shape[1]} target dims")

    # Drop targets that are constant on the TRAIN set: they carry no signal, are
    # unpredictable, and (e.g. non-brain fMRI voxels) only dilute the mean-r and
    # waste compute. The mask uses train variance only (no test peeking).
    keep = Y_train.std(0) > 0
    return (X_train, X_test, Y_train[:, keep], Y_test[:, keep])
