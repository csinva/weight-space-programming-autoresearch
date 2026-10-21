"""Small real-format fixtures for alignment, filtering and cross-dataset isolation."""
from dataclasses import FrozenInstanceError, replace
import io
import pickle

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import xarray as xr

from harness.datasets import PRESETS, dataset_key, get_dataset, with_subject
from harness.loaders import _SPLIT_MEMO, load_split
from harness.loaders.vision.looped import REGIONS, dataset_provenance
from harness.protocols import canonical_dataset, protocol_for, search_subjects_csv, validate_search_dataset
from harness.vendor.looped.data.stimuli import LazyEncodedHdf5Dict


@pytest.fixture(autouse=True)
def isolated_loader(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURO_NO_CACHE", "1")
    monkeypatch.setenv("NEURO_SPLIT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("NEURO_SUBJECT", raising=False)
    _SPLIT_MEMO.clear()


def _response(ids, **coordinates):
    ids = np.array(ids)
    return xr.DataArray(
        ids[:, None] * 10 + np.arange(3)[None, :], dims=("stimulus", "neuroid"),
        coords={"stimulus": ids, **{name: ("neuroid", values) for name, values in coordinates.items()}},
    )


def _write_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file)


def _write_images(path, n=6, encoded=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        if encoded:
            images = file.create_dataset("images", (n,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for i in range(n):
                buffer = io.BytesIO()
                Image.new("RGB", (3, 3), (i * 20, 10, 5)).save(buffer, format="PNG")
                images[i] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        else:
            values = np.zeros((n, 3, 3, 3), dtype=np.uint8)
            values[:, :, :, 0] = np.arange(n)[:, None, None] * 20
            values[:, :, :, 1] = 10
            file.create_dataset("imgBrick", data=values)


def _dataset(name, subject=None):
    # Tiny pixels exercise the adapter independently of the user-facing 128px floor.
    result = PRESETS[name]
    return replace(with_subject(result, subject or result.subject), imsize=2)


def test_protocols_fix_cohorts_metrics_and_aliases():
    assert canonical_dataset("LAION-fMRI") == "laion"
    assert canonical_dataset("nnn") == "triplen"
    assert canonical_dataset("triple-n") == "triplen"
    assert protocol_for("nsd") == protocol_for("laion")
    assert protocol_for("triplen").final_dataset == "tvsd"
    assert protocol_for("triplen").final_metric == "encoding"
    assert search_subjects_csv("laion") == "sub-01,sub-03,sub-05,sub-06,sub-07"
    assert len(protocol_for("nsd").final_subjects) == 8
    with pytest.raises(FrozenInstanceError):
        protocol_for("laion").search_metric = "encoding"
    with pytest.raises(ValueError, match="reserved"):
        validate_search_dataset("nsd")
    with pytest.raises(ValueError, match="Unknown"):
        canonical_dataset("nsdd")


def test_dataset_resolution_and_subject_binding(monkeypatch):
    monkeypatch.setenv("NEURO_IMSIZE", "256")
    assert get_dataset("laion_fmri", "3").subject == "sub-03"
    assert get_dataset("nnn", "2").subject == "M2"
    assert dataset_key(get_dataset("nnn")) == "triplen"
    assert get_dataset("nsd").imsize == 256
    assert get_dataset("tvsd", "n").stage == "final"
    monkeypatch.setenv("NEURO_IMSIZE", "64")
    with pytest.raises(ValueError, match="MIN_IMSIZE"):
        get_dataset("laion")


def test_laion_excludes_nsd_source_and_private_images_preserving_alignment(tmp_path, monkeypatch):
    from harness.vendor.looped.data.laion import _LAION_REGION_MAP
    root = tmp_path / "laion"
    monkeypatch.setenv("LAION_DATA_DIR", str(root / "processed"))
    monkeypatch.setenv("LAION_FMRI_DIR", str(root))
    # Deliberately unsorted source rows; 0 and 3 are private, 2 is NSD-source.
    response = _response([4, 0, 2, 1, 3], ncsnr=[0.2, 0.25, 0.8], x=[0, 1, 2], y=[0, 0, 0], z=[0, 0, 0])
    source = {
        "data": {key: {0: response} for key in _LAION_REGION_MAP.values()},
        "voxel_counts": {key: {0: (3, 3)} for key in _LAION_REGION_MAP.values()},
        "ncsnr_threshold": 0.2, "shared_ids": {1, 2, 4},
        "stimuli": pd.DataFrame({"dataset": ["LAION", "LAION", "NSD", "LAION", "OOD"]}),
    }
    _write_pickle(root / "processed/laion_data.pkl", source)
    _write_images(root / "stimuli/images/task-images_stimuli.h5", encoded=True)
    d = _dataset("laion")
    Xtr, Xte, Ytr, Yte, masks, nc = load_split(d)
    assert Xtr.shape == (0, 12) and Ytr.shape == (0, 4)
    assert Xte.shape == (2, 12) and Yte.shape == (2, 4)
    np.testing.assert_allclose(Xte[:, 0], [20 / 255, 80 / 255])
    np.testing.assert_array_equal(Yte[:, :2], [[11, 12], [41, 42]])
    assert set(masks) == set(REGIONS["laion"])
    assert all(mask.sum() == 2 for mask in masks.values())
    assert nc is None
    assert dataset_provenance(d)["test_ids"] == ["1", "4"]


def test_triplen_uses_all_images_and_each_monkeys_available_regions(tmp_path, monkeypatch):
    from harness.vendor.looped.data.triplen import _TRIPLEN_REGION_MAP
    root = tmp_path / "triplen"
    monkeypatch.setenv("TRIPLEN_DATA_DIR", str(root))
    monkeypatch.setenv("NSD_STIMULI_HDF5_PATH", str(root / "images.h5"))
    response = _response([4, 0, 3, 1, 2], reliability=[0.4, 0.6, 0.9], session=[1, 1, 2], unit=[0, 1, 2], unit_type=[1, 1, 2])
    response = response.assign_coords(fold=("stimulus", [4, 0, 3, 1, 2]))
    keys = set(_TRIPLEN_REGION_MAP.values())
    source = {
        "data": {key: ({1: response} if key in ("ventral", "IT") else {}) for key in keys},
        "unit_counts": {key: ({1: (3, 3)} if key in ("ventral", "IT") else {}) for key in keys},
        "reliability_threshold": 0.4, "stimuli": pd.DataFrame({"nsdId": range(5)}),
    }
    _write_pickle(root / "triplen_data.pkl", source)
    _write_images(root / "images.h5")
    d = _dataset("triplen", "M2")
    Xtr, Xte, Ytr, Yte, masks, _ = load_split(d)
    assert Xtr.shape[0] == Ytr.shape[0] == 0
    assert Xte.shape[0] == Yte.shape[0] == 5
    assert list(masks) == ["ventral visual stream"]
    np.testing.assert_array_equal(Yte[:, 0], [1, 11, 21, 31, 41])
    assert dataset_provenance(d)["test_ids"] == ["0", "1", "2", "3", "4"]


def test_tvsd_preserves_official_split_and_strict_oracle_filter(tmp_path, monkeypatch):
    from harness.vendor.looped.data.tvsd import _TVSD_REGION_MAP
    root = tmp_path / "tvsd"
    monkeypatch.setenv("TVSD_DATA_DIR", str(root))
    response = _response([5, 2, 1, 4, 0, 3], oracle=[0.4, 0.6, 0.9], electrode=[0, 1, 2])
    response = response.assign_coords(split=("stimulus", ["test", "train", "test", "train", "train", "test"]))
    source = {
        "data": {key: {0: response} for key in _TVSD_REGION_MAP.values()},
        "unit_counts": {key: {0: (3, 3)} for key in _TVSD_REGION_MAP.values()},
        "oracle_threshold": 0.4, "stimuli": pd.DataFrame({"stimulus_id": range(6)}),
    }
    _write_pickle(root / "tvsd_data.pkl", source)
    _write_images(root / "tvsd_stimuli.h5", encoded=True)
    d = _dataset("tvsd")
    Xtr, Xte, Ytr, Yte, _, nc = load_split(d)
    assert Xtr.shape[0] == Xte.shape[0] == 3
    np.testing.assert_array_equal(Ytr[:, 0], [1, 21, 41])
    np.testing.assert_array_equal(Yte[:, 0], [11, 31, 51])
    assert nc is None
    assert dataset_provenance(d)["train_ids"] == ["0", "2", "4"]


def test_nsd_filters_official_ncsnr_and_retains_shared_holdout(tmp_path, monkeypatch):
    from harness.vendor.looped.data.nsd import _NSD_REGION_MAP
    root = tmp_path / "nsd"
    monkeypatch.setenv("NSD_DATA_DIR", str(root))
    monkeypatch.setenv("NSD_BETAS_DIR", str(root / "betas"))
    monkeypatch.setenv("NSD_STIMULI_HDF5_PATH", str(root / "images.h5"))
    response = _response([3, 0, 2, 1], x=[0, 1, 2], y=[0, 0, 0], z=[0, 0, 0])
    source = {"data": {_NSD_REGION_MAP[r]: {0: response} for r in REGIONS["nsd"]}, "shared_ids": {1, 3}}
    _write_pickle(root / "nsd_data.pkl", source)
    _write_images(root / "images.h5")
    volume = np.array([0.2, 0.25, 0.8], dtype=np.float32).reshape(3, 1, 1)
    path = root / "betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    d = _dataset("nsd")
    Xtr, Xte, Ytr, Yte, masks, nc = load_split(d)
    assert Xtr.shape[0] == Xte.shape[0] == 2
    np.testing.assert_array_equal(Ytr[:, :2], [[1, 2], [21, 22]])
    np.testing.assert_array_equal(Yte[:, :2], [[11, 12], [31, 32]])
    assert all(mask.sum() == 2 for mask in masks.values())
    assert nc is None  # raw encoding Pearson r, not the retired normalized R²


def test_encoded_images_composite_transparency_over_scanner_grey(tmp_path):
    path = tmp_path / "transparent.h5"
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(buffer, format="PNG")
    with h5py.File(path, "w") as file:
        images = file.create_dataset("images", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        images[0] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    images = LazyEncodedHdf5Dict(path, [0])
    np.testing.assert_array_equal(images["0"], np.full((2, 2, 3), 128, dtype=np.uint8))
