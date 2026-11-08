"""Small real-format fixtures for alignment, filtering and cross-dataset isolation."""
from dataclasses import replace
import io
import os
import pickle

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import xarray as xr

from neuro_eval.datasets import PRESETS, with_subject
from neuro_eval.datasets import _SPLIT_MEMO, load_split
from neuro_eval.datasets import REGIONS, dataset_provenance


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
                # Image 4 is transparent; its decoded pixels must be scanner gray.
                color = (i * 20, 10, 5, 0 if i == 4 else 255)
                Image.new("RGBA", (3, 3), color).save(buffer, format="PNG")
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


@pytest.mark.parametrize("metric", ["rsa", "encoding"])
def test_laion_holds_out_half_of_each_source_and_trains_on_private_images(tmp_path, monkeypatch, metric):
    from neuro_eval.readers import _LAION_REGION_MAP
    root = tmp_path / "laion"
    monkeypatch.setenv("LAION_DATA_DIR", str(root / "processed"))
    monkeypatch.setenv("LAION_FMRI_DIR", str(root))
    # Deliberately unsorted rows; 0 and 3 are private, 0 and 2 are NSD-source.
    response = _response([4, 0, 2, 1, 3], ncsnr=[0.2, 0.25, 0.8], x=[0, 1, 2], y=[0, 0, 0], z=[0, 0, 0])
    source = {
        "data": {key: {0: response} for key in _LAION_REGION_MAP.values()},
        "voxel_counts": {key: {0: (3, 3)} for key in _LAION_REGION_MAP.values()},
        "ncsnr_threshold": 0.2, "shared_ids": {1, 2, 4},
        "stimuli": pd.DataFrame({"dataset": ["NSD", "LAION", "NSD", "LAION", "OOD"]}),
    }
    _write_pickle(root / "processed/laion_data.pkl", source)
    _write_images(root / "stimuli/images/task-images_stimuli.h5", encoded=True)
    # One image per source: each lands in the held-out half, leaving the search half empty.
    with pytest.raises(ValueError, match="No evaluation stimuli"):
        load_split(replace(_dataset("laion"), metric=metric))
    d = replace(_dataset("laion"), metric=metric, images="heldout")
    Xtr, Xte, Ytr, Yte, masks, nc = load_split(d)
    if metric == "rsa":
        assert Xtr.shape == (0, 12) and Ytr.shape == (0, 2)
        assert dataset_provenance(d)["train_ids"] == []
    else:
        assert Xtr.shape == (2, 12) and Ytr.shape == (2, 2)
        np.testing.assert_allclose(Xtr[:, 0], [0, 60 / 255])
        np.testing.assert_array_equal(Ytr, [[1, 2], [31, 32]])
        assert dataset_provenance(d)["train_ids"] == ["0", "3"]
    assert Xte.shape == (3, 12) and Yte.shape == (3, 2)
    np.testing.assert_allclose(Xte[:, 0], [20 / 255, 40 / 255, 128 / 255])
    np.testing.assert_allclose(Xte[2], 128 / 255)
    np.testing.assert_array_equal(Yte[:, :2], [[11, 12], [21, 22], [41, 42]])
    assert set(masks) == {"ventral visual stream", "hV4", "FFA", "PPA"}
    assert all(mask.sum() == 2 for mask in masks.values())
    assert nc is None
    assert dataset_provenance(d)["test_ids"] == ["1", "2", "4"]
    if metric == "rsa":
        assert load_split(d).source_groups == {"laion": [0], "thingsplus": [], "ood": [2], "nsd": [1]}
        assert dataset_provenance(d)["source_group_counts"] == {"laion": 1, "thingsplus": 0, "ood": 1, "nsd": 1}


def test_triplen_uses_all_images_and_each_monkeys_available_regions(tmp_path, monkeypatch):
    from neuro_eval.readers import _TRIPLEN_REGION_MAP
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
    assert list(masks) == ["ventral visual stream", "IT"]
    np.testing.assert_array_equal(Yte[:, 0], [1, 11, 21, 31, 41])
    assert dataset_provenance(d)["test_ids"] == ["0", "1", "2", "3", "4"]


def test_tvsd_preserves_official_split_and_strict_oracle_filter(tmp_path, monkeypatch):
    from neuro_eval.readers import _TVSD_REGION_MAP
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


def test_nsd_metadata_selects_volumes_preserves_holdout_and_invalidates_cache(tmp_path, monkeypatch):
    from neuro_eval.readers import _NSD_REGION_MAP
    monkeypatch.setenv("NSD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NSD_DATA_FILE", "nsd_data_ext.pkl")
    monkeypatch.setenv("NSD_BETAS_DIR", str(tmp_path / "betas"))
    monkeypatch.setenv("NSD_STIMULI_HDF5_PATH", str(tmp_path / "images.h5"))
    response = _response([3, 0, 2, 1], x=[1, 0, 1], y=[0, 1, 1], z=[1, 0, 0])
    source = {"data": {_NSD_REGION_MAP[r]: {2: response.copy()} for r in REGIONS["nsd"]},
              "shared_ids": {1, 3}}
    source["data"]["ventral"][2].attrs.update(resolution="1pt0mm", preprocessing="fithrf")
    _write_pickle(tmp_path / "nsd_data_ext.pkl", source)
    _write_images(tmp_path / "images.h5")
    early = tmp_path / "betas/ppdata/subj03/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"
    ventral = tmp_path / "betas/ppdata/subj03/func1pt0mm/betas_fithrf/ncsnr.nii.gz"

    def write_volume(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        volume = np.zeros((2, 2, 2), dtype=np.float32)
        volume[[1, 0, 1], [0, 1, 1], [1, 0, 0]] = values
        nib.save(nib.Nifti1Image(volume, np.eye(4)), path)

    write_volume(early, [0.2, 0.25, 0.8])
    write_volume(ventral, [0.8, 0.1, 0.1])
    d = replace(_dataset("nsd", "03"), stage="final", metric="encoding")  # final loads both streams
    # The two shared images split into one search image (3) and one held-out image (1).
    assert dataset_provenance(replace(d, images="heldout"))["test_ids"] == ["1"]
    before = load_split(d)
    Xtr, Xte, Ytr, Yte, masks, nc = before
    assert (Xtr.shape[0], Xte.shape[0]) == (2, 1)
    np.testing.assert_allclose(Xtr[:, 0], [0, 40 / 255])
    np.testing.assert_allclose(Xte[:, 0], [60 / 255])
    np.testing.assert_array_equal(Ytr, [[1, 2, 0], [21, 22, 20]])
    np.testing.assert_array_equal(Yte, [[31, 32, 30]])
    assert [m.sum() for m in masks.values()] == [2, 1]
    assert nc is None
    files = dataset_provenance(d)["files"]
    assert files["responses"]["path"] == str(tmp_path / "nsd_data_ext.pkl")
    assert files["ncsnr:early visual stream"]["path"] == str(early)
    assert files["ncsnr:ventral visual stream"]["path"] == str(ventral)
    assert load_split(d) is before
    previous = ventral.stat()
    write_volume(ventral, [0.1, 0.8, 0.8])
    os.utime(ventral, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    after = load_split(d)
    assert after is not before
    assert after[4]["ventral visual stream"].sum() == 2
    assert dataset_provenance(d)["files"] != files
