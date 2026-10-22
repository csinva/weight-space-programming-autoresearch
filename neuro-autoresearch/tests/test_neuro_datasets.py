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


def test_laion_excludes_nsd_source_and_private_images_preserving_alignment(tmp_path, monkeypatch):
    from neuro_eval.readers import _LAION_REGION_MAP
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
    assert load_split(d).source_groups == {"laion": [0], "thingsplus": [], "ood": [1]}
    assert dataset_provenance(d)["source_group_counts"] == {"laion": 1, "thingsplus": 0, "ood": 1}


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
    assert list(masks) == ["ventral visual stream"]
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
    response = _response([3, 0, 2, 1], x=[0, 1, 2], y=[0, 0, 0], z=[0, 0, 0])
    source = {"data": {_NSD_REGION_MAP[r]: {2: response.copy()} for r in REGIONS["nsd"]},
              "shared_ids": {1, 3}}
    source["data"]["ventral"][2].attrs.update(resolution="1pt0mm", preprocessing="fithrf")
    _write_pickle(tmp_path / "nsd_data_ext.pkl", source)
    _write_images(tmp_path / "images.h5")
    early = tmp_path / "betas/ppdata/subj03/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"
    ventral = tmp_path / "betas/ppdata/subj03/func1pt0mm/betas_fithrf/ncsnr.nii.gz"

    def write_volume(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        volume = np.array(values, dtype=np.float32).reshape(3, 1, 1)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), path)

    write_volume(early, [0.2, 0.25, 0.8])
    write_volume(ventral, [0.8, 0.1, 0.1])
    d = _dataset("nsd", "03")
    before = load_split(d)
    Xtr, Xte, Ytr, Yte, masks, nc = before
    assert Xtr.shape[0] == Xte.shape[0] == 2
    np.testing.assert_array_equal(Ytr, [[1, 2, 0], [21, 22, 20]])
    np.testing.assert_array_equal(Yte, [[11, 12, 10], [31, 32, 30]])
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


def test_prepared_pickle_cache_tracks_resolved_source_identity(tmp_path):
    from neuro_eval import readers
    path = tmp_path / "responses.pkl"
    _write_pickle(path, {"value": [1, 2]})
    first = readers.load_pickle(path)
    alias = tmp_path / "alias.pkl"
    alias.symlink_to(path)
    assert readers.load_pickle(alias) is first
    previous = path.stat()
    _write_pickle(path, {"value": [3, 4]})
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    assert readers.load_pickle(path) == {"value": [3, 4]}
    assert readers.load_pickle(path) is not first


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_encoded_stimuli_composite_transparency_over_scanner_grey(tmp_path, mode):
    from neuro_eval.readers import LazyEncodedHdf5Dict
    if mode == "P":
        im = Image.new("P", (2, 1), 0)
        im.putpalette([255, 0, 0] + [0] * 765)
        im.info["transparency"] = 0
    elif mode == "LA":
        im = Image.new(mode, (2, 1), (255, 0))
    else:
        im = Image.new(mode, (2, 1), (255, 0, 0, 0))
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    path = tmp_path / "images.h5"
    with h5py.File(path, "w") as file:
        images = file.create_dataset("images", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        images[0] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    stimuli = LazyEncodedHdf5Dict(str(path), [0])
    assert stimuli._file is None
    np.testing.assert_array_equal(stimuli["0"], np.full((1, 2, 3), 128, dtype=np.uint8))
    assert stimuli.__getstate__()["_file"] is None
    with pytest.raises(KeyError):
        stimuli[1]


def test_raw_stimuli_preserve_index_mapping_and_lazy_access(tmp_path):
    from neuro_eval.readers import LazyHdf5Dict
    path = tmp_path / "images.h5"
    _write_images(path)
    stimuli = LazyHdf5Dict(str(path), "imgBrick", [4, 1])
    assert stimuli._file is None
    assert stimuli.keys() == ["1", "4"]
    assert 4 in stimuli and len(stimuli) == 2
    np.testing.assert_array_equal(stimuli[4][0, 0], [80, 10, 0])
    with pytest.raises(KeyError):
        stimuli[2]


def test_reader_environment_is_anchored_to_relocated_repository(tmp_path, monkeypatch):
    from neuro_eval import readers
    relocated = tmp_path / "relocated"
    (relocated / "neuro_eval").mkdir(parents=True)
    (relocated / ".env").write_text("NSD_ROOT=/fixture/nsd\nLAION_DATA_DIR=/fixture/laion\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("NSD_ROOT=/wrong/cwd\nLAION_DATA_DIR=/wrong/cwd\n")
    monkeypatch.setattr(readers, "__file__", str(relocated / "neuro_eval/readers.py"))
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("NSD_ROOT", raising=False)
    monkeypatch.delenv("LAION_DATA_DIR", raising=False)
    assert readers.nsd_root() == "/fixture/nsd"
    assert readers.get_env_var("LAION_DATA_DIR") == "/fixture/laion"
    monkeypatch.setenv("LAION_DATA_DIR", "/shell/override")
    assert readers.get_env_var("LAION_DATA_DIR") == "/shell/override"


def test_nsd_reliability_reads_voxel_coordinates_and_rejects_invalid_values(tmp_path, monkeypatch):
    from neuro_eval import readers
    path = tmp_path / "ncsnr.nii.gz"
    volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    monkeypatch.setattr(readers, "voxel_ncsnr_path", lambda *_: str(path))
    response = _response([0, 1], x=[1, 0, 1], y=[0, 1, 1], z=[1, 0, 0])
    np.testing.assert_array_equal(readers.load_voxel_ncsnr(response, 0), [5, 2, 6])
    with pytest.raises(ValueError, match="lack voxel coordinate"):
        readers.load_voxel_ncsnr(response.drop_vars("z"), 0)
    with pytest.raises(ValueError, match="outside ncsnr volume"):
        readers.load_voxel_ncsnr(response.assign_coords(x=("neuroid", [2, 0, 1])), 0)
    volume[1, 0, 1] = np.nan
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
    with pytest.raises(ValueError, match="Non-finite ncsnr"):
        readers.load_voxel_ncsnr(response, 0)
