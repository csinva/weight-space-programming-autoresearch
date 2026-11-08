"""Pooled early + ventral ("visual") search region and the new search defaults."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from neuro_eval import benchmarks
from neuro_eval.benchmarks import SEARCH_REGIONS, default_search_region, protocol_for
from neuro_eval.datasets import PRESETS, _SPLIT_MEMO, get_dataset, load_split, with_subject
from neuro_eval._evaluation import selected_masks
from neuro_eval.encoding import encoding_scores

from test_datasets import _response, _write_images, _write_pickle

UNION = SEARCH_REGIONS["visual"]
EARLY, VENTRAL = "early visual stream", "ventral visual stream"


@pytest.fixture(autouse=True)
def isolated_loader(monkeypatch, tmp_path):
    for key in ("NEURO_SUBJECT", "NEURO_REGION", "NEURO_METRIC", "NEURO_STAGE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEURO_NO_CACHE", "1")
    monkeypatch.setenv("NEURO_SPLIT_CACHE", str(tmp_path / "cache"))
    _SPLIT_MEMO.clear()


def test_laion_defaults_to_pooled_visual_encoding_and_triplen_to_ventral():
    d = get_dataset("laion")
    assert (d.search_region, d.metric, d.reward_roi) == ("visual", "encoding", UNION)
    assert default_search_region("laion") == "visual"
    assert default_search_region("nsd") == "visual"
    t = get_dataset("triplen")
    assert (t.search_region, t.metric) == ("ventral", "encoding")
    visual = protocol_for("triplen", "visual", "encoding")
    assert (visual.name, visual.search_subjects) == ("triplen-visual-encoding-to-tvsd-encoding", ("M1", "M4"))
    assert protocol_for("laion", "visual", "encoding").name == "laion-visual-encoding-to-laion-encoding"


def test_launcher_default_region_and_metric(monkeypatch):
    from search import run
    monkeypatch.setattr(run, "preflight", lambda agents: None, raising=False)
    seen = {}
    monkeypatch.setattr("builtins.print", lambda *a, **k: seen.setdefault("lines", []).append(" ".join(map(str, a))))
    # main() writes the resolved contract into os.environ; keep it test-local.
    monkeypatch.setenv("NEURO_REGION", "")
    monkeypatch.setenv("NEURO_METRIC", "unset")
    monkeypatch.delenv("NEURO_METRIC")  # restored to absent at teardown
    assert run.main(["--show-config"]) == 0
    assert "configuration: region=visual dataset=laion metric=encoding" in seen["lines"][0]
    seen.clear()
    monkeypatch.setenv("NEURO_REGION", "")
    assert run.main(["--dataset", "triplen", "--show-config"]) == 0
    assert "region=ventral dataset=triplen metric=encoding" in seen["lines"][0]


def _write_laion(root, coordinates):
    """Distinct x coordinates per anatomical block; each block has three voxels."""
    from neuro_eval.readers import _LAION_REGION_MAP
    data = {}
    for name, key in _LAION_REGION_MAP.items():
        x = coordinates.get(name, [200, 201, 202])  # unrelated areas
        data[key] = {0: _response([4, 0, 2, 1, 3], ncsnr=[0.5, 0.5, 0.5], x=x, y=[0, 0, 0], z=[0, 0, 0])}
    source = {
        "data": data, "voxel_counts": {key: {0: (3, 3)} for key in _LAION_REGION_MAP.values()},
        "ncsnr_threshold": 0.2, "shared_ids": {1, 2, 4},
        "stimuli": pd.DataFrame({"dataset": ["NSD", "LAION", "NSD", "LAION", "OOD"]}),
    }
    _write_pickle(root / "processed/laion_data.pkl", source)
    _write_images(root / "stimuli/images/task-images_stimuli.h5", encoded=True)


def _visual_dataset():
    d = with_subject(PRESETS["laion"], "sub-01")
    # Each fixture source has one shared image, so all three fall in the held-out half.
    return replace(d, imsize=2, search_region="visual", search_metric="encoding",
                   metric="encoding", reward_roi=UNION, images="heldout")


COORDS = {EARLY: [0, 1, 2], VENTRAL: [10, 11, 12],
          "V1": [0, 1, 99], "V2": [1, 98, 97], "V3": [2, 96, 95],
          "hV4": [10, 94, 93], "FFA": [11, 12, 92], "PPA": [12, 91, 90]}


def test_union_pools_every_unit_of_both_streams_with_stream_and_area_diagnostics(tmp_path, monkeypatch):
    root = tmp_path / "laion"
    monkeypatch.setenv("LAION_DATA_DIR", str(root / "processed"))
    monkeypatch.setenv("LAION_FMRI_DIR", str(root))
    _write_laion(root, COORDS)
    d = _visual_dataset()
    Xtr, Xte, Ytr, Yte, masks, _ = load_split(d)
    assert Ytr.shape[1] == Yte.shape[1] == 6  # three early + three ventral voxels, each once
    np.testing.assert_array_equal(masks[UNION], np.ones(6, dtype=bool))
    np.testing.assert_array_equal(masks[EARLY], [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(masks[VENTRAL], [0, 0, 0, 1, 1, 1])
    # Areas intersect the pooled units by coordinate, within their own stream.
    np.testing.assert_array_equal(masks["V1"], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(masks["FFA"], [0, 0, 0, 0, 1, 1])
    selected = selected_masks(masks, d)
    assert set(selected) == {UNION, EARLY, VENTRAL, "V1", "V2", "V3", "hV4", "FFA", "PPA"}


def test_union_rejects_streams_that_share_units(tmp_path, monkeypatch):
    root = tmp_path / "laion"
    monkeypatch.setenv("LAION_DATA_DIR", str(root / "processed"))
    monkeypatch.setenv("LAION_FMRI_DIR", str(root))
    _write_laion(root, {**COORDS, VENTRAL: [2, 11, 12]})
    with pytest.raises(ValueError, match="share units"):
        load_split(_visual_dataset())


def test_union_score_is_the_voxel_pooled_mean_not_a_stream_average():
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(60, 4, generator=generator)
    # Two well-predicted early voxels, four poorly predicted ventral voxels.
    signal = features @ torch.randn(4, 6, generator=generator)
    noise = torch.randn(60, 6, generator=generator) * torch.tensor([0.1, 0.1, 3, 3, 3, 3])
    targets = signal + noise
    early = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    masks = {UNION: np.ones(6, dtype=bool), EARLY: early, VENTRAL: ~early}
    scored = encoding_scores(features[:40], features[40:], targets[:40], targets[40:], masks, torch.device("cpu"))
    r = scored["roi_corr"]
    pooled = (2 * r[EARLY]["test"] + 4 * r[VENTRAL]["test"]) / 6
    assert r[UNION]["test"] == pytest.approx(pooled)
    assert r[UNION]["test"] != pytest.approx((r[EARLY]["test"] + r[VENTRAL]["test"]) / 2)


def test_union_components_cover_the_final_regions():
    assert benchmarks.UNION_ROIS[UNION] == tuple(benchmarks.REGIONS["nsd"])
    assert benchmarks.UNION_ROIS[UNION] == tuple(benchmarks.REGIONS["tvsd"])
