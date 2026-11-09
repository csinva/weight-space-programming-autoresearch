"""Panel E data: which texture statistics of the searched model drive FFA and PPA.

Texture statistics carry about half of the gain beyond edges in FFA and PPA (panel D). This script splits
the texture-statistics columns into the model's five texture operations and attributes the gain to each,
per voxel and ROI, with the same readout and split as panel D (erjpu's search setting, protocol v4: all
five LAION subjects, readout fit on each subject's unique images, scored on all shared images).

  M  second-order modulation   oriented filters applied to V1 energy maps (texture boundaries, contrast modulation)
  V  V1 channel covariance     covariance between orientation, DoG and color channels, whole image and per quadrant
  J  joint covariance          covariance between curvature, phase, geometry and color channels
  O  cross-scale relations     persistence, difference and parent-child covariance across adjacent octaves
  R  geometry co-occurrence    co-occurrence of non-curvature geometry channels across scales

Two quantities per sub-statistic, on signed r^2 (sign(r) r^2):
  beyond edges: exact Shapley over the five sub-statistics in the game v(S) = score(E + S) - score(E)
  unique:       score(six families) - score(six families without this sub-statistic)

Reads cached features from data/d/ (run d_families.py first). Writes data/e/ (per-voxel r cache) and
data/e.json. Usage: python figure/e_texture.py [--d-dir PATH]
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d_families import (BLOCKS, D_ROIS, DATA, FROZEN, REPO, V4_SUBJECTS, block_labels, load_champion,  # noqa: E402
                        shapley, v4_selected_ids)

SUBSTATS = {
    "M": ("Second-order modulation", ["second_order_grid", "second_order_stats"],
          "oriented filters on V1 energy maps: texture boundaries and contrast modulation"),
    "V": ("V1 channel covariance", ["cov_within_scale", "cov_quadrant"],
          "which orientations, blob sizes and colors co-occur, over the image and per quadrant"),
    "J": ("Joint feature covariance", ["cov_joint"],
          "co-occurrence of curvature, phase, geometry and color channels"),
    "O": ("Cross-scale relations", ["octave_persist_diff", "octave_parent_child_cov"],
          "whether structure persists, changes or covaries from one octave to the next"),
    "R": ("Geometry co-occurrence", ["geom_rel_fine", "geom_comp_fine", "geom_rel_coarse"],
          "co-occurrence of structure-tensor and isophote-curvature channels across scales"),
}
OTHERS = "EKSCG"  # the five other families of panel D


def column_groups(model):
    lab = block_labels(dict(zip(BLOCKS, model._widths)))
    block = np.concatenate([[b] * w for b, w in zip(BLOCKS, model._widths)])
    sub = np.full(len(lab), "", dtype=object)
    for k, (_, blocks, _) in SUBSTATS.items():
        sub[(lab == "T") & np.isin(block, blocks)] = k
    assert ((lab == "T") == (sub != "")).all(), sorted(set(block[(lab == "T") & (sub == "")]))
    return lab, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-dir", type=Path, default=DATA / "d")
    ap.add_argument("--out", type=Path, default=DATA / "e.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.out.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(NEURO_DATASET="laion", NEURO_STAGE="search", NEURO_METRIC="encoding",
                      NEURO_REGION="visual", NEURO_IMSIZE="128")
    sys.path.insert(0, str(REPO))
    import neuro_eval.datasets as nd
    from neuro_eval.datasets import get_dataset, with_subject, load_split
    from neuro_eval.encoding import design, _fit_predict, pearson, ENCODING_FEATURES
    nd._selected_ids = v4_selected_ids

    device = torch.device(args.device)
    model = load_champion()
    with torch.no_grad():
        model(torch.rand(2, 128 * 128 * 3, device=device))
    lab, sub = column_groups(model)
    counts = {k: int((sub == k).sum()) for k in SUBSTATS}
    print("columns per sub-statistic:", counts, flush=True)

    players = list(SUBSTATS)
    subsets = [frozenset(c) for k in range(0, 6) for c in itertools.combinations(players, k)]
    name = lambda S: "E+" + "".join(p for p in players if p in S)
    base_other = np.isin(lab, list(OTHERS))
    fits = {name(S): (lab == "E") | np.isin(sub, list(S)) for S in subsets}
    fits.update({f"six-{p}": base_other | ((sub != "") & (sub != p)) for p in players})
    fits["six"] = base_other | (sub != "")

    base = get_dataset("laion", region="visual", metric="encoding")
    per_subject = {}
    for s in V4_SUBJECTS:
        d = with_subject(base, s)
        Xtr, Xte, Ytr, Yte, _, _ = load_split(d)
        A = torch.from_numpy(np.load(args.d_dir / f"features_{s}.npy"))
        assert A.shape == (len(Xtr) + len(Xte), len(lab)), A.shape
        masks = np.load(args.d_dir / f"voxels_{s}.npz")
        Y = torch.cat([Ytr, Yte])
        tr, te = np.arange(len(Xtr)), np.arange(len(Xtr), len(Xtr) + len(Xte))
        rpath = out / f"voxel_r_{s}.npz"
        cache = dict(np.load(rpath)) if rpath.exists() else {}
        t0 = time.time()
        for key, keep in fits.items():
            if key in cache:
                continue
            Z = design(A[:, torch.as_tensor(np.flatnonzero(keep))], tr, device, features=ENCODING_FEATURES)
            _, Yt, pred = _fit_predict(Z, Y, tr, te)
            cache[key] = pearson(Yt, pred).float().cpu().numpy()
            del Z, Yt, pred
            torch.cuda.empty_cache()
            np.savez(rpath, **cache)
            print(f"{s} {key:10s} pooled r={cache[key].mean():.4f} ({len(cache)}/{len(fits)}, {time.time() - t0:.0f}s)",
                  flush=True)
        r2 = {k: np.sign(v.astype(np.float64)) * v.astype(np.float64) ** 2 for k, v in cache.items()}
        v = {S: r2[name(S)] - r2["E+"] for S in subsets}
        phi = shapley(v, players)
        roi = {}
        for area in D_ROIS:
            m = masks[f"mask_{area}"].astype(bool)
            roi[area] = {"beyond_edges": {p: float(phi[p][m].mean()) for p in players},
                         "unique": {p: float((r2["six"] - r2[f"six-{p}"])[m].mean()) for p in players},
                         "texture_beyond_edges": float(v[frozenset(players)][m].mean())}
        per_subject[s] = roi

    def mean(f):
        return {p: {a: float(np.mean([f(per_subject[s][a], p) for s in V4_SUBJECTS])) for a in D_ROIS} for p in players}

    share = lambda x, p: x["beyond_edges"][p] / sum(x["beyond_edges"].values())
    data = {
        "rois": D_ROIS,
        "substats": {k: {"name": n, "blocks": b, "description": desc, "n_features": counts[k]}
                     for k, (n, b, desc) in SUBSTATS.items()},
        "share_beyond_edges": mean(share),
        "share_beyond_edges_subjects": {p: {a: {s: share(per_subject[s][a], p) for s in V4_SUBJECTS} for a in D_ROIS}
                                        for p in players},
        "gain_beyond_edges_r2": mean(lambda x, p: x["beyond_edges"][p]),
        "unique_r2": mean(lambda x, p: x["unique"][p]),
        "texture_gain_beyond_edges_r2": {a: float(np.mean([per_subject[s][a]["texture_beyond_edges"]
                                                           for s in V4_SUBJECTS])) for a in D_ROIS},
        "subjects": list(V4_SUBJECTS),
        "provenance": {"model": str(FROZEN.relative_to(REPO)), "dataset": "laion", "stage": "search",
                       "protocol": "looped-cross-dataset-v4 (erjpu search setting): all five subjects, all shared test images",
                       "metric": "encoding",
                       "value": "exact Shapley over texture sub-statistics in v(S) = r2(E+S) - r2(E), signed r^2, "
                                "ROI voxel mean then subject mean"},
    }
    args.out.write_text(json.dumps(data, indent=2))
    for p in players:
        print(f"{SUBSTATS[p][0]:26s}", {a: round(100 * data["share_beyond_edges"][p][a]) for a in D_ROIS})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
