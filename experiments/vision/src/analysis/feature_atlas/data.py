"""Data for the feature atlas of the LAION searched model (erjpu): what its 9,996 features respond to.

Writes artifacts/analysis/feature_atlas/: atlas.json (per-feature metadata, map position, top images,
grating tuning, single-feature brain relevance), thumbs.jpg (sprite of every image that is a top image
of some feature), maps.json + maps.png (intermediate channel maps of the model on six example images).

- Images: sub-01's LAION images in erjpu's search setting (protocol v4: unique + shared), 128 px; features
  are the cached ones from src/figures/fmri/d_families.py (run it first).
- Map: t-SNE of features, each described by its z-scored responses over all images (PCA to 50 dims).
- Brain relevance: per voxel squared correlation between one feature and the voxel response on the shared
  (test) images, averaged over the ROI's voxels and then over the five subjects. Univariate, for intuition.
- Tuning: responses to sinusoidal gratings (12 orientations x 6 spatial frequencies), z-scored against the
  feature's distribution over natural images.

Usage: python src/analysis/feature_atlas/data.py
"""
import io
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/figures/fmri"))
sys.path.insert(0, str(REPO / "src"))
from d_families import (BLOCKS, C30, FROZEN, GMAP, ORI13, PH9, V4_SUBJECTS, block_labels,  # noqa: E402
                        v4_selected_ids)

OUT = REPO / "artifacts/analysis/feature_atlas"
FEAT = REPO / "artifacts/figures/fmri/data/d"
ROIS = ["V1", "hV4", "FFA", "PPA"]
FAMILY = {"E": "Edges", "K": "Color", "S": "Luminance curvature", "C": "Corners and contours",
          "T": "Texture statistics", "G": "Global layout", "X": "Mixed"}

N30 = ([f"luminance energy, fine, {o * 22.5:g} deg" for o in range(8)]
       + [f"luminance energy, coarse, {o * 22.5:g} deg" for o in range(8)]
       + ["luminance blob, bright", "luminance blob, dark"]
       + [f"red-green energy, {o * 45:g} deg" for o in range(4)]
       + [f"blue-yellow energy, {o * 45:g} deg" for o in range(4)]
       + ["red blob", "green blob", "yellow blob", "blue blob"])
NORI = ["orientation cos 2θ", "orientation sin 2θ", "orientation strength (2nd harmonic)",
        "orientation cos 4θ", "orientation sin 4θ", "corner/junction strength (4th harmonic)",
        "contour curvature +x", "contour curvature −x", "contour curvature +y", "contour curvature −y",
        "curvature magnitude", "iso-orientation surround", "cross-orientation surround (pop-out)"]
NPH = ["bright line (ridge)", "dark line (crease)", "line vs edge", "phase coherence", "fine vs coarse energy",
       "luminance gradient +x", "luminance gradient −x", "luminance gradient +y", "luminance gradient −y"]
NGEOM = ["dark spot (cup)", "dark line (rut)", "saddle", "bright line (ridge)", "bright spot (dome)",
         "corner strength (structure tensor min)", "edge strength (structure tensor max)", "edge coherence",
         "isophote curving one way", "isophote curving other way", "line anisotropy |Lxx-Lyy|",
         "diagonal anisotropy |2Lxy|", "red-green 2D structure", "blue-yellow 2D structure"]

BLOCK_INFO = {
    "path": "Global layout: coarse paths of contrast across a 32 px image (where large structures run)",
    "persistence": "Global layout: counts of light and dark extrema that persist across thresholds",
    "s1_cort_mean_o0": "V1-like energies, fine scale, averaged on a 9x9 grid (foveally magnified)",
    "s1_std_peak_o0": "V1-like energies, fine scale: local variability and peaks on a 5x5 grid",
    "s1_cort_mean_o1": "V1-like energies, middle scale, averaged on a 6x6 grid",
    "s1_std_peak_o1": "V1-like energies, middle scale: local variability and peaks on a 3x3 grid",
    "s1_cort_mean_o2": "V1-like energies, coarse scale, averaged on a 4x4 grid",
    "s1_std_peak_o2": "V1-like energies, coarse scale: variability and peaks on a 2x2 grid",
    "s1_global_stats": "V1-like energies, coarse scale: whole-image mean and spread",
    "color_grid": "Raw luminance, red-green and blue-yellow on a 6x6 grid",
    "second_order_grid": "Second-order filters on V1 energy maps (texture boundaries), 3x3 grid",
    "second_order_stats": "Second-order filters on V1 energy maps: whole-image mean and spread",
    "cov_within_scale": "Which V1 channels co-occur across the image (covariance, two scales)",
    "cov_quadrant": "Which V1 channels co-occur within each image quadrant",
    "orient_field_o0": "Orientation field and contour curvature, fine scale, 4x4 grid",
    "orient_field_o1": "Orientation field and contour curvature, middle scale, 4x4 grid",
    "orient_field_o2": "Orientation field and contour curvature, coarse scale: 2x2 grid, mean, spread",
    "phase_o0": "Phase: lines vs edges, polarity and gradient sign, fine scale, 4x4 grid",
    "phase_o1": "Phase: lines vs edges, polarity and gradient sign, middle scale, 4x4 grid",
    "phase_o2": "Phase: lines vs edges, polarity and gradient sign, coarse scale: 2x2, mean, spread",
    "cov_joint": "Co-occurrence of curvature, phase, geometry and color channels",
    "octave_persist_diff": "Whether structure persists or changes from one octave to the next",
    "octave_parent_child_cov": "Covariance of fine-scale channels with their coarse-scale parents",
    "ecc_symmetry": "Left-right symmetry and centre-periphery differences of summary channels",
    "geom_o0_mean": "Luminance curvature and structure tensor, finest scale, mean on 4x4 grid",
    "geom_o0_std": "Luminance curvature and structure tensor, finest scale, spread on 3x3 grid",
    "geom_o1_mean": "Luminance curvature and structure tensor, scale 2, mean on 4x4 grid",
    "geom_o1_std": "Luminance curvature and structure tensor, scale 2, spread on 3x3 grid",
    "geom_o2_mean": "Luminance curvature and structure tensor, scale 3, mean on 3x3 grid",
    "geom_o2_std": "Luminance curvature and structure tensor, scale 3, spread, mean and std",
    "geom_o3": "Luminance curvature and structure tensor, scale 4: 2x2 grid, mean, spread",
    "geom_o4": "Luminance curvature and structure tensor, scale 5: mean and spread",
    "geom_rel_fine": "Co-occurrence of geometry channels across fine scales",
    "geom_comp_fine": "Co-occurrence of shape-index components across fine scales",
    "geom_rel_coarse": "Co-occurrence of geometry channels across coarse scales",
    "geom_symmetry": "Left-right symmetry of geometry channels",
}


def channel_names(widths):
    """Readable name for every column: channel and spatial cell where the block is a per-channel grid."""
    def grid(names, cells, side=None):
        side = side or int(round(math.sqrt(cells)))
        if side * side == cells and cells > 1:
            return [f"{n} · cell {c // side + 1},{c % side + 1} of {side}x{side}" for n in names for c in range(cells)]
        return [f"{n}" for n in names for _ in range(cells)]

    def tail(names):  # 2x2 grid then global mean and std
        return grid(names, 4) + [f"{n} · mean" for n in names] + [f"{n} · spread" for n in names]

    N = {}
    for b, cells in (("s1_cort_mean_o0", 81), ("s1_std_peak_o0", 25), ("s1_cort_mean_o1", 36),
                     ("s1_std_peak_o1", 9), ("s1_cort_mean_o2", 16), ("s1_std_peak_o2", 4)):
        N[b] = grid(N30, cells)
    N["s1_global_stats"] = [f"{n} · mean" for n in N30] + [f"{n} · spread" for n in N30]
    N["color_grid"] = grid(["luminance", "red-green", "blue-yellow"], 36)
    for pre, names in (("orient_field", NORI), ("phase", NPH)):
        N[f"{pre}_o0"], N[f"{pre}_o1"] = grid(names, 16), grid(names, 16)
        N[f"{pre}_o2"] = tail(names)
    for b, cells in (("geom_o0_mean", 16), ("geom_o0_std", 9), ("geom_o1_mean", 16), ("geom_o1_std", 9),
                     ("geom_o2_mean", 9)):
        N[b] = grid([n + (" · spread" if "std" in b else "") for n in NGEOM], cells)
    N["geom_o2_std"] = grid([n + " · spread" for n in NGEOM], 4) + [f"{n} · mean" for n in NGEOM] + \
        [f"{n} · spread" for n in NGEOM]
    N["geom_o3"] = tail(NGEOM)
    N["geom_o4"] = [f"{n} · mean" for n in NGEOM] + [f"{n} · spread" for n in NGEOM]
    names = []
    for b, w in zip(BLOCKS, widths):
        own = N.get(b)
        if own is None or len(own) != w:
            own = [f"component {i + 1}" for i in range(w)]
        names += own
    return names


def load_module(grad=False):
    source = FROZEN.read_text()
    marker = "            out_chunks.append(torch.cat(feats, dim=1))"
    source = source.replace(marker, "            self._widths = [f.shape[1] for f in feats]\n" + marker)
    if grad:
        source = source.replace("    @torch.no_grad()\n    def __call__", "    def __call__")
    module = types.ModuleType("champion")
    exec(compile(source, str(FROZEN), "exec"), module.__dict__)
    return module


def gratings(n_ori=12, sfs=(2, 4, 8, 16, 32, 48), size=128):
    yy, xx = np.mgrid[0:size, 0:size] / size
    imgs, meta = [], []
    for f in sfs:
        for k in range(n_ori):
            th = math.pi * k / n_ori
            g = 0.5 + 0.4 * np.sin(2 * math.pi * f * (xx * math.cos(th) + yy * math.sin(th)))
            imgs.append(np.repeat(g[..., None], 3, axis=2))
            meta.append((f, k * 180 / n_ori))
    return np.stack(imgs).astype(np.float32), meta


def jpeg_sprite(images, size, cols):
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * size, rows * size), (128, 128, 128))
    for i, im in enumerate(images):
        tile = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
        sheet.paste(tile, ((i % cols) * size, (i // cols) * size))
    return sheet


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(NEURO_DATASET="laion", NEURO_STAGE="search", NEURO_METRIC="encoding",
                      NEURO_REGION="visual", NEURO_IMSIZE="128")
    import neuro_eval.datasets as nd
    from neuro_eval.datasets import get_dataset, with_subject, load_split
    nd._selected_ids = v4_selected_ids
    dev = torch.device("cuda")

    module = load_module()
    model = module.MODEL
    with torch.no_grad():
        model(torch.rand(2, 128 * 128 * 3, device=dev))
    widths = model._widths
    lab = block_labels(dict(zip(BLOCKS, widths)))
    names = channel_names(widths)
    block_of = np.concatenate([[i] * w for i, w in enumerate(widths)])
    assert len(names) == len(lab) == 9996

    # Images and features (sub-01, train + test, in the cached order).
    base = get_dataset("laion", region="visual", metric="encoding")
    Xtr, Xte, *_ = load_split(with_subject(base, "sub-01"))
    X = torch.cat([Xtr, Xte]).reshape(-1, 128, 128, 3).numpy()
    A = torch.from_numpy(np.load(FEAT / "features_sub-01.npy")).to(dev)
    assert A.shape[0] == len(X)
    mu, sd = A.mean(0), A.std(0).clamp_min(1e-8)
    Z = (A - mu) / sd
    top = torch.topk(Z, 9, dim=0).indices.T.cpu().numpy()            # (9996, 9)
    print("top images done", flush=True)

    # Map of features by response profile.
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    P = PCA(50, random_state=0).fit_transform(Z.T.cpu().numpy())
    xy = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(P)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    print("t-SNE done", flush=True)

    # Single-feature brain relevance, averaged over subjects.
    rel = {r: torch.zeros(len(lab), device=dev) for r in ROIS}
    for s in V4_SUBJECTS:
        Xtr_s, Xte_s, Ytr, Yte, *_ = load_split(with_subject(base, s))
        F_s = torch.from_numpy(np.load(FEAT / f"features_{s}.npy"))[len(Xtr_s):].to(dev)
        Yt = Yte.to(dev).float()
        Fz = (F_s - F_s.mean(0)) / F_s.std(0).clamp_min(1e-8)
        Yz = (Yt - Yt.mean(0)) / Yt.std(0).clamp_min(1e-8)
        R = (Fz.T @ Yz) / (len(Fz) - 1)                                  # (9996, voxels)
        masks = np.load(FEAT / f"voxels_{s}.npz")
        for r in ROIS:
            m = torch.as_tensor(masks[f"mask_{r}"].astype(bool), device=dev)
            rel[r] += (R[:, m] ** 2).mean(1) / len(V4_SUBJECTS)
        del R
        print("relevance", s, flush=True)

    # Grating tuning.
    G, gmeta = gratings()
    with torch.no_grad():
        RG = model(torch.from_numpy(G.reshape(len(G), -1)).to(dev))
    RG = ((RG - mu) / sd).cpu().numpy().reshape(6, 12, -1)              # (sf, ori, feature)
    ori_tune = RG.max(0).T                                               # (feature, 12)
    sf_tune = RG.max(1).T                                                # (feature, 6)

    # Thumbnails of every image that is a top image of some feature.
    used = np.unique(top)
    remap = {int(i): k for k, i in enumerate(used)}
    jpeg_sprite(X[used], 48, 64).save(OUT / "thumbs.jpg", quality=82)
    print("thumbnails:", len(used), flush=True)

    # Intermediate channel maps on six example images (scale 0, the model's own preprocessing).
    examples = [int(i) for i in np.linspace(len(Xtr) + 5, len(X) - 5, 6)]
    x = torch.from_numpy(X[examples]).permute(0, 3, 1, 2).to(dev).clamp(0, 1)
    with torch.no_grad():
        x_ret = (x + 0.04).pow(0.45)
        r, g, b = x_ret[:, 0:1], x_ret[:, 1:2], x_ret[:, 2:3]
        lum = (r + g + b) / math.sqrt(3.0)
        rg = (r - g) / math.sqrt(2.0)
        by = (r + g - 2.0 * b) / math.sqrt(6.0)
        filters = module._make_gabor_bank(dev, torch.float32)
        chrom = filters[[16, 17, 20, 21, 24, 25, 28, 29, 32, 33]]
        norm_map, even_l, odd_l, energy_l, lum_ref = module._stage1_filter(lum, rg, by, filters, chrom)
        groups = {
            "V1-like energies": (norm_map, [N30[i] for i in range(16)] + ["luminance blob, bright (small)",
                                 "luminance blob, bright (large)", "luminance blob, dark (small)",
                                 "luminance blob, dark (large)"] + N30[18:26] +
                                 ["red blob (small)", "red blob (large)", "green blob (small)", "green blob (large)",
                                  "yellow blob (small)", "yellow blob (large)", "blue blob (small)", "blue blob (large)"]),
            "Orientation field and curvature": (module._nematic_curvature_map(
                0.5 * (norm_map[:, 0:8] + norm_map[:, 8:16])), NORI),
            "Phase": (module._phase_congruency_map(even_l, odd_l, energy_l, lum_ref, norm_map), NPH),
            "Luminance curvature and structure tensor": (module._surface_geom_map(lum, rg, by), NGEOM),
        }
    tiles, chans = [], []
    for gname, (fmap, cn) in groups.items():
        fmap = fmap.float().cpu().numpy()
        assert fmap.shape[1] == len(cn), (gname, fmap.shape, len(cn))
        for c in range(fmap.shape[1]):
            m = fmap[:, c]
            lo, hi = np.percentile(m, 1), np.percentile(m, 99)
            chans.append({"group": gname, "name": cn[c], "row": len(chans)})
            for k in range(len(examples)):
                t = np.clip((m[k] - lo) / (hi - lo + 1e-9), 0, 1)
                tiles.append(np.repeat(t[..., None], 3, axis=2))
    jpeg_sprite(tiles, 56, len(examples)).save(OUT / "maps.jpg", quality=85)
    jpeg_sprite(list(X[examples]), 56, len(examples)).save(OUT / "map_images.jpg", quality=88)
    (OUT / "maps.json").write_text(json.dumps({"n_images": len(examples), "tile": 56, "channels": chans}))

    rel_np = {r: rel[r].cpu().numpy() for r in ROIS}
    atlas = {
        "n_features": len(lab), "n_images": int(len(X)), "thumb": 48, "thumb_cols": 64,
        "families": FAMILY, "rois": ROIS,
        "blocks": [{"name": b, "info": BLOCK_INFO.get(b, b), "width": int(w)} for b, w in zip(BLOCKS, widths)],
        "grating": {"orientations_deg": [k * 15 for k in range(12)], "cycles_per_image": [2, 4, 8, 16, 32, 48]},
        "rel_max": {r: float(rel_np[r].max()) for r in ROIS},
        "features": [
            [round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4), str(lab[i]), int(block_of[i]), names[i],
             [remap[int(j)] for j in top[i]],
             [round(float(rel_np[r][i]), 5) for r in ROIS],
             [round(float(v), 2) for v in ori_tune[i]], [round(float(v), 2) for v in sf_tune[i]]]
            for i in range(len(lab))],
        "fields": ["x", "y", "family", "block", "name", "top_images", "relevance", "orientation_tuning", "sf_tuning"],
        "provenance": {"model": str(FROZEN.relative_to(REPO)), "images": "LAION sub-01, protocol v4 search split",
                       "relevance": "mean over ROI voxels of squared single-feature correlation on shared images, "
                                    "mean over 5 subjects"},
    }
    (OUT / "atlas.json").write_text(json.dumps(atlas, separators=(",", ":")))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
