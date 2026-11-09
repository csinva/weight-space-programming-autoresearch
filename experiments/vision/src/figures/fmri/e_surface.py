"""What erjpu's surface-shape computation does (panel E of the fMRI figure).

erjpu's `_surface_geom_map` treats luminance as a height map. From its Hessian it computes the shape index
(2/pi) atan2(-trace, deviator) and five soft shape bins centred at -1, -0.5, 0, 0.5, 1, each weighted by
locally normalized curvedness (channels 0-4). By the sign convention (verified at startup on synthetic
luminance) a local brightness maximum is +1, so the bins are cup, rut, saddle, ridge, dome; they are stored
in the spec's order dome, ridge, saddle, rut, cup. The function is called from the frozen model file, not
reimplemented, on luminance preprocessed exactly as in erjpu's __call__, at 128 px, finest scale.

(i)   Example images: fixed THINGSplus images unique to search subjects (no shared or held-out images).
      Class = argmax over the five bins per pixel; opacity = summed bin weight / its 99th percentile in that
      image, clipped to [0, 1].
(ii)  Synthetic stimuli built from five height maps z(x, y) of equal curvature magnitude 1/RS: spherical cap
      (dome), inverted cap (cup), cylinder (ridge), trough (rut) and quadric saddle, the first four surrounded by a
      flat floor, the saddle held constant along rays beyond its radius. "luminance": luminance =
      0.5 + z * LUM_GAIN (bright blob, dark blob, bright bar, dark bar, saddle). "shaded": Lambertian shading
      of the same surface, light from the upper left, scaled so that a flat floor is mid-grey. Response = mean of each
      bin over the central disk of radius EVAL_RADIUS, where every surface has its intended shape class and no rim.
(iii) ROI contribution from panel D's per-voxel results (--d-dir): surface shape's share of the gain beyond
      edges (as d_families.write_panel_data: ROI voxel means of r2_beyond_edges_*, S over the sum of the five
      non-edge families) and its unique signed r^2 (ROI mean of sign(r) r^2 for all six families minus that
      without surface shape, from voxel_r_<subject>.npz).

Format in src/figures/fmri/spec.md.

Usage: python src/figures/fmri/e_surface.py [--out OUT_NPZ] [--d-dir D_DIR]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "artifacts/figures/fmri/data"
FROZEN = REPO / ("artifacts/final/laion-visual-encoding-to-nsd-encoding/"
                 "cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633/model.py")
ROIS = ["V1", "V2", "V3", "hV4", "FFA", "PPA"]
FAMILIES = "EKSCTG"
CLASSES = ["dome", "ridge", "saddle", "rut", "cup"]
BIN_CENTRES = [-1.0, -0.5, 0.0, 0.5, 1.0]            # erjpu's bin order (channels 0-4)
CLASS_CENTRE = {"dome": 1.0, "ridge": 0.5, "saddle": 0.0, "rut": -0.5, "cup": -1.0}
CLASS_BIN = [BIN_CENTRES.index(CLASS_CENTRE[c]) for c in CLASSES]   # [4, 3, 2, 1, 0]
SIZE = 128
# LAION stimulus indices (row of task-images_metadata.csv = index in task-images_stimuli.h5); THINGSplus
# images unique to search subjects, chosen for single objects with smooth shading.
EXAMPLE_IDS = [22685, 22986, 23081, 23260, 23486, 23868]   # bell, cup, face, helmet, mushroom, skull
OPACITY_PERCENTILE = 99.0

# Synthetic geometry (pixels; image centre at (63.5, 63.5), x to the right, y downwards).
RS = 60.0           # radius of curvature of every surface
R_SHAPE = 48.0      # radius (cap, saddle) or half-width (cylinder, trough) of the curved surface
EVAL_RADIUS = 24.0  # responses are averaged over this central disk
LUM_GAIN = 0.4 / 24.0          # luminance per pixel of height (the cap is 24 px high -> 0.9)
LIGHT = (-1.0, -1.0, 1.5)      # towards the light: upper left, in front
AMBIENT = 0.15
SURFACES = {  # luminance name, shaded name, surface
    "dome": ("bright blob", "sphere"),
    "cup": ("dark blob", "bowl"),
    "ridge": ("bright bar", "cylinder"),
    "rut": ("dark bar", "trough"),
    "saddle": ("saddle", "saddle"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_surface_geom_map():
    """erjpu's `_surface_geom_map`, executed from the frozen model file."""
    module = types.ModuleType("champion")
    exec(compile(FROZEN.read_text(), str(FROZEN), "exec"), module.__dict__)
    return module._surface_geom_map


def shape_bins(geom, rgb):
    """rgb (n, H, W, 3) in [0, 1] -> (n, 5, H, W) shape bins in CLASSES order, erjpu's preprocessing."""
    x = torch.as_tensor(np.asarray(rgb, dtype=np.float32)).permute(0, 3, 1, 2)
    x = x.clamp(0.0, 1.0)
    x_ret = (x + 0.04).pow(0.45)
    r, g, b = x_ret[:, 0:1], x_ret[:, 1:2], x_ret[:, 2:3]
    lum = (r + g + b) * (1.0 / math.sqrt(3.0))
    rg = (r - g) * (1.0 / math.sqrt(2.0))
    by = (r + g - 2.0 * b) * (1.0 / math.sqrt(6.0))
    with torch.no_grad():
        out = geom(lum, rg, by)
    assert out.shape[1] == 14, out.shape
    return out[:, CLASS_BIN].numpy()


def grid():
    c = (SIZE - 1) / 2.0
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    return x - c, y - c


def height(kind):
    """Height map z(x, y) in pixels; curvature magnitude 1/RS inside R_SHAPE, continuous beyond it."""
    x, y = grid()
    cap = lambda d: np.sqrt(RS ** 2 - np.minimum(d, R_SHAPE) ** 2) - math.sqrt(RS ** 2 - R_SHAPE ** 2)
    r = np.hypot(x, y)
    if kind == "dome":
        return cap(r)
    if kind == "cup":
        return -cap(r)
    if kind == "ridge":
        return cap(np.abs(x))
    if kind == "rut":
        return -cap(np.abs(x))
    if kind == "saddle":
        # (x^2 - y^2) / (2 RS) = r^2 cos(2 theta) / (2 RS); beyond R_SHAPE, constant along rays (continuous rim)
        return (x ** 2 - y ** 2) / (2 * RS) * (np.minimum(r, R_SHAPE) / np.maximum(r, 1e-9)) ** 2
    raise ValueError(kind)


def shade(z):
    """Lambertian shading of z, scaled so a flat floor facing the viewer is 0.5."""
    zy, zx = np.gradient(z)
    n = np.stack([-zx, -zy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=0)
    light = np.asarray(LIGHT) / np.linalg.norm(LIGHT)
    lam = lambda ndotl: AMBIENT + (1 - AMBIENT) * np.maximum(ndotl, 0.0)
    return 0.5 * lam(np.tensordot(light, n, axes=1)) / lam(light[2])


def synthetic():
    names, kinds, images, targets = [], [], [], []
    for kind, (lum_name, shaded_name) in SURFACES.items():
        z = height(kind)
        for stim_kind, name, img in (("luminance", lum_name, 0.5 + LUM_GAIN * z), ("shaded", shaded_name, shade(z))):
            assert 0.0 <= img.min() and img.max() <= 1.0, (name, img.min(), img.max())
            names.append(name)
            kinds.append(stim_kind)
            images.append(img.astype(np.float32))
            targets.append(kind)
    return names, kinds, np.stack(images), targets


def eval_mask():
    x, y = grid()
    return np.hypot(x, y) < EVAL_RADIUS


def check_sign_convention(geom):
    """Bright blob -> dome, dark blob -> cup, bright bar -> ridge, dark bar -> rut, x^2 - y^2 -> saddle."""
    mask = eval_mask()
    for kind in SURFACES:
        img = (0.5 + LUM_GAIN * height(kind)).astype(np.float32)
        bins = shape_bins(geom, np.repeat(img[None, :, :, None], 3, axis=3))[0]
        got = CLASSES[int(np.argmax(bins[:, mask].mean(1)))]
        frac = float((bins.argmax(0)[mask] == CLASSES.index(kind)).mean())
        assert got == kind and frac > 0.9, (kind, got, frac)


def examples(geom):
    sys.path.insert(0, str(REPO / "src"))
    from neuro_eval.readers import LazyEncodedHdf5Dict, laion_stimuli_path
    path = laion_stimuli_path()
    stimuli = LazyEncodedHdf5Dict(path, EXAMPLE_IDS)
    images = []
    for sid in EXAMPLE_IDS:  # same pixel pipeline as neuro_eval.datasets._pixels (rgb-pil-bilinear-hwc-v1)
        im = Image.fromarray(np.asarray(stimuli[str(sid)], dtype=np.uint8)).convert("RGB")
        im = im.resize((SIZE, SIZE), resample=Image.Resampling.BILINEAR)
        images.append(np.asarray(im, dtype=np.float32) / 255.0)
    images = np.stack(images)
    bins = shape_bins(geom, images)
    cls = bins.argmax(1).astype(np.int8)
    weight = bins.sum(1)
    scale = np.percentile(weight.reshape(len(weight), -1), OPACITY_PERCENTILE, axis=1)[:, None, None]
    curvedness = np.clip(weight / scale, 0.0, 1.0).astype(np.float32)
    meta = Path(path).parents[1] / "task-images_metadata.csv"
    names = {}
    if meta.exists():
        rows = meta.read_text().splitlines()[1:]
        names = {sid: rows[sid].split(",")[0] for sid in EXAMPLE_IDS}
    return images, cls, curvedness, {"path": path, "ids": EXAMPLE_IDS, "names": [names.get(i) for i in EXAMPLE_IDS]}


def roi_contribution(d_dir):
    subjects = sorted(p.name[len("voxel_r_"):-len(".npz")] for p in d_dir.glob("voxel_r_*.npz"))
    assert subjects, f"no voxel_r_*.npz in {d_dir}"
    signed = lambda r: np.sign(r) * r.astype(np.float64) ** 2
    rest = [p for p in FAMILIES if p != "E"]
    share = np.zeros((len(subjects), len(ROIS)))
    unique = np.zeros((len(subjects), len(ROIS)))
    for i, s in enumerate(subjects):
        r = np.load(d_dir / f"voxel_r_{s}.npz")
        phi = np.load(d_dir / f"shapley_{s}.npz")
        vox = np.load(d_dir / f"voxels_{s}.npz")
        loss = signed(r[FAMILIES]) - signed(r[FAMILIES.replace("S", "")])
        for j, roi in enumerate(ROIS):
            m = vox[f"mask_{roi}"].astype(bool)
            assert m.shape == loss.shape and m.any(), (s, roi)
            gains = {p: phi[f"r2_beyond_edges_{p}"][m].mean() for p in rest}
            share[i, j] = gains["S"] / sum(gains.values())
            unique[i, j] = loss[m].mean()
    return subjects, share.astype(np.float32), unique.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA / "e.npz")
    ap.add_argument("--d-dir", type=Path, default=DATA / "d")
    args = ap.parse_args()

    geom = load_surface_geom_map()
    check_sign_convention(geom)

    ex_images, ex_class, ex_curv, ex_meta = examples(geom)

    names, kinds, syn_images, targets = synthetic()
    bins = shape_bins(geom, np.repeat(syn_images[..., None], 3, axis=3))
    response = bins[:, :, eval_mask()].mean(2).astype(np.float32)
    for name, kind, target, row in zip(names, kinds, targets, response):
        print(f"{kind:9s} {name:12s} (surface {target:6s}) -> {CLASSES[int(row.argmax())]:6s} "
              + " ".join(f"{c}={v:.4f}" for c, v in zip(CLASSES, row)), flush=True)

    subjects, share, unique = roi_contribution(args.d_dir)
    for j, roi in enumerate(ROIS):
        print(f"{roi:4s} share beyond edges {share[:, j].mean():.3f}  unique r^2 {unique[:, j].mean():.5f}", flush=True)

    rel = lambda p: str(p.resolve().relative_to(REPO)) if p.resolve().is_relative_to(REPO) else str(p)
    provenance = {
        "model": rel(FROZEN), "model_sha256": sha256(FROZEN), "function": "_surface_geom_map (channels 0-4)",
        "class_to_bin": {c: {"channel": b, "shape_index_centre": BIN_CENTRES[b]} for c, b in zip(CLASSES, CLASS_BIN)},
        "sign_convention": "verified: bright blob=dome, dark blob=cup, bright bar=ridge, dark bar=rut, x^2-y^2=saddle",
        "preprocessing": "clamp [0,1]; x_ret=(x+0.04)^0.45; lum=(r+g+b)/sqrt3, rg=(r-g)/sqrt2, by=(r+g-2b)/sqrt6",
        "scale": "finest (128 px input, no pooling)",
        "examples": {"dataset": "laion", "stimuli_hdf5": ex_meta["path"], "ids": ex_meta["ids"],
                     "names": ex_meta["names"], "pixel_pipeline": "rgb-pil-bilinear-hwc-v1", "imsize": SIZE,
                     "selection": "fixed THINGSplus images unique to search subjects",
                     "class": "argmax of the five shape bins",
                     "curvedness": f"sum of the five shape bins / its per-image {OPACITY_PERCENTILE:g}th percentile, "
                                   "clipped to [0,1]"},
        "synthetic": {"grayscale_input": "R=G=B", "radius_of_curvature_px": RS, "shape_radius_px": R_SHAPE,
                      "surfaces": "spherical cap, inverted cap, cylinder, trough (|x| < radius, full height), "
                                  "quadric saddle (x^2-y^2)/(2 RS), constant along rays beyond the radius; "
                                  "flat floor outside the others",
                      "luminance": f"0.5 + {LUM_GAIN:.6f} * z", "light_direction": LIGHT, "ambient": AMBIENT,
                      "shading": "0.5 * (ambient + (1-ambient) max(n.l, 0)) / (ambient + (1-ambient) l_z)",
                      "response": f"mean of each shape bin over the central disk of radius {EVAL_RADIUS:g} px",
                      "target_surface": targets},
        "roi": {"d_dir": rel(args.d_dir), "subjects": subjects, "dataset": "laion", "stage": "search",
                "share_beyond_edges": "ROI voxel means of r2_beyond_edges_*, S / sum over K,S,C,T,G",
                "unique_r2": "ROI mean of sign(r) r^2 (EKSCTG) - sign(r) r^2 (EKCTG)"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, classes=np.array(CLASSES), example_images=ex_images.astype(np.float32),
             example_class=ex_class, example_curvedness=ex_curv,
             synthetic_names=np.array(names), synthetic_kind=np.array(kinds), synthetic_images=syn_images,
             synthetic_class_response=response, roi_names=np.array(ROIS), roi_share_beyond_edges=share,
             roi_unique_r2=unique, subjects=np.array(subjects), provenance=np.array(json.dumps(provenance)))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
