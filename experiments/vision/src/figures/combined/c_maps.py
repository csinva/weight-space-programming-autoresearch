"""Panel c data of the combined figure: one map per computational family of the searched model (erjpu).

For each example image, the maps are computed at the model's finest scale with its own functions:
  edges                 mean of the 16 normalized luminance Gabor energies
  color                 mean of the 8 normalized red-green and blue-yellow Gabor energies
  corners and contours  corner strength of the orientation field
  luminance curvature   saddle bin of the Hessian shape index
  texture statistics    mean second-order modulation energy (texture boundaries; the model applies this stage
                        one octave coarser, repeated here on the finest scale for display)
Global layout has no map: its features are symmetry, centre-versus-periphery and blob-count statistics.

Example images: the photograph of the model figure and a shared LAION image chosen by --shared (index into the
shared images of artifacts/figures/circuit/data/readout.npz).

Writes artifacts/figures/combined/data/c.npz. Usage: python src/figures/combined/c_maps.py [--shared I ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/figures/circuit"))
from common import intermediates, load_model  # noqa: E402

OUT = REPO / "artifacts/figures/combined/data"
KEYS = ["edges", "color", "corners", "curvature", "texture"]


@torch.no_grad()
def maps(g, x):
    m = intermediates(g, x)
    s1 = m["stage1"][0]
    ori = 0.5 * (s1[:, 0:8] + s1[:, 8:16])
    sel = F.avg_pool2d(torch.cat([ori, s1[:, 20:24], s1[:, 24:28]], dim=1), 2, 2)
    sel = sel - sel.mean(dim=(2, 3), keepdim=True)
    conv = F.conv2d(F.pad(sel, (2, 2, 2, 2), mode="reflect"), m["s2_filters"].repeat(16, 1, 1, 1), groups=16)
    conv = conv.reshape(len(x), 16, 9, sel.shape[2], sel.shape[3])
    energy = torch.sqrt(conv[:, :, 0:8:2] ** 2 + conv[:, :, 1:8:2] ** 2 + 1e-6)
    texture = F.interpolate(torch.log1p(energy).mean(dim=(1, 2))[:, None], size=x.shape[-1], mode="bilinear")[:, 0]
    return {"edges": s1[:, 0:16].mean(1), "color": s1[:, 20:28].mean(1), "corners": m["curv"][0][:, 5],
            "curvature": m["geom"][0][:, 2], "texture": texture}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=int, nargs="*", default=[])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, g = load_model()
    images = [np.load(REPO / "artifacts/figures/model/data.npz")["images"][0]]
    shared = np.load(REPO / "artifacts/figures/circuit/data/readout.npz")["images"]
    images += [shared[i].astype(np.float32) / 255 for i in args.shared]
    images = np.stack(images).astype(np.float32)
    out = maps(g, torch.from_numpy(images).permute(0, 3, 1, 2).clamp(0, 1))
    np.savez_compressed(OUT / "c.npz", images=images, shared=np.array(args.shared),
                        **{k: v.numpy() for k, v in out.items()})
    print("wrote", OUT / "c.npz", images.shape)


if __name__ == "__main__":
    main()
