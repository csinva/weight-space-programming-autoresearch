"""Panel a data: components of the searched model (erjpu) on one LAION image, and the area shares of Fig. 3e.

Writes artifacts/figures/circuit/data/a.npz: the example image (the first example of the model figure), four
first-stage kernels and the mean oriented-energy map, four second-order kernels and the texture-boundary
map (Stage 2A), the 24 x 24 channel correlation matrix (Stage 2B, half resolution), the 12 x 8 fine-to-coarse
correlation matrix (Stage 2F, full to half resolution), and each texture operation's share of the texture
gain beyond edges in hV4, FFA and PPA (artifacts/figures/fmri/data/e.json).

Usage: python src/figures/circuit/a_circuit.py
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import AREAS, DATA, REPO, channels24, correlation, intermediates, load_model, octave_summaries, second_order

OPS = ["M", "V", "O", "R", "J"]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    _, g = load_model()
    image = np.load(REPO / "artifacts/figures/model/data.npz")["images"][0]
    x = torch.from_numpy(image).permute(2, 0, 1)[None].float().clamp(0, 1)
    with torch.no_grad():
        m = intermediates(g, x)
        energy = m["stage1"][0][0, 0:16].mean(0)
        boundary = torch.log1p(second_order(m))[0].mean(dim=(0, 1))
        cov = correlation(channels24(m["stage1"][1]))[0]
        o = octave_summaries(g, m)
        env = torch.cat([o[1][:, 4:8], m["curv"][1][:, [2, 5, 10, 12]]], dim=1)
        cross = correlation(F.avg_pool2d(o[0], 2, 2), env)[0]
    e = json.loads((REPO / "artifacts/figures/fmri/data/e.json").read_text())
    shares = np.array([[e["share_beyond_edges"][k][a] for a in AREAS] for k in OPS])
    np.savez_compressed(DATA / "a.npz", image=image, energy=energy.numpy(), boundary=boundary.numpy(),
                        gabors=m["filters"][[0, 8, 16, 24], 0].numpy(), s2_kernels=m["s2_filters"][[0, 2, 4, 6], 0].numpy(),
                        cov=cov.numpy(), cross=cross.numpy(), shares=shares, ops=OPS, areas=AREAS,
                        op_names=[e["substats"][k]["name"] for k in OPS])
    print("shares (rows M V O R J; columns hV4 FFA PPA)\n", shares.round(3))


if __name__ == "__main__":
    main()
