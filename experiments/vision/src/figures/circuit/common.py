"""Shared helpers for the circuit figure: the frozen searched model (erjpu) and its intermediate maps."""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/figures/fmri"))
sys.path.insert(0, str(REPO / "src"))
from d_families import FROZEN, load_champion  # noqa: E402

DATA = REPO / "artifacts/figures/circuit/data"
AREAS = ["hV4", "FFA", "PPA"]


def load_model(grad=False):
    """The frozen model and its module globals. With grad=True the forward pass keeps gradients; the two
    global-layout blocks that cannot be differentiated (path maps and blob counts, 82 features) are detached."""
    model = load_champion()
    forward = type(model).__call__
    g = forward.__wrapped__.__globals__
    if not grad:
        return model, g
    for name in ("_persistence_features", "_path_maps", "_additive_path_maps"):
        g[name] = (lambda f: lambda t: f(t.detach()))(g[name])
    return (lambda x: forward.__wrapped__(model, x)), g


def intermediates(g, x):
    """Per-scale maps of the model on images x (B, 3, H, W) in [0, 1], computed with the model's own functions
    as in its forward pass (three scales: full, 1/2 and 1/4 resolution)."""
    x_ret = (x.clamp(0, 1) + 0.04).pow(0.45)
    r, gr, b = x_ret[:, 0:1], x_ret[:, 1:2], x_ret[:, 2:3]
    lum, rg, by = (r + gr + b) / math.sqrt(3.0), (r - gr) / math.sqrt(2.0), (r + gr - 2.0 * b) / math.sqrt(6.0)
    filters = g["_make_gabor_bank"](x.device, x.dtype)
    chrom = filters[[16, 17, 20, 21, 24, 25, 28, 29, 32, 33]]
    out = {"stage1": [], "curv": [], "phase": [], "geom": [], "filters": filters,
           "s2_filters": g["_make_stage2_bank"](x.device, x.dtype)}
    for scale in range(3):
        if scale:
            lum, rg, by = (F.avg_pool2d(t, 2, 2) for t in (lum, rg, by))
        norm_map, even, odd, energy, lum_ref = g["_stage1_filter"](lum, rg, by, filters, chrom)
        out["stage1"].append(norm_map)
        out["curv"].append(g["_nematic_curvature_map"](0.5 * (norm_map[:, 0:8] + norm_map[:, 8:16])))
        out["phase"].append(g["_phase_congruency_map"](even, odd, energy, lum_ref, norm_map))
        out["geom"].append(g["_surface_geom_map"](lum, rg, by))
    return out


def second_order(maps):
    """Stage 2A: oriented modulation energy of the 16 half-resolution energy maps.
    Returns (B, 16 input channels, 4 modulation orientations, h, w) before the model's log compression."""
    m1 = maps["stage1"][1]
    sel = torch.cat([0.5 * (m1[:, 0:8] + m1[:, 8:16]), m1[:, 20:24], m1[:, 24:28]], dim=1)
    sel = F.avg_pool2d(sel, 2, 2)
    sel = sel - sel.mean(dim=(2, 3), keepdim=True)
    conv = F.conv2d(F.pad(sel, (2, 2, 2, 2), mode="reflect"), maps["s2_filters"].repeat(16, 1, 1, 1), groups=16)
    conv = conv.reshape(len(sel), 16, 9, sel.shape[2], sel.shape[3])
    return torch.sqrt(conv[:, :, 0:8:2] ** 2 + conv[:, :, 1:8:2] ** 2 + 1e-6)


def channels24(smap):
    """Stage 2B input: 8 orientation energies, 4 luminance blobs, 8 color energies, 4 color blobs."""
    return torch.cat([0.5 * (smap[:, 0:8] + smap[:, 8:16]), smap[:, 16:20], smap[:, 20:28],
                      0.5 * (smap[:, 28:32] + smap[:, 32:36])], dim=1)


def correlation(a, b=None):
    """Spatial correlation between channels of a (B, C, H, W) and b, as the model's _upper_tri_gram/_cross_gram."""
    def z(t):
        t = t.flatten(2)
        t = t - t.mean(2, keepdim=True)
        return t / (t.std(2, keepdim=True) + 1e-5)
    za, zb = z(a), z(a if b is None else b)
    return za @ zb.transpose(1, 2) / za.shape[2]


def octave_summaries(g, maps):
    """Stage 2F inputs: the 12 summary channels of each scale."""
    return [g["_octave_summary12"](maps["stage1"][k], maps["curv"][k], maps["phase"][k], maps["geom"][k])
            for k in range(3)]
