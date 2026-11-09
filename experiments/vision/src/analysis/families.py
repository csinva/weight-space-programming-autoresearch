"""Per-voxel Shapley attribution of the LAION champion erjpu over six computational families.

Every one of erjpu's 9,996 feature columns gets one label by tracing it to the intermediate
channels it is computed from (see erjpu's model.py):

  E  edges            light/dark Gabor energy and DoG channels of the V1-like stage, luminance layout
  K  color            red-green / blue-yellow energy and DoG channels of the V1-like stage, color layout
  S  surface shape    Hessian shape index (dome, ridge, saddle, rut, cup), curvedness and anisotropy
                      channels, and anything computed only from them (co-occurrence, symmetry, cross-scale)
  C  corners/contours structure tensor (corners, edge strength, coherence), isophote and orientation-field
                      curvature, surround pop-out, phase (line vs edge)
  T  texture stats    co-occurrence and cross-scale relations of non-surface channels, second-order edges
  G  global layout    symmetry and centre-vs-periphery of non-surface channels, bright/dark blob count,
                      color similarity and boundary crossings from corner/centre anchors (path maps)
  X  mixed            co-occurrence of a surface channel with a non-surface channel -> never used

Rules: a block computed from one family's channels takes that family; a column summarising one channel
takes that channel's family; a co-occurrence column takes S if both channels are surface, X if exactly one
is, and T otherwise.

For all 2^6 = 64 subsets of {E, K, S, C, T, G} the search encoding readout (center design, per-voxel
5-fold-CV ridge) is refit on the current LAION search split and scored per voxel on the test images.
Exact Shapley values per voxel are computed for two games, both from the same 64 fits:
  full:         v(S) = score(S), all six families compete, v(empty) = 0
  beyond_edges: v(S) = score(S + E) - score(E), the five non-edge families share the gain over edges
Scores are kept as Pearson r and as signed explained variance sign(r) r^2.

Usage: python src/analysis/families.py OUTDIR
"""
import itertools
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / ("artifacts/final/laion-visual-encoding-to-nsd-encoding/"
                 "cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633/model.py")
FAMILIES = {"E": "edges", "K": "color", "S": "surface shape", "C": "corners and contours",
            "T": "texture statistics", "G": "global layout"}

BLOCKS = ["path", "persistence",
          "s1_cort_mean_o0", "s1_std_peak_o0", "s1_cort_mean_o1", "s1_std_peak_o1",
          "s1_cort_mean_o2", "s1_std_peak_o2", "s1_global_stats",
          "color_grid", "second_order_grid", "second_order_stats",
          "cov_within_scale", "cov_quadrant",
          "orient_field_o0", "orient_field_o1", "orient_field_o2",
          "phase_o0", "phase_o1", "phase_o2",
          "cov_joint", "octave_persist_diff", "octave_parent_child_cov", "ecc_symmetry",
          "geom_o0_mean", "geom_o0_std", "geom_o1_mean", "geom_o1_std",
          "geom_o2_mean", "geom_o2_std", "geom_o3", "geom_o4",
          "geom_rel_fine", "geom_comp_fine", "geom_rel_coarse", "geom_symmetry"]

# Channel families of erjpu's intermediate maps.
C30 = ["E"] * 18 + ["K"] * 12  # 16 luminance Gabor energies + 2 luminance DoG, 8 RG/BY energies + 4 RG/BY DoG
GMAP = list("SSSSS") + ["C", "C", "C", "C", "C", "S", "S", "C", "C"]  # 5 shape bins, mu_min, mu_max,
#                                                     coherence, k+, k-, |Lxx-Lyy|, |2Lxy|, chromatic i2D rg/by
G6 = list("SSSCCC")    # convex (ridge+dome), concave (cup+rut), saddle, mu_min, k+, k-
GC6 = list("SSCCSS")   # dome-ridge, cup-rut, mu_max, coherence, |Lxx-Lyy|, |2Lxy|
OCT12 = list("EEEEE") + ["S"] + list("KK") + list("CCCC")  # ori4, dog, convex, rg, by, m2, |curv|, mu_min, phase coh
ENV8 = ["E", "S", "K", "K", "C", "C", "C", "C"]            # dog, convex, rg, by, m2, m4, |curv|, e_ortho
SYM9 = list("EEE") + ["S"] + list("KK") + list("CCC")      # h0, hv, dog, convex, rg, by, m2, |curv|, phase coh
CSC24 = list("CCCCCCC") + G6 + list("CCCCC") + list("EEKKKK")  # curvature 7, geometry 6, phase 5, DoG/color 6
GIN12 = G6 + GC6
GIN11 = G6[:4] + ["C"] + GC6


def pair(a, b):
    """Co-occurrence of two channels: pure surface -> S, surface with anything else -> X, else texture."""
    if a == "S" and b == "S":
        return "S"
    return "X" if "S" in (a, b) else "T"


def single(a, other):
    """A statistic of one channel (cross-scale, symmetry): surface -> S, else `other`."""
    return "S" if a == "S" else other


def csc_pairs():
    """Exact channel pairs behind the cov_joint columns (index logic copied from erjpu)."""
    ti, tj = torch.triu_indices(24, 24, offset=1)
    in_iso = (ti == 13) | (ti == 14) | (ti == 18) | (ti == 19)
    jn_iso = (tj == 13) | (tj == 14) | (tj == 18) | (tj == 19)
    in_g4 = (ti == 7) | (ti == 8) | (ti == 11) | (ti == 12)
    jn_g4 = (tj == 7) | (tj == 8) | (tj == 11) | (tj == 12)
    drop = (in_iso & jn_iso) | (in_g4 & jn_g4) | ((ti == 5) & (tj == 6)) | ((ti == 7) & (tj == 9)) | (ti == 15) | (tj == 15)
    keep = (tj < 20) & ~((ti == 0) & (tj == 19)) & ~drop
    return list(zip(ti[keep].tolist(), tj[keep].tolist()))


def block_labels(widths):
    tri6 = list(zip(*torch.triu_indices(6, 6, offset=1).tolist()))
    cross = lambda A, B: [pair(a, b) for a in A for b in B]
    per_channel = lambda chans, cells: [chans[c] for c in range(len(chans)) for _ in range(cells)]
    L = {}
    uniform = {"path": "G", "persistence": "G", "second_order_grid": "T", "second_order_stats": "T",
               "cov_within_scale": "T", "cov_quadrant": "T"}
    uniform.update({b: "C" for b in BLOCKS if b.startswith(("orient_field", "phase_"))})
    for b, f in uniform.items():
        L[b] = [f] * widths[b]
    # V1-like stage: 30 channels pooled on grids (channel-major), split into luminance (E) and color (K).
    for b, cells in (("s1_cort_mean_o0", 81), ("s1_std_peak_o0", 25), ("s1_cort_mean_o1", 36),
                     ("s1_std_peak_o1", 9), ("s1_cort_mean_o2", 16), ("s1_std_peak_o2", 4)):
        L[b] = per_channel(C30, cells)
    L["s1_global_stats"] = C30 + C30                       # mean 30, std 30
    L["color_grid"] = ["E"] * 36 + ["K"] * 72               # luminance, RG, BY on 6x6
    L["cov_joint"] = [pair(CSC24[i], CSC24[j]) for i, j in csc_pairs()] * 2
    L["octave_persist_diff"] = [single(OCT12[c % 12], "T") for c in range(24) for _ in range(4)] * 2
    L["octave_parent_child_cov"] = cross(OCT12, ENV8) * 2
    L["ecc_symmetry"] = [single(SYM9[k % 9], "G") for k in range(27)] * 2
    L["geom_o0_mean"], L["geom_o0_std"] = per_channel(GMAP, 16), per_channel(GMAP, 9)
    L["geom_o1_mean"], L["geom_o1_std"] = per_channel(GMAP, 16), per_channel(GMAP, 9)
    L["geom_o2_mean"] = per_channel(GMAP, 9)
    L["geom_o2_std"] = per_channel(GMAP, 4) + GMAP + GMAP
    L["geom_o3"] = per_channel(GMAP, 4) + GMAP + GMAP
    L["geom_o4"] = GMAP + GMAP
    L["geom_rel_fine"] = cross(G6, G6) * 2
    L["geom_comp_fine"] = cross(GC6, GC6) * 2
    L["geom_rel_coarse"] = ([pair(G6[i], G6[j]) for i, j in tri6] + [pair(GC6[i], GC6[j]) for i, j in tri6]
                            + cross(G6, G6) + cross(GC6, GC6) + [single(GMAP[c], "T") for c in range(12)])
    L["geom_symmetry"] = ([single(GIN12[k % 12], "G") for k in range(24)] * 2
                          + [single(GIN11[k % 11], "G") for k in range(22)])
    for b in BLOCKS:
        assert len(L[b]) == widths[b], (b, len(L[b]), widths[b])
    return np.array([x for b in BLOCKS for x in L[b]])


def load_champion():
    source = FROZEN.read_text()
    marker = "            out_chunks.append(torch.cat(feats, dim=1))"
    assert source.count(marker) == 1
    patched = source.replace(marker, "            self._widths = [f.shape[1] for f in feats]\n" + marker)
    module = types.ModuleType("champion")
    exec(compile(patched, str(FROZEN), "exec"), module.__dict__)
    return module.MODEL


def shapley(values, players):
    """Exact Shapley values. values: {frozenset: (n_voxels,) array}, with every subset of players present."""
    n = len(players)
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(others, k):
                S = frozenset(S)
                phi[p] = phi[p] + w * (values[S | {p}] - values[S])
    return phi


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(NEURO_DATASET="laion", NEURO_STAGE="search", NEURO_METRIC="encoding",
                      NEURO_REGION="visual", NEURO_IMSIZE="128")
    sys.path.insert(0, str(REPO / "src"))
    from neuro_eval.benchmarks import protocol_for
    from neuro_eval.datasets import get_dataset, with_subject, load_split, _load_native
    from neuro_eval._evaluation import selected_masks
    from neuro_eval.encoding import design, _fit_predict, pearson, ENCODING_FEATURES

    device = torch.device("cuda")
    model = load_champion()
    with torch.no_grad():
        model(torch.rand(2, 128 * 128 * 3, device=device))
    assert len(model._widths) == len(BLOCKS)
    lab = block_labels(dict(zip(BLOCKS, model._widths)))
    assert len(lab) == 9996
    counts = {k: int((lab == k).sum()) for k in "EKSCTGX"}
    print("columns per label:", counts, flush=True)
    (out / "column_labels.json").write_text(json.dumps({"labels": lab.tolist(), "counts": counts,
                                                       "families": FAMILIES}))

    subjects = list(protocol_for("laion", "visual", "encoding").search_subjects)
    base = get_dataset("laion", region="visual", metric="encoding")
    players = list(FAMILIES)
    subsets = [frozenset(c) for k in range(1, 7) for c in itertools.combinations(players, k)]
    key = lambda S: "".join(p for p in players if p in S)

    results = {}
    for s in subjects:
        d = with_subject(base, s)
        Xtr, Xte, Ytr, Yte, masks, _ = load_split(d)
        masks = {k: np.asarray(v, dtype=bool) for k, v in selected_masks(masks, d).items()}
        native, subj = _load_native(d)
        xyz = np.concatenate([np.asarray(native["units"][r][subj]) for r in native["regions"]])
        assert len(xyz) == Ytr.shape[1], (len(xyz), Ytr.shape[1])
        fpath = out / f"features_{s}.npy"
        if fpath.exists():
            A = torch.from_numpy(np.load(fpath))
        else:
            X = torch.cat([Xtr, Xte]).to(device)
            with torch.no_grad():
                A = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)]).cpu()
            np.save(fpath, A.numpy())
        Y = torch.cat([Ytr, Yte])
        ntr, nte = len(Xtr), len(Xte)
        tr, te = np.arange(ntr), np.arange(ntr, ntr + nte)
        rpath = out / f"voxel_r_{s}.npz"
        cache = dict(np.load(rpath)) if rpath.exists() else {}
        t0 = time.time()
        for S in subsets + ["all_columns"]:
            name = S if isinstance(S, str) else key(S)
            if name in cache:
                continue
            keep = np.ones(len(lab), bool) if S == "all_columns" else np.isin(lab, list(S))
            Z = design(A[:, torch.as_tensor(np.flatnonzero(keep))], tr, device, features=ENCODING_FEATURES)
            _, Yt, pred = _fit_predict(Z, Y, tr, te)
            cache[name] = pearson(Yt, pred).float().cpu().numpy()
            del Z, Yt, pred
            torch.cuda.empty_cache()
            np.savez(rpath, **cache)
            print(f"{s} {name:12s} pooled r={cache[name].mean():.4f} ({len(cache)}/{len(subsets) + 1}, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        np.savez(out / f"voxels_{s}.npz", xyz=xyz, **{f"mask_{k}": v for k, v in masks.items()})
        results[s] = dict(r=cache, masks=masks)

    # Exact Shapley per voxel, for r and signed explained variance, in both games.
    summary = {"subjects": subjects, "families": FAMILIES, "columns_per_label": counts, "rois": {}, "scores": {}}
    for s in subjects:
        r = results[s]["r"]
        arrays = {}
        for scale, f in (("r", lambda x: x), ("r2", lambda x: np.sign(x) * x ** 2)):
            v = {frozenset(): np.zeros_like(r["E"])}
            v.update({S: f(r[key(S)].astype(np.float64)) for S in subsets})
            phi_full = shapley(v, players)
            rest = [p for p in players if p != "E"]
            vb = {S: v[S | {"E"}] - v[frozenset({"E"})] for S in v if "E" not in S}
            phi_beyond = shapley(vb, rest)
            for p in players:
                arrays[f"{scale}_full_{p}"] = phi_full[p]
            for p in rest:
                arrays[f"{scale}_beyond_edges_{p}"] = phi_beyond[p]
            arrays[f"{scale}_six_families"] = v[frozenset(players)]
            arrays[f"{scale}_edges_only"] = v[frozenset({"E"})]
        arrays["r_all_columns"] = r["all_columns"]
        np.savez(out / f"shapley_{s}.npz", **arrays)
        summary["rois"][s] = {}
        for roi, m in results[s]["masks"].items():
            summary["rois"][s][roi] = {k: float(a[m].mean()) for k, a in arrays.items()}
        summary["scores"][s] = {name: float(r[name].mean()) for name in [key(S) for S in subsets] + ["all_columns"]}
    rois = list(summary["rois"][subjects[0]])
    summary["roi_mean"] = {roi: {k: float(np.mean([summary["rois"][s][roi][k] for s in subjects]))
                                 for k in summary["rois"][subjects[0]][roi]} for roi in rois}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary written", flush=True)


if __name__ == "__main__":
    main()
