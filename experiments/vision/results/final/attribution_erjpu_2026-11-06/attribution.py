"""Computational-family importance for the search champion erjpu on LAION (search data only).

For each subset S of feature families, refit the exact search encoding readout
(neuro_eval.encoding.design 'center' + _fit_predict: per-voxel 5-fold-CV ridge)
on LAION train images and score Pearson r per voxel on the shared test images.
Region scores are mean voxel r over each mask (pooled early+ventral, streams, areas),
then averaged equally over the five subjects.

Analyses: full model, leave-one-family-out (unique), family-alone, base+family
(gain over Stage 1), and permutation-sampled Shapley values (v(empty)=0).
Uncertainty: paired image bootstrap over the 1,492 test images (same resamples for
every subset), plus subject spread. NSD is not used here.

Usage: python attribution.py OUTDIR [--perms N]
"""
import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path("<repo>")
RUN = REPO / "search_runs/laion__visual__encoding__2026-11-05"
FROZEN = REPO / "final_results/laion-visual-encoding-to-nsd-encoding/cef6828c767a85e0d94bb8a9ee4b01069e04b4a9aace7c1f391ae5ee0068d633/model.py"
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# Order of feats.append(...) in the champion (36 blocks); verified against recorded widths.
BLOCKS = ["path", "persistence",
          "s1_cort_mean_o0", "s1_std_peak_o0", "s1_cort_mean_o1", "s1_std_peak_o1",
          "s1_cort_mean_o2", "s1_std_peak_o2", "s1_global_stats",
          "color_grid", "second_order_grid", "second_order_stats",
          "cov_within_scale", "cov_quadrant",
          "orient_field_o0", "orient_field_o1", "orient_field_o2",
          "phase_o0", "phase_o1", "phase_o2",
          "cov_joint_geom_curv_phase_color",
          "octave_persist_diff", "octave_parent_child_cov",
          "ecc_symmetry",
          "geom_o0_mean", "geom_o0_std", "geom_o1_mean", "geom_o1_std",
          "geom_o2_mean", "geom_o2_std", "geom_o3", "geom_o4",
          "geom_rel_fine", "geom_comp_fine", "geom_rel_coarse", "geom_symmetry"]
FAMILIES = {
    "stage1_V1_energy": ["s1_cort_mean_o0", "s1_std_peak_o0", "s1_cort_mean_o1", "s1_std_peak_o1",
                         "s1_cort_mean_o2", "s1_std_peak_o2", "s1_global_stats"],
    "color_layout": ["color_grid"],
    "second_order": ["second_order_grid", "second_order_stats"],
    "texture_covariance": ["cov_within_scale", "cov_quadrant"],
    "orientation_field": ["orient_field_o0", "orient_field_o1", "orient_field_o2"],
    "phase_congruency": ["phase_o0", "phase_o1", "phase_o2"],
    "joint_covariance": ["cov_joint_geom_curv_phase_color"],
    "cross_octave": ["octave_persist_diff", "octave_parent_child_cov"],
    "surface_geometry": ["geom_o0_mean", "geom_o0_std", "geom_o1_mean", "geom_o1_std",
                         "geom_o2_mean", "geom_o2_std", "geom_o3", "geom_o4"],
    "geometry_relations": ["geom_rel_fine", "geom_comp_fine", "geom_rel_coarse"],
    "global_layout_symmetry": ["ecc_symmetry", "geom_symmetry", "path", "persistence"],
}
BASE = "stage1_V1_energy"
N_BOOT = 200


def load_champion():
    source = FROZEN.read_text()
    marker = "            out_chunks.append(torch.cat(feats, dim=1))"
    assert source.count(marker) == 1
    patched = source.replace(marker, "            self._widths = [f.shape[1] for f in feats]\n" + marker)
    module = types.ModuleType("champion")
    exec(compile(patched, str(FROZEN), "exec"), module.__dict__)
    original = types.ModuleType("champion_original")
    exec(compile(source, str(FROZEN), "exec"), original.__dict__)
    return module.MODEL, original.MODEL


def block_columns(widths):
    assert len(widths) == len(BLOCKS), (len(widths), len(BLOCKS))
    cols, start = {}, 0
    for name, w in zip(BLOCKS, widths):
        cols[name] = np.arange(start, start + w)
        start += w
    assert start == 9996
    fam = {f: np.concatenate([cols[b] for b in bs]) for f, bs in FAMILIES.items()}
    assert sorted(np.concatenate(list(fam.values())).tolist()) == list(range(9996))
    return cols, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--conditional", action="store_true",
                    help="Shapley over the gain above Stage 1: Stage 1 always present, v(S)=score(S+stage1)-score(stage1)")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    env = json.loads((RUN / "env.json").read_text())
    from neuro_eval.model import EVAL_ENV_KEYS
    os.environ.update({k: str(env[k]) for k in EVAL_ENV_KEYS if k in env})
    os.environ.update(NEURO_DATASET="laion", NEURO_STAGE="search", NEURO_METRIC="encoding", NEURO_REGION="visual")
    from neuro_eval.datasets import get_dataset, with_subject, load_split
    from neuro_eval._evaluation import selected_masks
    from neuro_eval.encoding import design, _fit_predict, pearson, ENCODING_FEATURES

    device = torch.device("cuda")
    model, original = load_champion()
    base = get_dataset("laion", region="visual", metric="encoding")
    subjects = env["NEURO_SUBJECT"].split(",")

    # 1. Features once per subject (cached), with a bitwise check against the unmodified model.
    data = {}
    for s in subjects:
        d = with_subject(base, s)
        Xtr, Xte, Ytr, Yte, masks, _ = load_split(d)
        masks = selected_masks(masks, d)
        path = out / f"features_{s}.npy"
        if path.exists():
            A = torch.from_numpy(np.load(path))
        else:
            X = torch.cat([Xtr, Xte]).to(device)
            with torch.no_grad():
                A = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)]).cpu()
                check = original(X[:512]).cpu()  # same batch as the first chunk: must be bitwise equal
            assert torch.equal(check, A[:512]), "instrumented model differs from frozen champion"
            np.save(path, A.numpy())
        data[s] = dict(A=A, Y=torch.cat([Ytr, Yte]), ntr=len(Xtr), nte=len(Xte), masks=masks)
        print(f"features {s}: {tuple(A.shape)}", flush=True)
    _, fam_cols = block_columns(model._widths) if hasattr(model, "_widths") else (None, None)
    if fam_cols is None:  # features came from cache; recover widths from one forward
        with torch.no_grad():
            model(torch.rand(1, 128 * 128 * 3, device=device))
        _, fam_cols = block_columns(model._widths)
    rois = list(data[subjects[0]]["masks"])
    nte = data[subjects[0]]["nte"]
    rng = np.random.default_rng(0)
    boot = torch.as_tensor(rng.integers(0, nte, size=(N_BOOT, nte)), device=device)

    cache_path = out / "subset_scores.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def key(subset):
        return "+".join(sorted(subset))

    def score(subset):
        """Subject-mean ROI scores and their paired bootstrap distributions for a family subset."""
        k = key(subset)
        if k in cache:
            return cache[k]
        if not subset:
            cache[k] = {"roi": {r: 0.0 for r in rois}, "boot": {r: [0.0] * N_BOOT for r in rois},
                        "subjects": {s: {r: 0.0 for r in rois} for s in subjects}}
            return cache[k]
        cols = torch.as_tensor(np.sort(np.concatenate([fam_cols[f] for f in subset])))
        per_subject, boots = {}, {r: np.zeros(N_BOOT) for r in rois}
        for s in subjects:
            dd = data[s]
            tr, te = np.arange(dd["ntr"]), np.arange(dd["ntr"], dd["ntr"] + dd["nte"])
            Z = design(dd["A"][:, cols], tr, device, features=ENCODING_FEATURES)
            _, Yte, pred = _fit_predict(Z, dd["Y"], tr, te)
            r = pearson(Yte, pred).double().cpu().numpy()
            per_subject[s] = {}
            for name, mask in dd["masks"].items():
                m = torch.as_tensor(np.asarray(mask), device=device)
                per_subject[s][name] = float(r[np.asarray(mask)].mean())
                ym, pm = Yte[:, m], pred[:, m]
                for b0 in range(0, N_BOOT, 20):  # bounded memory: 20 resamples at a time
                    idx = boot[b0:b0 + 20]
                    yb, pb = ym[idx], pm[idx]  # (20, n, v)
                    yb, pb = yb - yb.mean(1, keepdim=True), pb - pb.mean(1, keepdim=True)
                    rb = (yb * pb).sum(1) / (yb.norm(dim=1) * pb.norm(dim=1)).clamp_min(1e-12)
                    boots[name][b0:b0 + 20] += rb.mean(1).double().cpu().numpy() / len(subjects)
                    del yb, pb, rb
            del Z, Yte, pred
        cache[k] = {"roi": {r: float(np.mean([per_subject[s][r] for s in subjects])) for r in rois},
                    "boot": {r: boots[r].round(6).tolist() for r in rois}, "subjects": per_subject}
        cache_path.write_text(json.dumps(cache))
        return cache[k]

    fams = list(FAMILIES)
    t0 = time.time()
    full = score(fams)
    print(f"full pooled {full['roi'][rois[0]]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for f in fams:
        score([g for g in fams if g != f]); score([f])
        if f != BASE:
            score([BASE, f])
        print(f"single analyses {f} done ({time.time() - t0:.0f}s)", flush=True)

    # Shapley by permutation sampling; marginal contributions reuse cached prefixes.
    players = [f for f in fams if f != BASE] if args.conditional else fams
    perms_path = out / ("shapley_conditional_permutations.json" if args.conditional else "shapley_permutations.json")
    perms = json.loads(perms_path.read_text()) if perms_path.exists() else []
    prng = random.Random(2 if args.conditional else 1)
    orders = []
    for _ in range(args.perms):  # fixed seeded orders; resume skips completed ones
        order = players[:]
        prng.shuffle(order)
        orders.append(order)
    while len(perms) < args.perms:
        order = orders[len(perms)]
        start = [BASE] if args.conditional else []
        marg, prefix, prev = {}, start, score(start)
        for f in order:
            prefix = prefix + [f]
            cur = score(prefix)
            marg[f] = {r: cur["roi"][r] - prev["roi"][r] for r in rois}
            prev = cur
        perms.append({"order": order, "marginal": marg})
        perms_path.write_text(json.dumps(perms))
        print(f"shapley permutation {len(perms)}/{args.perms} ({time.time() - t0:.0f}s, {len(cache)} subsets)", flush=True)

    if args.conditional:
        summarize_conditional(out, cache, perms, players, rois, key)
    else:
        summarize(out, cache, perms, fams, rois, key, subjects)


def summarize_conditional(out, cache, perms, players, rois, key):
    base, full = cache[key([BASE])], cache[key([BASE] + players)]
    table = {f: {r: {"shapley_gain": float(np.mean([p["marginal"][f][r] for p in perms])),
                     "se": float(np.std([p["marginal"][f][r] for p in perms], ddof=1) / np.sqrt(len(perms)))}
                 for r in rois} for f in players}
    gain = {r: full["roi"][r] - base["roi"][r] for r in rois}
    (out / "attribution_conditional_summary.json").write_text(json.dumps(
        {"stage1_score": base["roi"], "full_score": full["roi"], "gain_over_stage1": gain,
         "n_permutations": len(perms), "table": table}, indent=2))
    print("conditional summary written", flush=True)


def summarize(out, cache, perms, fams, rois, key, subjects):
    def ci(x):
        x = np.asarray(x)
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    full = cache[key(fams)]
    table = {}
    for f in fams:
        loo = cache[key([g for g in fams if g != f])]
        alone = cache[key([f])]
        row = {}
        for r in rois:
            fb, lb = np.asarray(full["boot"][r]), np.asarray(loo["boot"][r])
            drop = full["roi"][r] - loo["roi"][r]
            subj = [full["subjects"][s][r] - loo["subjects"][s][r] for s in subjects]
            entry = {"unique_drop": drop, "unique_drop_ci95": ci(fb - lb),
                     "unique_drop_subjects_positive": int(sum(x > 0 for x in subj)),
                     "alone": alone["roi"][r], "alone_ci95": ci(alone["boot"][r])}
            if f != BASE:
                bf = cache[key([BASE, f])]
                bb = cache[key([BASE])]
                entry["gain_over_stage1"] = bf["roi"][r] - bb["roi"][r]
                entry["gain_over_stage1_ci95"] = ci(np.asarray(bf["boot"][r]) - np.asarray(bb["boot"][r]))
            vals = [p["marginal"][f][r] for p in perms]
            entry["shapley"] = float(np.mean(vals)) if vals else None
            entry["shapley_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
            row[r] = entry
        table[f] = row
    result = {"full": {r: full["roi"][r] for r in rois}, "full_ci95": {r: ci(full["boot"][r]) for r in rois},
              "families": FAMILIES, "n_shapley_permutations": len(perms), "n_bootstrap": N_BOOT,
              "subjects": subjects, "table": table}
    (out / "attribution_summary.json").write_text(json.dumps(result, indent=2))
    print("summary written", flush=True)


if __name__ == "__main__":
    main()
