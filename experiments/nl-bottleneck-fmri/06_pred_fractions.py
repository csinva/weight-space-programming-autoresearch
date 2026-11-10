"""Step 6: prediction similarity between each rebuild and every original, over several voxel fractions.

04_evaluate.py stores, for each original, its predicted hidden-story responses on its 10% best-predicted hidden
voxels, and reports for each rebuild the per-voxel correlation between its predictions and its own original's,
averaged over those voxels (pred_similarity). This recomputes each rebuild's predictions and compares them with
every original i, on original i's best 1%, 2%, 5% and 10% of voxels. Row i, column j of the resulting matrix is
then how closely the rebuilds of model j predict like original i; the diagonal at 10% reproduces pred_similarity.
An exact permutation test over all pairings of originals with rebuilt models compares the diagonal mean with the
off-diagonal mean (diagonal_test in pred_matrix.json).
Those voxels all lie within the stored top 10%, so the originals are not refit.

  python 06_pred_fractions.py   -> WORK/recon_pred_fractions/<name>-r<k>.json, then pred_matrix.json (next to
                                   results.json): for each fraction, the matrix averaged over each model's rebuilds
"""
import itertools
import json
from importlib import import_module
from pathlib import Path

import numpy as np
from common import MODELS, WORK

FRACTIONS = (.01, .02, .05, .1)
evaluate = import_module('04_evaluate')


def similarity(p, q):
    pz = (p - p.mean(0)) / (p.std(0) + 1e-9)
    qz = (q - q.mean(0)) / (q.std(0) + 1e-9)
    return float(np.nanmean((pz * qz).mean(0)))


def permutation_test(M):
    """Exact test of whether the diagonal mean exceeds the off-diagonal mean: over every way of pairing the originals
    (rows) with the rebuilt models (columns), the fraction whose paired mean minus unpaired mean is at least the
    observed one (the observed pairing included)."""
    M = np.asarray(M)
    n, total, rows = len(M), M.sum(), np.arange(len(M))
    gap = lambda perm: M[rows, perm].mean() - (total - M[rows, perm].sum()) / (n * n - n)
    observed = gap(rows)
    null = np.array([gap(np.array(perm)) for perm in itertools.permutations(range(n))])
    return dict(diagonal_mean=float(M.diagonal().mean()), off_diagonal_mean=float((total - np.trace(M)) / (n * n - n)),
                p_value=float((null >= observed - 1e-12).mean()), n_permutations=len(null))


def main():
    out = WORK / 'recon_pred_fractions'
    out.mkdir(exist_ok=True)
    refs = {name: np.load(WORK / 'originals_eval' / f'{name}.npz') for _, name in MODELS}
    refs = {name: dict(top=r['top'], pred=r['pred_top'], n=len(r['corr_hidden'])) for name, r in refs.items()}
    for rd in sorted((WORK / 'recon').glob('*-r*')):
        meta = json.loads((rd / 'meta.json').read_text()) if (rd / 'meta.json').exists() else None
        if not meta or not meta.get('final') or (out / f'{rd.name}.json').exists():
            continue
        _, _, pred = evaluate.locked(evaluate.evaluate, rd / 'interpretable_transformers_lib' / f"{meta['final']}.py")
        vs = {}
        for name, ref in refs.items():  # ref['top'] is sorted by that original's hidden correlation, best last
            p, q = pred[:, ref['top']], ref['pred']
            vs[name] = {f'top{round(f * 100)}pct': similarity(p[:, -max(1, round(f * ref['n'])):], q[:, -max(1, round(f * ref['n'])):])
                        for f in FRACTIONS}
        (out / f'{rd.name}.json').write_text(json.dumps(dict(name=meta['name'], rep=meta['rep'], versus_original=vs), indent=1))
        print(rd.name, {k: round(v, 4) for k, v in vs[meta['name']].items()}, flush=True)
    names = [name for _, name in MODELS]
    runs = [json.loads(f.read_text()) for f in sorted(out.glob('*.json'))]
    matrix = {}
    for f in FRACTIONS:
        key = f'top{round(f * 100)}pct'
        # row: original i (its voxels and predictions); column: rebuilds of model j, averaged over their seeds
        matrix[key] = [[float(np.mean([r['versus_original'][orig][key] for r in runs if r['name'] == reb]))
                        for reb in names] for orig in names]
    (Path(__file__).resolve().parent / 'pred_matrix.json').write_text(json.dumps(dict(
        iterations=[it for it, _ in MODELS], names=names, rows='original', columns='rebuilt (mean over seeds)',
        n_rebuilds={reb: sum(r['name'] == reb for r in runs) for reb in names}, matrix=matrix,
        diagonal_test={key: permutation_test(M) for key, M in matrix.items()}), indent=1))


if __name__ == '__main__':
    main()
