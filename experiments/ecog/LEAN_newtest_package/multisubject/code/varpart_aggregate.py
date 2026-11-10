"""Multi-subject LEAN-v20 variance partitioning, phase 3 (tables), from varpart-v20-newtest/phase3_aggregate.py.
usage: varpart_aggregate.py <SUBJ>   (reads <SUBJ>/varpart/fits, writes RESULTS.md, results_*.csv, perelec_R2.npz)
Electrode sets: grid = the subject's first n_grid electrodes (grid_masks), all = every electrode."""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'varpart-v20'))
import groups as G  # noqa: E402

MS = os.path.dirname(os.path.abspath(__file__))
SUBJ = sys.argv[1]
CFG = json.load(open(os.path.join(MS, SUBJ, 'split.json')))
HERE = os.path.join(MS, SUBJ, 'varpart')
B = 2000
ESETS = {'grid': slice(0, CFG['n_grid']), 'all': slice(0, CFG['n_electrodes'])}


def r2(r_e):
    return r_e * np.abs(r_e)


def main():
    fits = {}
    for f in glob.glob(os.path.join(HERE, 'fits', '*.npz')):
        if f.endswith('.tmp.npz'):
            continue
        z = np.load(f, allow_pickle=True)
        fits[str(z['fit_id'])] = z['r_chunk_elec']
    expected = [fid for fid, _ in G.fit_list()]
    missing = [fid for fid in expected if fid not in fits]
    full = fits.get('full')
    if full is None:
        print('full model fit missing; nothing to aggregate'); return
    nch = full.shape[0]
    rng = np.random.RandomState(0)
    boots = [rng.randint(0, nch, nch) for _ in range(B)]

    def summary(rr, es):
        return float(r2(rr.mean(0))[es].mean()), float(rr.mean(0)[es].mean())

    def boot(rr, es):
        return np.array([r2(rr[b].mean(0))[es].mean() for b in boots])

    lines = [f'# LEAN-v20 variance partitioning — {SUBJ}, held-out session {CFG["held_out_session"]}', '',
             f'{nch} test chunks (the whole held-out session, never used in training); 500 training chunks from '
             f'the other sessions; {CFG["n_electrodes"]} electrodes, grid = first {CFG["n_grid"]}. Harness ridge '
             '(per-electrode alpha by bootstrapped CV). Variance explained per electrode = r·|r| with r the '
             'per-chunk-averaged Pearson r; averaged over electrodes. 95% CIs from a paired bootstrap over test '
             'chunks.', '']
    if missing:
        lines += [f'**{len(missing)} of {len(expected)} fits missing** — tables below are partial.', '']

    val = []
    for k, es in ESETS.items():
        R2, r = summary(full, es)
        val.append(f'{k}: mean r = {r:.4f}, R² = {100 * R2:.3f}% of variance')
    lines += ['## Full model', ''] + [f'- {v}' for v in val] + ['']

    perelec = {}
    for level, us in G.units().items():
        rows = []
        for key, label, gs in us:
            k = G.fit_key(gs)
            a, d = fits.get(f'alone:{k}'), fits.get(f'drop:{k}')
            ndim = len(G.raw_cols(gs))
            row = dict(level=level, unit=key, label=label, raw_dims=ndim)
            for en, es in ESETS.items():
                R2f, rf = summary(full, es)
                bf = boot(full, es)
                if a is not None:
                    R2a, ra = summary(a, es); ba = boot(a, es)
                    row.update({f'{en}_alone_R2': R2a, f'{en}_alone_share': R2a / R2f,
                                f'{en}_alone_r': ra,
                                f'{en}_alone_lo': np.percentile(ba, 2.5),
                                f'{en}_alone_hi': np.percentile(ba, 97.5)})
                if d is not None:
                    R2d, rd = summary(d, es); bd = boot(d, es)
                    u = bf - bd
                    row.update({f'{en}_unique_R2': R2f - R2d, f'{en}_unique_share': (R2f - R2d) / R2f,
                                f'{en}_drop_dr': rf - rd,
                                f'{en}_unique_lo': np.percentile(u, 2.5),
                                f'{en}_unique_hi': np.percentile(u, 97.5)})
                    if en == 'all':
                        perelec[f'{level}:{key}:unique'] = r2(full.mean(0)) - r2(d.mean(0))
                if a is not None and en == 'all':
                    perelec[f'{level}:{key}:alone'] = r2(a.mean(0))
            rows.append(row)
        with open(os.path.join(HERE, f'results_{level}.csv'), 'w', newline='') as fh:
            keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('level', 'unit', 'label', 'raw_dims'), k))
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

        for en in ('grid', 'all'):
            R2f, _ = summary(full, ESETS[en])
            lines += [f'## {level.capitalize()} level — {en} (full R² = {100 * R2f:.3f}%)', '',
                      '| Unit | Dims | Alone R² (%) | Alone share | Unique R² (%) [95% CI] | Unique share |',
                      '|---|---|---|---|---|---|']
            srt = sorted(rows, key=lambda r: -r.get(f'{en}_unique_R2', -9))
            tot_u = 0.0
            for r in srt:
                al = (f"{100 * r[f'{en}_alone_R2']:.3f}" if f'{en}_alone_R2' in r else '—')
                als = (f"{100 * r[f'{en}_alone_share']:.0f}%" if f'{en}_alone_share' in r else '—')
                if f'{en}_unique_R2' in r:
                    un = (f"{100 * r[f'{en}_unique_R2']:+.4f} [{100 * r[f'{en}_unique_lo']:+.4f}, "
                          f"{100 * r[f'{en}_unique_hi']:+.4f}]")
                    us_ = f"{100 * r[f'{en}_unique_share']:+.1f}%"
                    tot_u += r[f'{en}_unique_share']
                else:
                    un, us_ = '—', '—'
                lines.append(f"| {r['label']} | {r['raw_dims']} | {al} | {als} | {un} | {us_} |")
            lines += ['', f'Sum of unique shares: {100 * tot_u:.1f}% → shared (not attributable to any one unit): '
                          f'{100 * (1 - tot_u):.1f}%', '']

    np.savez(os.path.join(HERE, 'perelec_R2.npz'), **perelec,
             full=r2(full.mean(0)))
    with open(os.path.join(HERE, 'RESULTS.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({'fits_found': len(fits), 'expected': len(expected), 'missing': missing[:10]}))


if __name__ == '__main__':
    main()
