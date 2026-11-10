"""Text-bottleneck results: original LEAN-v20 vs 5 rebuilds (from the 197-word description) vs 5 controls
(no description), each scored once on six held-out sessions (grid electrodes).

1. Reproduction: the original, scored through the same wrapper (features on CPU), against the stored LEAN-v20
   scores (features on GPU): runs-wave/newtest (patient1, all 493) and runs-wave/multisubj (others).
2. Performance: per-subject grid r; paired chunk-bootstrap 95% CIs (B = 2000) for rebuild mean - original,
   control mean - original and rebuild mean - control mean.
3. Closeness to the original (the paper's measures): correlation of each model's held-out grid predictions with
   the original's (patient1, all 493 chunks; Pearson r per chunk and electrode, averaged), and dimensionality ratio
   H / 3,852 (H = P / 4), geometric mean per arm.
Writes RESULTS.md, results.csv, fig_bottleneck.{png,pdf}.
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BN = os.path.dirname(os.path.abspath(__file__))
RW = '/path/to/ecog_data/circuit-building-autoresearch/runs-wave'
MS, NT = f'{RW}/multisubj', f'{RW}/newtest'
SUBJ = [s for s in ['patient1ALL', 'patient2', 'patient3', 'patient4', 'patient5', 'patient6'] if s not in os.environ.get('BN_SKIP', '').split(',')]
REB = [f'rebuild_{k}' for k in range(1, 6)]
CTL = [f'control_{k}' for k in range(1, 6)]
MODELS = ['original'] + REB + CTL
H0 = 3852
B = 2000


def boot(d, rng):
    n = len(d)
    bs = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rng = np.random.RandomState(0)
    prim = set(json.load(open(f'{NT}/test_primary.json')))
    v20 = next(t['id'] for t in json.load(open(f'{NT}/trials.json')) if t['name'] == 'LEAN-v18.q45-SLOWQUAD5SCALES')
    rec, chunks = {}, {}
    sets = []
    for S in SUBJ:
        cfg = json.load(open(f'{MS}/{S}/split.json'))
        ng, test = cfg['n_grid'], cfg['test']
        g = {}
        for m in MODELS:
            r = json.load(open(f'{BN}/scoring/results/{m}_{S}.json'))
            assert r['status'] == 'ok' and r['n_test'] == len(test), (m, S, r)
            rec[(m, S)] = r
            g[m] = np.load(f'{BN}/scoring/results/{m}_{S}_r.npy')[:, :ng].mean(1)
        ref = np.load(f'{NT}/results/{v20}_rnew.npy') if S == 'patient1ALL' else np.load(f'{MS}/results/{S}/{v20}_r.npy')
        g['stored v20'] = ref[:, :ng].mean(1)
        g['rebuild mean'] = np.mean([g[m] for m in REB], 0)
        g['control mean'] = np.mean([g[m] for m in CTL], 0)
        name = 'patient1 (all 493)' if S == 'patient1ALL' else S
        sets.append((name, g))
        if S == 'patient1ALL':
            mk = np.array([c in prim for c in test])
            sets.append(('patient1 (primary 317)', {k: v[mk] for k, v in g.items()}))
            chunks[S] = test

    L = ['# Text bottleneck for LEAN-v20 (paper Fig. 2g protocol)', '',
         'A fresh Opus 5.5 session described LEAN-v20 in 197 words (`frozen/DESCRIPTION.md`); 5 fresh sessions rebuilt '
         'it from that description and the task alone; 5 controls got the identical task with no description '
         '(PROTOCOL.md; sandbox and audits in AUDIT.md, INVARIANCE.md). Every model scored once with the LEAN harness '
         'on six held-out sessions. Grid electrodes; test r = mean over held-out chunks.', '',
         '## 1. Reproduction of the original', '',
         '| Subject | original (this wrapper, CPU features) | stored LEAN-v20 (GPU features) | difference |', '|---|---|---|---|']
    for name, g in sets:
        L.append(f'| {name} | {g["original"].mean():.5f} | {g["stored v20"].mean():.5f} | '
                 f'{g["original"].mean() - g["stored v20"].mean():+.5f} |')
    L += ['', '## 2. Held-out performance', '',
          '| Subject | original | rebuilds (5) | rebuild mean | controls (5) | control mean | rebuild − original | '
          'control − original | rebuild − control |', '|---|---|---|---|---|---|---|---|---|']
    rows = []
    for name, g in sets:
        rb = ' '.join(f'{g[m].mean():.4f}' for m in REB)
        ct = ' '.join(f'{g[m].mean():.4f}' for m in CTL)
        d1 = boot(g['rebuild mean'] - g['original'], rng)
        d2 = boot(g['control mean'] - g['original'], rng)
        d3 = boot(g['rebuild mean'] - g['control mean'], rng)
        f = lambda d: f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]'
        L.append(f'| {name} | {g["original"].mean():.4f} | {rb} | {g["rebuild mean"].mean():.4f} | {ct} | '
                 f'{g["control mean"].mean():.4f} | {f(d1)} | {f(d2)} | {f(d3)} |')
        row = dict(subject=name, original=g['original'].mean(), stored_v20=g['stored v20'].mean(),
                   rebuild_mean=g['rebuild mean'].mean(), control_mean=g['control mean'].mean(),
                   rebuild_minus_original=d1[0], rebuild_minus_original_lo=d1[1], rebuild_minus_original_hi=d1[2],
                   control_minus_original=d2[0], control_minus_original_lo=d2[1], control_minus_original_hi=d2[2],
                   rebuild_minus_control=d3[0], rebuild_minus_control_lo=d3[1], rebuild_minus_control_hi=d3[2])
        row.update({m: g[m].mean() for m in REB + CTL})
        rows.append(row)

    # 3. closeness: prediction correlation with the original (patient1, all 493) and dimensionality ratio
    P0 = np.load(f'{BN}/scoring/preds/original_patient1ALL.npz')
    pc = {}
    for m in REB + CTL:
        Pm = np.load(f'{BN}/scoring/preds/{m}_patient1ALL.npz')
        vals = []
        for c in chunks['patient1ALL']:
            a = P0[c].astype(np.float64); b = Pm[c].astype(np.float64)
            a -= a.mean(0); b -= b.mean(0)
            den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
            vals.append(np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1), 0.0))
        pc[m] = float(np.mean(vals))
    dims = {m: rec[(m, 'patient1ALL')]['P'] // 4 for m in MODELS}
    gm = lambda ms: float(np.exp(np.mean([np.log(dims[m] / H0) for m in ms])))
    L += ['', '## 3. Closeness to the original', '',
          'Prediction correlation = Pearson r between a model\'s held-out grid predictions and the original\'s, per '
          'chunk and electrode, averaged (patient1, 493 chunks). Dimensionality ratio = H / 3,852.', '',
          '| Model | prediction correlation with original | dims (H) | H / 3,852 |', '|---|---|---|---|']
    for m in REB + CTL:
        L.append(f'| {m} | {pc[m]:.3f} | {dims[m]} | {dims[m] / H0:.3f} |')
    L += ['', f'- Prediction correlation, mean: rebuilds {np.mean([pc[m] for m in REB]):.3f} '
              f'(range {min(pc[m] for m in REB):.3f}–{max(pc[m] for m in REB):.3f}) vs controls '
              f'{np.mean([pc[m] for m in CTL]):.3f} (range {min(pc[m] for m in CTL):.3f}–{max(pc[m] for m in CTL):.3f}).',
          f'- Dimensionality ratio, geometric mean: rebuilds {gm(REB):.3f}× vs controls {gm(CTL):.3f}× the original.']
    with open(f'{BN}/results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(pred_corr=pc, dims=dims), open(f'{BN}/closeness.json', 'w'), indent=1)
    open(f'{BN}/RESULTS.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))

    show = [s for s in sets if s[0] != 'patient1 (primary 317)']
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for i, (name, g) in enumerate(show):
        a.scatter(np.full(5, i - 0.15) + np.linspace(-0.05, 0.05, 5), [g[m].mean() for m in REB], s=18,
                  color='#0072B2', zorder=3, label='rebuilds (from description)' if i == 0 else None)
        a.scatter(np.full(5, i + 0.15) + np.linspace(-0.05, 0.05, 5), [g[m].mean() for m in CTL], s=18,
                  color='#9a9a9a', zorder=3, label='controls (no description)' if i == 0 else None)
        a.hlines(g['original'].mean(), i - 0.32, i + 0.32, color='#222222', lw=1.8, zorder=4,
                 label='original LEAN-v20' if i == 0 else None)
    a.set_xticks(range(len(show))); a.set_xticklabels([s[0] for s in show], fontsize=8.5)
    a.set_ylabel('held-out test r, grid electrodes'); a.grid(axis='y', color='#e7e6e2', lw=0.6); a.set_axisbelow(True)
    a.legend(loc='upper right', fontsize=7.5, frameon=False)
    a.set_title('a  Held-out performance', loc='left', fontsize=10, weight='bold')
    for xi, (ms, col, lab) in enumerate(((REB, '#0072B2', 'rebuilds'), (CTL, '#9a9a9a', 'controls'))):
        b.scatter(np.full(5, xi) + np.linspace(-0.08, 0.08, 5), [pc[m] for m in ms], s=26, color=col, zorder=3)
        b.hlines(np.mean([pc[m] for m in ms]), xi - 0.2, xi + 0.2, color=col, lw=2)
    b.set_xticks([0, 1]); b.set_xticklabels(['rebuilds', 'controls']); b.set_xlim(-0.5, 1.5)
    b.set_ylabel('correlation of predictions with the original\'s (patient1)')
    b.grid(axis='y', color='#e7e6e2', lw=0.6); b.set_axisbelow(True)
    b.set_title('b  Closeness to the original', loc='left', fontsize=10, weight='bold')
    for ax in (a, b):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{BN}/fig_bottleneck.{ext}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
