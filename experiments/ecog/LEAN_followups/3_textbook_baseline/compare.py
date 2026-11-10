"""Textbook baseline vs the search, per subject (grid electrodes, held-out sessions).

For every subject: per-chunk grid-mean r of the 5 textbook agents (scoring/results), the search's trial 52,
LEAN-v16, LEAN-v20 and the model the search kept (runs-wave/newtest for patient1, runs-wave/multisubj for the
others) and Whisper (both runs). Test r = mean over held-out chunks. Differences use a paired bootstrap over
chunks (B = 2000). patient1 is reported on all 493 never-used chunks and on the 317-chunk primary subset.
Writes RESULTS.md, results.csv and fig_textbook_vs_search.{png,pdf}.
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TB = os.path.dirname(os.path.abspath(__file__))
RW = '/path/to/ecog_data/circuit-building-autoresearch/runs-wave'
MS, NT = f'{RW}/multisubj', f'{RW}/newtest'
SUBJ = ['patient1ALL', 'patient2', 'patient3', 'patient4', 'patient5', 'patient6']
LEAN = {'trial 52': 'Gammatone-BEST+HL+modbank+STMF14', 'v16': 'LEAN-v16', 'v20': 'LEAN-v18.q45-SLOWQUAD5SCALES',
        'kept': 'LEAN-v35-AUTOLAGZ'}
B = 2000


def grid_chunks(r, ng):
    return r[:, :ng].mean(1)


def load(S):
    cfg = json.load(open(f'{MS}/{S}/split.json'))
    ng, test = cfg['n_grid'], cfg['test']
    out = {}
    for k in range(1, 6):
        rec = json.load(open(f'{TB}/scoring/results/agent{k}_{S}.json'))
        assert rec['status'] == 'ok' and rec['n_test'] == len(test), (k, S, rec)
        out[f'agent {k}'] = grid_chunks(np.load(f'{TB}/scoring/results/agent{k}_{S}_r.npy'), ng)
    ids = {t['name']: t['id'] for t in json.load(open(f'{NT}/trials.json'))}
    for lab, name in LEAN.items():
        if S == 'patient1ALL':
            r = np.load(f'{NT}/results/{ids[name]}_rnew.npy')
        else:
            r = np.load(f'{MS}/results/{S}/{ids[name]}_r.npy')
        assert r.shape[0] == len(test), (lab, S, r.shape)
        out[lab] = grid_chunks(r, ng)
    if S == 'patient1ALL':
        z = np.load(f'{RW}/lean-whisper-newtest/perchunk_r.npz')
        assert [str(c) for c in z['chunks_new']] == test
        out['Whisper'] = grid_chunks(z['r_new_whisper'], ng)
    else:
        z = np.load(f'{MS}/{S}/both/perchunk_r.npz')
        assert [str(c) for c in z['chunks']] == test
        out['Whisper'] = grid_chunks(z['r_whisper'], ng)
    out['agent mean'] = np.mean([out[f'agent {k}'] for k in range(1, 6)], 0)
    return out, test


def boot_diff(a, b, rng):
    n = len(a); d = a - b
    bs = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    prim = set(json.load(open(f'{NT}/test_primary.json')))
    sets = []
    for S in SUBJ:
        out, test = load(S)
        sets.append(('patient1 (all 493)' if S == 'patient1ALL' else S, out))
        if S == 'patient1ALL':
            m = np.array([c in prim for c in test])
            sets.append(('patient1 (primary 317)', {k: v[m] for k, v in out.items()}))
    cols = [f'agent {k}' for k in range(1, 6)] + ['agent mean', 'trial 52', 'v16', 'v20', 'kept', 'Whisper']
    rows, lines = [], []
    rng = np.random.RandomState(0)
    lines += ['# Textbook baseline vs the search (grid electrodes, held-out sessions)', '',
              'Five isolated agents (claude-opus-4-7, the search model) each built a one-shot "textbook" feature space '
              'from the literature, with no access to the search or its results and no performance feedback '
              '(PROTOCOL.md, AUDIT.md). Scored once each with the LEAN harness. Test r = mean over held-out chunks of '
              'the per-chunk Pearson r, averaged over grid electrodes. Differences: paired chunk bootstrap, 95% CI.', '',
              '| Subject | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    for name, out in sets:
        lines.append(f'| {name} | ' + ' | '.join(f'{out[c].mean():.4f}' for c in cols) + ' |')
    lines += ['', '## Search minus textbook (agent mean), with 95% CI', '',
              '| Subject | trial 52 − textbook | v16 − textbook | v20 − textbook | kept − textbook | Whisper − textbook | best agent | trial 52 − best agent |',
              '|---|---|---|---|---|---|---|---|']
    for name, out in sets:
        best = max(range(1, 6), key=lambda k: out[f'agent {k}'].mean())
        cells, row = [], dict(subject=name, n_chunks=len(out['v20']))
        for c in cols:
            row[c] = float(out[c].mean())
        for lab in ('trial 52', 'v16', 'v20', 'kept', 'Whisper'):
            d, lo, hi = boot_diff(out[lab], out['agent mean'], rng)
            row[f'{lab}_minus_textbook'] = d; row[f'{lab}_minus_textbook_lo'] = lo; row[f'{lab}_minus_textbook_hi'] = hi
            cells.append(f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        d, lo, hi = boot_diff(out['trial 52'], out[f'agent {best}'], rng)
        row.update(best_agent=best, trial52_minus_best=d, trial52_minus_best_lo=lo, trial52_minus_best_hi=hi)
        lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {best} ({out[f"agent {best}"].mean():.4f}) | '
                     f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}] |')
        rows.append(row)
    # trials the search needed before the model it kept (best so far by patient1 old-test score) beat the textbook mean
    ts = list(csv.DictReader(open(f'{MS}/summary/trial_scores_grid.csv')))
    old = np.array([float(r['patient1_old']) for r in ts])
    pick = np.array([int(np.argmax(old[:i + 1])) for i in range(len(old))])
    colmap = {'patient1 (all 493)': 'patient1_new_all493', 'patient1 (primary 317)': 'patient1_new_primary', 'patient2': 'patient2',
              'patient3': 'patient3', 'patient4': 'patient4', 'patient5': 'patient5', 'patient6': 'patient6'}
    lines += ['', '## Search trials needed to beat the textbook mean', '',
              'First trial (Figure 1 order, of ' + str(len(ts)) + ') at which the model the search was keeping '
              '(best so far on the patient1 old test) scores above the 5-agent textbook mean on that subject, and the trial '
              'from which it stays above.', '']
    for name, out in sets:
        y = np.array([float(r[colmap[name]]) for r in ts])[pick]
        hit = np.nonzero(y > out['agent mean'].mean())[0]
        below = np.nonzero(y <= out['agent mean'].mean())[0]
        first = int(hit[0]) + 1 if len(hit) else None
        stay = int(below[-1]) + 2 if len(below) else 1
        lines.append(f'- {name}: first above at trial {first} ({ts[pick[hit[0]]]["name"] if len(hit) else "never"}); '
                     f'stays above from trial {stay}')
        for row in rows:
            if row['subject'] == name:
                row['first_trial_beating_textbook'] = first
                row['stays_above_textbook_from_trial'] = stay

    with open(f'{TB}/results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    open(f'{TB}/RESULTS.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    show = [s for s in sets if s[0] != 'patient1 (primary 317)']
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(show))
    for i, (name, out) in enumerate(show):
        ag = [out[f'agent {k}'].mean() for k in range(1, 6)]
        ax.scatter(np.full(5, i - 0.18) + np.linspace(-0.06, 0.06, 5), ag, s=16, color='#9a9a9a', zorder=3,
                   label='textbook agents (5)' if i == 0 else None)
        ax.scatter(i - 0.18, np.mean(ag), s=60, marker='_', color='#555555', linewidths=2, zorder=4,
                   label='textbook mean' if i == 0 else None)
        for dx, lab, col, mk in ((0.0, 'trial 52', '#2b6cb0', 'o'), (0.12, 'v20', '#0072B2', 's'),
                                 (0.24, 'Whisper', '#E69F00', 'D')):
            ax.scatter(i + dx, out[lab].mean(), s=34, color=col, marker=mk, edgecolors='white', linewidths=0.8,
                       zorder=4, label={'trial 52': 'search, trial 52', 'v20': 'LEAN-v20', 'Whisper': 'Whisper'}[lab]
                       if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in show], fontsize=8.5)
    ax.set_ylabel('held-out test r, grid electrodes')
    ax.grid(axis='y', color='#e7e6e2', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper right', fontsize=7.5, frameon=False, ncol=2)
    ax.set_title('One-shot textbook feature spaces vs the search, per subject', loc='left', fontsize=10, weight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{TB}/fig_textbook_vs_search.{ext}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
