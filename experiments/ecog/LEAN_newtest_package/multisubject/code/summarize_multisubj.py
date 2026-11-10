"""Cross-subject summary of the LEAN rerun: tables and three figures in multisubj/summary/.

Subjects: patient1 on its unfiltered held-out set (all 493 never-used chunks of session 450a36e4, from
runs-wave/newtest + varpart-v20-newtest r_all + lean-whisper-newtest) and patient2/743/749/798/836 on
their held-out sessions (this folder). Grid electrodes throughout (each subject's grid mask).

  ms_fig1_progression  every Figure 1 trial on each subject (dots, Figure 1 order); black step = the
                       model the search kept (best so far by patient1 old-test score) on that subject;
                       dashed = that subject's Whisper-only model (same ridge, 4 delays)
  ms_fig2_families     LEAN-v20 family unique variance (% of the full model, 95% CI) per subject
  ms_fig3_lean_whisper r of LEAN-v20 / Whisper / Both, and the LEAN-vs-Whisper partition per subject
"""
import csv
import glob
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MS = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(MS)
REPO = os.path.dirname(RW)
NT = os.path.join(RW, 'newtest')
OUT = os.path.join(MS, 'summary')
SUBJ = ['patient2', 'patient3', 'patient4', 'patient5', 'patient6']
ALL = ['patient1'] + SUBJ
INK, MUTED, GRID = '#222222', '#8a8984', '#e7e6e2'
PH = ['#2b6cb0', '#9c4221']
LEAN_C, WH_C, BOTH_C, SHARED_C = '#0072B2', '#E69F00', '#8E44AD', '#cfcdc8'
FAM = {'modulation': 'Modulation filters', 'interactions': 'Interactions (v3e)', 'scattering': 'Wavelet scattering',
       'contrast': 'Contrast', 'cochleagram_context': 'Cochleagram + context', 'pitch_voicing': 'Pitch and voicing',
       'shape_flux_formants': 'Shape, flux, formants'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#b9b8b3', 'axes.linewidth': 0.6,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#52514e',
                     'legend.frameon': False, 'savefig.dpi': 220, 'figure.facecolor': 'white'})


def prep_patient1_all():
    """patient1 family partition on all 493 held-out chunks, from the stored varpart-v20-newtest fits."""
    d = os.path.join(MS, 'patient1ALL')
    os.makedirs(os.path.join(d, 'varpart', 'fits'), exist_ok=True)
    json.dump(dict(subject='patient1ALL', held_out_session='450a36e4', n_electrodes=166, n_grid=125,
                   train=json.load(open(f'{NT}/train_frozen.json')), test=json.load(open(f'{NT}/test_candidates.json'))),
              open(os.path.join(d, 'split.json'), 'w'), indent=0)
    for f in glob.glob(os.path.join(RW, 'varpart-v20-newtest', 'fits', '*.npz')):
        z = np.load(f, allow_pickle=True)
        np.savez(os.path.join(d, 'varpart', 'fits', os.path.basename(f)), fit_id=z['fit_id'], raw_cols=z['raw_cols'],
                 r_chunk_elec=z['r_all'], chunks=z['chunks'])
    subprocess.run([sys.executable, os.path.join(MS, 'varpart_aggregate.py'), 'patient1ALL'], check=True,
                   stdout=subprocess.DEVNULL)


def load():
    trials = json.load(open(os.path.join(NT, 'trials.json')))
    seq = []
    for t in trials:
        p = os.path.join(NT, 'results', t['id'] + '.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if r['status'] != 'ok':
            continue
        row = dict(order=t['order'], id=t['id'], name=t['name'], phase=t['phase'], old=r['old_grid'],
                   patient1=r['newall_grid'], patient1_primary=r['new_grid'])
        for s in SUBJ:
            m = json.load(open(os.path.join(MS, 'results', s, t['id'] + '.json')))
            row[s] = m['grid'] if m['status'] == 'ok' else np.nan
            row[s + '_all'] = m.get('all_elec', np.nan)
        seq.append(row)
    both = {}
    for r in csv.DictReader(open(os.path.join(RW, 'lean-whisper-newtest', 'results.csv'))):
        if r['test'].startswith('new all') and r['electrodes'] == 'grid':
            both['patient1'] = r
    for s in SUBJ:
        both[s] = next(r for r in csv.DictReader(open(os.path.join(MS, s, 'both', 'results.csv'))) if r['electrodes'] == 'grid')
    fam = {}
    for s in ALL:
        d = 'patient1ALL' if s == 'patient1' else s
        fam[s] = {r['unit']: r for r in csv.DictReader(open(os.path.join(MS, d, 'varpart', 'results_family.csv')))}
    cfg = {s: json.load(open(os.path.join(MS, s, 'split.json'))) for s in SUBJ}
    cfg['patient1'] = dict(n_grid=125, n_electrodes=166, test=json.load(open(f'{NT}/test_candidates.json')),
                        held_out_session='450a36e4')
    return seq, both, fam, cfg


def fig1(seq, both, cfg):
    n = len(seq)
    x = np.arange(1, n + 1)
    old = np.array([r['old'] for r in seq])
    pick = np.array([int(np.argmax(old[:i + 1])) for i in range(n)])
    phases = []
    for i, r in enumerate(seq):
        if not phases or phases[-1][0] != r['phase']:
            phases.append([r['phase'], i, i + 1])
        phases[-1][2] = i + 1
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, ALL):
        y = np.array([r[s] for r in seq])
        for k, (_, a, b) in enumerate(phases):
            ax.axvspan(a + 0.5, b + 0.5, color=PH[k % 2], alpha=0.05, lw=0)
            ax.scatter(x[a:b], y[a:b], s=5, color=PH[k % 2], alpha=0.35, lw=0, zorder=2)
        ax.step(x, y[pick], where='post', color=INK, lw=1.6, zorder=4)
        wh = float(both[s]['r_whisper'])
        ax.axhline(wh, color='#555555', ls=(0, (5, 3)), lw=1.0, zorder=1)
        ax.text(n * 0.99, wh + 0.0012, f'Whisper {wh:.4f}', ha='right', va='bottom', fontsize=7.5, color='#555555')
        ax.text(n * 0.99, y[pick][-1] - 0.0016, f'kept model {y[pick][-1]:.4f}', ha='right', va='top', fontsize=7.5,
                color=INK, weight='bold')
        c = cfg[s]
        rr = np.corrcoef(old, y)[0, 1]
        ttl = f"{s}" + ('  (design subject)' if s == 'patient1' else '')
        ax.set_title(ttl, loc='left', fontsize=9.5, weight='bold', color=INK)
        ax.text(0.01, 0.97, f"{len(c['test'])} held-out chunks, {c['n_grid']} grid electrodes\n"
                            f"r(trial scores, patient1 old test) = {rr:.2f}", transform=ax.transAxes, va='top',
                fontsize=7.3, color='#52514e')
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel('encoder trial (Figure 1 order)')
    for ax in axes[:, 0]:
        ax.set_ylabel('held-out test r, grid electrodes')
    axes[0, 0].set_ylim(0.025, 0.09)
    fig.suptitle('LEAN search history rerun on six subjects, each scored on a held-out session (dots: trials; '
                 'black: model the search kept on patient1)', x=0.01, ha='left', fontsize=10.5, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'ms_fig1_progression.{ext}'), bbox_inches='tight')
    plt.close(fig)
    return pick


def fig2(fam):
    order = sorted(FAM, key=lambda u: -float(fam['patient1'][u]['grid_unique_share']))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True, sharey=True)
    y = np.arange(len(order))
    for ax, s in zip(axes.flat, ALL):
        f = fam[s]
        full = float(f[order[0]]['grid_alone_R2']) / float(f[order[0]]['grid_alone_share'])
        u = np.array([100 * float(f[k]['grid_unique_share']) for k in order])
        lo = np.array([100 * float(f[k]['grid_unique_lo']) / full for k in order])
        hi = np.array([100 * float(f[k]['grid_unique_hi']) / full for k in order])
        ax.axvline(0, color='#b9b8b3', lw=0.8)
        ax.hlines(y, lo, hi, color=LEAN_C, lw=1.2, zorder=3)
        ax.scatter(u, y, s=22, color=LEAN_C, edgecolors='white', linewidths=0.8, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([FAM[k] for k in order]); ax.set_ylim(len(y) - 0.5, -0.6)
        ax.tick_params(axis='y', length=0)
        tot = sum(float(f[k]['grid_unique_share']) for k in order)
        ax.set_title(f'{s}  (full R² {100 * full:.2f}%, shared {100 * (1 - tot):.0f}%)', loc='left', fontsize=9,
                     weight='bold', color=INK)
        ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel('unique variance, % of full LEAN-v20 model (95% CI)')
    fig.suptitle('LEAN-v20 feature families: unique variance on each subject\'s held-out session (grid electrodes)',
                 x=0.01, ha='left', fontsize=10.5, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'ms_fig2_families.{ext}'), bbox_inches='tight')
    plt.close(fig)
    return order


def fig3(both):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    y = np.arange(len(ALL))
    for m, col, lab in (('r_lean', LEAN_C, 'LEAN-v20'), ('r_whisper', WH_C, 'Whisper'), ('r_both', BOTH_C, 'Both')):
        v = [float(both[s][m]) for s in ALL]
        a.scatter(v, y, s=34, color=col, edgecolors='white', linewidths=0.8, zorder=3, label=lab)
    for i, s in enumerate(ALL):
        v = [float(both[s][k]) for k in ('r_lean', 'r_whisper', 'r_both')]
        a.hlines(i, min(v), max(v), color='#cfcdc8', lw=1.0, zorder=2)
    for m, lab in (('r_lean', 'LEAN'), ('r_whisper', 'Whisper'), ('r_both', 'Both')):
        a.annotate(lab, (float(both['patient1'][m]), 0), xytext=(0, 7), textcoords='offset points', ha='center',
                   fontsize=7, color='#52514e')
    a.set_yticks(y); a.set_yticklabels(ALL); a.set_ylim(len(y) - 0.5, -0.8); a.tick_params(axis='y', length=0)
    a.set_xlabel('held-out test r, grid electrodes'); a.grid(axis='x', color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(loc='lower right', fontsize=7.5, handletextpad=0.2)
    a.set_title('a  LEAN-v20, Whisper and both in one ridge', loc='left', fontsize=9.5, weight='bold')
    for i, s in enumerate(ALL):
        r = both[s]
        ul, uw, sh = (100 * float(r[k]) for k in ('unique_lean_share', 'unique_whisper_share', 'shared_share'))
        left = 0.0
        for v, col, tag in ((ul, LEAN_C, 'LEAN'), (sh, SHARED_C, 'shared'), (uw, WH_C, 'Whisper')):
            w = max(v, 0.0)
            b.barh(i, w, left=left, height=0.62, color=col, edgecolor='white', linewidth=1.0)
            if w >= 6:
                b.text(left + w / 2, i, f'{v:.0f}%', ha='center', va='center', fontsize=7.2,
                       color='white' if col != SHARED_C else INK)
            left += w + 0.0
        if ul < 6:
            b.text(-1.0, i, f'{ul:.1f}%', ha='right', va='center', fontsize=7.2, color=LEAN_C)
    b.set_yticks(y); b.set_yticklabels(ALL); b.set_ylim(len(y) - 0.5, -0.8); b.tick_params(axis='y', length=0)
    b.set_xlim(-9, 104); b.set_xlabel('% of R²(Both)')
    for x0, tag, col in ((2, 'unique LEAN', LEAN_C), (50, 'shared', '#52514e'), (98, 'unique Whisper', '#a06d00')):
        b.text(x0, -0.62, tag, ha='center' if tag == 'shared' else ('left' if x0 < 50 else 'right'), fontsize=7.5,
               color=col, weight='bold')
    b.set_title('b  Variance partition between LEAN-v20 and Whisper', loc='left', fontsize=9.5, weight='bold')
    b.spines['left'].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'ms_fig3_lean_whisper.{ext}'), bbox_inches='tight')
    plt.close(fig)


def tables(seq, both, fam, cfg, pick, order):
    old = np.array([r['old'] for r in seq])
    idx = {r['name']: i for i, r in enumerate(seq)}
    k16, k20 = idx['LEAN-v16'], idx['LEAN-v18.q45-SLOWQUAD5SCALES']
    kept = pick[-1]
    rows = []
    for s in ALL:
        y = np.array([r[s] for r in seq])
        b = both[s]; c = cfg[s]
        rows.append(dict(
            subject=s, held_out_session=c['held_out_session'], test_chunks=len(c['test']), grid_electrodes=c['n_grid'],
            v16=y[k16], v20=y[k20], kept=y[kept], gain_v16_v20=y[k20] - y[k16], best_trial=np.nanmax(y),
            corr_trials_vs_patient1_old=np.corrcoef(old, y)[0, 1],
            corr_trials_vs_patient1_new=np.corrcoef(np.array([r['patient1'] for r in seq]), y)[0, 1],
            whisper=float(b['r_whisper']), both=float(b['r_both']), lean_v20_in_both_run=float(b['r_lean']),
            both_minus_whisper=float(b['r_both']) - float(b['r_whisper']),
            unique_lean=100 * float(b['unique_lean_share']),
            unique_lean_ci=f"[{100 * float(b['unique_lean_lo']) / float(b['R2_both']):.1f}, {100 * float(b['unique_lean_hi']) / float(b['R2_both']):.1f}]",
            unique_whisper=100 * float(b['unique_whisper_share']), shared=100 * float(b['shared_share']),
            **{f'fam_unique_{u}': 100 * float(fam[s][u]['grid_unique_share']) for u in order}))
    with open(os.path.join(OUT, 'summary_table.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, 'trial_scores_grid.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['fig1_order', 'id', 'name', 'phase', 'patient1_old', 'patient1_new_primary', 'patient1_new_all493'] + SUBJ)
        for r in seq:
            w.writerow([r['order'], r['id'], r['name'], r['phase'], f"{r['old']:.5f}", f"{r['patient1_primary']:.5f}",
                        f"{r['patient1']:.5f}"] + [f'{r[s]:.5f}' for s in SUBJ])
    L = ['# Cross-subject summary (grid electrodes, held-out sessions)', '',
         'patient1 = all 493 never-used chunks of its held-out session (unfiltered, like the others). Trial scores: '
         f'{len(seq)} Figure 1 trials. "Kept" = the model the search kept on patient1 (LEAN-v35-AUTOLAGZ). '
         'Unique/shared as % of R²(Both).', '',
         '| Subject | Test chunks | v16 | v20 | Kept | v16→v20 | Whisper | Both | Both − Whisper | Unique LEAN [95% CI] | Unique Whisper | Shared | r(trials, patient1 old) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['subject']} | {r['test_chunks']} | {r['v16']:.4f} | {r['v20']:.4f} | {r['kept']:.4f} | "
                 f"{r['gain_v16_v20']:+.4f} | {r['whisper']:.4f} | {r['both']:.4f} | {r['both_minus_whisper']:+.4f} | "
                 f"{r['unique_lean']:.1f}% {r['unique_lean_ci']} | {r['unique_whisper']:.1f}% | {r['shared']:.1f}% | "
                 f"{r['corr_trials_vs_patient1_old']:.2f} |")
    L += ['', '## LEAN-v20 family unique variance (% of full model)', '',
          '| Subject | ' + ' | '.join(FAM[u] for u in order) + ' |', '|---|' + '---|' * len(order)]
    for r in rows:
        L.append(f"| {r['subject']} | " + ' | '.join(f"{r[f'fam_unique_{u}']:.1f}" for u in order) + ' |')
    open(os.path.join(OUT, 'SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    prep_patient1_all()
    seq, both, fam, cfg = load()
    pick = fig1(seq, both, cfg)
    order = fig2(fam)
    fig3(both)
    tables(seq, both, fam, cfg, pick, order)
