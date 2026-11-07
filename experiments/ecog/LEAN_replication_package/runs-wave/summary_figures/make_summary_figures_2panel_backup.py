"""Summary figures for the waveform -> ECoG hand-built encoder (LEAN) work.

fig1_progression: every scored autoresearch attempt by branch (all-166 phase and grid
phase), the promoted LEAN versions against Whisper, and what the v21 bug fixes removed.
fig2_breakdown: LEAN-v21 variance partitioning (grid electrodes) at family, section and
group level.

All scores: fixed 23-chunk speech-rich test set, 500 training chunks, ridge on 4 FIR
delays, canonical per-chunk per-electrode Pearson r (mean over electrodes).
"""
from __future__ import annotations

import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RW = os.path.dirname(HERE)

# Okabe-Ito audio blue as the single accent (this is audio-only work); light blue for the
# second metric; ink and grays for context and text.
BLUE, LBLUE = '#0072B2', '#56B4E9'
INK, INK2, MUTED, GRID, CTX = '#1F2A44', '#52514e', '#8a8984', '#e7e6e2', '#b9b8b3'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8.5,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.labelsize': 7.5,
    'axes.edgecolor': CTX, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 7, 'savefig.dpi': 300,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def hgrid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- data
EXCLUDE = re.compile(r'refer|MEL-CTRL|baseline', re.I)

BRANCHES_166 = [  # (label, runs) in chronological order
    ('Hand-built LEAN, v0 → v16', ['wave-slurm-23628821', 'wave-trim', 'wave-lean']),
    ('A  Auditory periphery', ['reset-A']),
    ('B  Spectro-temporal modulation', ['reset-B']),
    ('C  Phonological features', ['reset-C']),
    ('D  Wavelet scattering', ['reset-D']),
    ('E  Divisive normalization', ['reset-E']),
    ('F  Predictive coding', ['reset-F']),
    ('G  Texture statistics', ['reset-G']),
    ('H  Fixed random reservoir', ['reset-H']),
    ('I  Phase / fine structure', ['reset-I']),
    ('Graft best blocks onto v16', ['graft-lean']),
    ('Linguistic (transcript) features', ['lean-interp-ling']),
    ('Interpretable interactions → v17', ['lean-interp']),
    ('Dense soft maps', ['exceed-dense']),
]
BRANCHES_GRID = [
    ('Front-end features (contrast)', ['exceed-frontend']),
    ('Wavelet scattering', ['exceed-scatter']),
    ('Combine → v18', ['exceed-combine']),
    ('Quad banks + gates → v19, v20', ['exceed-v18']),
    ('Long context (reservoir)', ['exceed-v20-context']),
    ('Cross-band coupling, PAC', ['exceed-v20-deep']),
    ('Speaker normalization', ['exceed-v20-speaker']),
]


def load_trials():
    seen, out = set(), {}
    for label, runs in BRANCHES_166 + BRANCHES_GRID:
        vals = []
        for run in runs:
            rows = list(csv.reader(open(os.path.join(RW, run, 'results', 'overall_results.csv'))))
            if rows and rows[0][0] == 'subject':
                rows = rows[1:]
            for r in rows:
                if len(r) < 7:
                    continue
                key = (r[6], r[1])
                if key in seen or EXCLUDE.search(r[6]):
                    continue
                seen.add(key)
                vals.append(float(r[1]))
        out[label] = np.array(vals)
    return out


def full_r(path):
    rr = np.load(glob.glob(path)[0], allow_pickle=True)['r_chunk_elec'].mean(0)
    return float(rr.mean()), float(rr[:125].mean())


# Promoted versions (all-166, grid). Sources: v16 graft-lean GRAFT-v0-baseline; v17
# lean-v17-clean + exceed-scatter LEAN-v17-baseline; v18 combine-all166eval; v19
# v19-all166eval + exceed-v18 q29; v20/v21 varpart full fits (v20 equals the canonical
# diag_v20 run exactly); Whisper and LEAN-v20+Whisper from lean-whisper/diag_*.out.
v20_all, v20_grid = full_r(os.path.join(RW, 'varpart-v20', 'fits', 'full_*.npz'))
v21_all, v21_grid = full_r(os.path.join(RW, 'varpart-v21', 'fits', 'full_*.npz'))
VERSIONS = [('v16', 0.0760, None), ('v17', 0.0778, 0.08665), ('v18', 0.0800, 0.0891),
            ('v19', 0.0805, 0.09033), ('v20', v20_all, v20_grid), ('v21', v21_all, v21_grid)]
WHISPER = {'all': 0.0817, 'grid': 0.0933}
COMBINED = {'all': 0.0920, 'grid': 0.1038}  # LEAN-v20 + Whisper features, one ridge

# Columns removed by the v21 bug fixes (3,852 -> 3,392), by cause.
REMOVED = [  # (short label, full description, columns)
    ('±0.5 cyc/ch copies', '±0.5 cyc/ch Gabor copies (identical)', 294),
    ('12 Hz = 8 Hz parts', '12 Hz Gabor cos/sin equal to 8 Hz', 60),
    ('Scattering < 4 Hz', 'Scattering paths below 4 Hz (dead)', 54),
    ('Slow-bank 2 Hz copy', 'Slow-bank 2 Hz copy of the quadrature bank', 36),
    ('Flatness gates', 'Flatness gates (never open)', 12),
    ('Short-window octave', 'Short-window contrast octave (always 0)', 4),
]


FAM_SHORT = {'modulation': 'Modulation filters', 'cochleagram_context': 'Cochleagram + context',
             'interactions': 'Interactions (v3e)', 'scattering': 'Wavelet scattering',
             'contrast': 'Contrast', 'pitch_voicing': 'Pitch and voicing',
             'shape_flux_formants': 'Shape, flux, formants'}
SEC_SHORT = {'A': 'Cochleagram core', 'B': 'Pitch', 'C': 'Spectral shape', 'D': 'Spectral flux',
             'E': '1-D modulation', 'F': 'STMF', 'G': 'Formants', 'H': 'Slow context',
             'I': 'Pitch/voicing dynamics', 'J': 'Interactions (v3e)', 'K': 'Voice quality',
             'L': 'Contrast', 'M': 'Quadrature bank', 'N': 'Voicing gates',
             'O': 'Slow quad. bank', 'P': 'Wavelet scattering'}


def load_varpart(v, level):
    return list(csv.DictReader(open(os.path.join(RW, f'varpart-{v}', f'results_{level}.csv'))))


# ----------------------------------------------------------------------------- fig 1
def strip(ax, trials, branches, refs, xlim, title):
    labels = [b for b, _ in branches]
    rng = np.random.RandomState(0)
    for i, lab in enumerate(labels):
        v = trials[lab]
        if len(v) == 0:
            continue
        y = i + rng.uniform(-0.22, 0.22, len(v))
        ax.scatter(v, y, s=7, color=CTX, alpha=0.75, linewidths=0, zorder=2)
        ax.scatter([v.max()], [i], s=26, color=BLUE, edgecolors='white', linewidths=1.0, zorder=4)
        ax.text(xlim[1], i, f'  {v.max():.4f}', va='center', ha='left', fontsize=6.5, color=INK2,
                clip_on=False)
    span = xlim[1] - xlim[0]
    for x, txt, side, row in refs:  # side: label left/right of its line; row 0 = top
        ax.axvline(x, color=INK, linewidth=0.8, zorder=3)
        sgn = -1 if side == 'left' else 1
        ax.text(x + sgn * 0.012 * span, (-1.3, -0.7)[row], txt, ha='right' if sgn < 0 else 'left',
                va='center', fontsize=6.5, color=INK, zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f'{lab}  ({len(trials[lab])})' for lab in labels])
    ax.set_ylim(len(labels) - 0.4, -1.7)
    ax.set_xlim(*xlim)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    hgrid(ax, 'x')
    ax.set_title(title, pad=5)


def fig1(trials):
    fig = plt.figure(figsize=(7.4, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[len(BRANCHES_166) + 1.5,
                          len(BRANCHES_GRID) + 2.5], hspace=0.36, wspace=0.62,
                          left=0.265, right=0.985, top=0.925, bottom=0.125)
    a = fig.add_subplot(gs[0, 0])
    strip(a, trials, BRANCHES_166,
          [(0.0760, 'Start: LEAN-v16 0.0760', 'left', 0), (0.0778, 'End: LEAN-v17 0.0778', 'right', 1),
           (WHISPER['all'], 'Whisper 0.0817', 'right', 0)],
          (0.040, 0.085), 'a  Every scored attempt, all 166 electrodes')
    a.set_xlabel('Test r (mean over 166 electrodes)')
    b = fig.add_subplot(gs[1, 0])
    strip(b, trials, BRANCHES_GRID,
          [(0.08665, 'Start: LEAN-v17 0.0867', 'left', 0), (WHISPER['grid'], 'Whisper 0.0933', 'right', 0)],
          (0.069, 0.095), 'b  Every scored attempt, 125 grid electrodes')
    b.set_xlabel('Test r (mean over 125 grid electrodes)')

    c = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(VERSIONS))
    for key, col, idx in (('grid', BLUE, 2), ('all', LBLUE, 1)):
        y = np.array([v[idx] if v[idx] is not None else np.nan for v in VERSIONS])
        c.plot(xs, y, color=col, linewidth=2, solid_capstyle='round', zorder=3)
        c.scatter(xs, y, s=22, color=col, edgecolors='white', linewidths=1.0, zorder=4)
        c.axhline(WHISPER[key], color=col, linewidth=0.8, alpha=0.9, zorder=2)
        c.scatter([len(VERSIONS) + 0.4], [COMBINED[key]], s=26, facecolors='white',
                  edgecolors=col, linewidths=1.4, zorder=4)
        name = 'Grid (125)' if key == 'grid' else 'All 166'
        last = y[-1]
        c.text(len(VERSIONS) - 0.85, last - 0.0016, f'{name}\n{last:.4f}', fontsize=6.5,
               color=INK, ha='center', va='top')
        c.text(-0.3, WHISPER[key] + 0.0004, f'Whisper, {name.lower()}: {WHISPER[key]:.4f}',
               fontsize=6.5, color=INK2, va='bottom')
        c.text(len(VERSIONS) + 0.15, COMBINED[key], f'{COMBINED[key]:.4f}',
               fontsize=6.5, color=INK2, ha='right', va='center')
    c.set_xticks(list(xs) + [len(VERSIONS) + 0.4])
    c.set_xticklabels([v[0] for v in VERSIONS] + ['v20 +\nWhisper'])
    c.set_xlim(-0.4, len(VERSIONS) + 0.9)
    c.set_ylim(0.0735, 0.1065)
    c.set_ylabel('Test r')
    hgrid(c, 'y')
    c.set_title('c  Versions vs. Whisper', pad=5)

    d = fig.add_subplot(gs[1, 1])
    labs = [r[0] for r in REMOVED][::-1]
    vals = [r[2] for r in REMOVED][::-1]
    d.barh(range(len(vals)), vals, height=0.42, color=BLUE, zorder=3)
    for i, (v, lab) in enumerate(zip(vals, labs)):
        d.text(v + 6, i, f'{v}', va='center', fontsize=6.5, color=INK2)
        d.text(0, i + 0.27, lab, va='bottom', ha='left', fontsize=6.5, color=INK)
    d.set_yticks([])
    d.set_ylim(-0.5, len(vals) - 0.25)
    d.spines['left'].set_visible(False)
    d.set_xlim(0, 340)
    hgrid(d, 'x')
    d.set_xlabel('Raw columns removed (of 3,852)')
    d.set_title('d  Removed in v21 (460)', pad=5)
    fmt = lambda x: f'{x:+.4f}'.replace('-', '\u2212')
    pos = d.get_position()
    fig.text(pos.x0, 0.008, f'Also: voicing gates rewired\n(v20: stuck at 1.0).\n'
                            f'Test r, v21 \u2212 v20: grid {fmt(v21_grid - v20_grid)},\n'
                            f'all 166 {fmt(v21_all - v20_all)}', fontsize=6.5, color=INK2,
             ha='left', va='bottom')
    fig.suptitle('Hand-built audio features (LEAN) for patient1 high-gamma: progression',
                 x=0.02, ha='left', fontsize=10, fontweight='bold', color=INK)
    return fig


# ----------------------------------------------------------------------------- fig 2
def fig2():
    fam = load_varpart('v21', 'family')
    sec21 = {r['unit']: r for r in load_varpart('v21', 'section')}
    sec20 = {r['unit']: r for r in load_varpart('v20', 'section')}
    grp = load_varpart('v21', 'group')
    full = float(fam[0]['grid_alone_R2']) / float(fam[0]['grid_alone_share'])
    pct = lambda r, k: 100 * float(r[k])

    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.75, 1.1], height_ratios=[1, 1.35],
                          hspace=0.5, wspace=0.5, left=0.225, right=0.975, top=0.9, bottom=0.08)

    # a: family alone vs unique (aligned rows, separate scales)
    fam_s = sorted(fam, key=lambda r: -float(r['grid_unique_share']))
    lab = [FAM_SHORT[r['unit']] for r in fam_s]
    y = np.arange(len(fam_s))
    a1 = fig.add_subplot(gs[0, 0])
    al = [pct(r, 'grid_alone_share') for r in fam_s]
    a1.barh(y, al, height=0.55, color=LBLUE, zorder=3)
    for i, v in enumerate(al):
        a1.text(v - 2, i, f'{v:.0f}%', va='center', ha='right', fontsize=6.5, color='white',
                fontweight='bold')
    a1.set_yticks(y)
    a1.set_yticklabels([f'{l}  ({r["raw_dims"]})' for l, r in zip(lab, fam_s)])
    a1.set_ylim(len(y) - 0.5, -0.6)
    a1.set_xlim(0, 100)
    a1.tick_params(axis='y', length=0)
    a1.spines['left'].set_visible(False)
    hgrid(a1, 'x')
    a1.set_xlabel('Alone: % of full model')
    a1.set_title('a  Family alone vs. unique', pad=5)
    a2 = fig.add_subplot(gs[0, 1], sharey=a1)
    un = [pct(r, 'grid_unique_share') for r in fam_s]
    lo = [100 * float(r['grid_unique_lo']) / full for r in fam_s]
    hi = [100 * float(r['grid_unique_hi']) / full for r in fam_s]
    a2.axvline(0, color=CTX, linewidth=0.8)
    a2.hlines(y, lo, hi, color=BLUE, linewidth=1.2, zorder=3)
    a2.scatter(un, y, s=24, color=BLUE, edgecolors='white', linewidths=1.0, zorder=4)
    a2.tick_params(axis='y', length=0, labelleft=False)
    a2.spines['left'].set_visible(False)
    a2.set_xlim(-1, 8.5)
    hgrid(a2, 'x')
    a2.set_xlabel('Unique: % of full model\n(95% CI over test chunks)')

    # b: unique vs shared by granularity
    b = fig.add_subplot(gs[0, 2])
    levels = [('7 families', 'family'), ('16 sections', 'section'), ('51 groups', 'group')]
    us = [sum(float(r['grid_unique_share']) for r in load_varpart('v21', lv)) * 100 for _, lv in levels]
    yb = np.arange(len(levels))
    b.barh(yb, us, height=0.5, color=BLUE, zorder=3)
    b.barh(yb, [100 - u for u in us], left=[u + 0.6 for u in us], height=0.5, color='#dcdad5', zorder=3)
    for i, u in enumerate(us):
        b.text(u + 3, i, f'shared {100 - u:.0f}%', va='center', fontsize=6.5, color=INK2)
        b.text(-1.5, i + 0.42, f'unique {u:.1f}%', va='center', ha='left', fontsize=6.5, color=INK)
    b.set_yticks(yb)
    b.set_yticklabels([l for l, _ in levels])
    b.set_ylim(len(levels) - 0.3, -0.6)
    b.set_xlim(0, 100)
    b.tick_params(axis='y', length=0)
    b.spines['left'].set_visible(False)
    b.set_xlabel('% of explained variance')
    b.set_title('b  Unique vs. shared', pad=5)

    # c: section unique, v21 with CI and v20 for comparison
    c = fig.add_subplot(gs[1, 0:2])
    secs = sorted(sec21.values(), key=lambda r: float(r['grid_unique_share']))
    yc = np.arange(len(secs))
    u21 = [pct(r, 'grid_unique_share') for r in secs]
    lo21 = [100 * float(r['grid_unique_lo']) / full for r in secs]
    hi21 = [100 * float(r['grid_unique_hi']) / full for r in secs]
    u20 = [pct(sec20[r['unit']], 'grid_unique_share') for r in secs]
    c.axvline(0, color=CTX, linewidth=0.8)
    c.hlines(yc, lo21, hi21, color=BLUE, linewidth=1.2, zorder=3)
    c.scatter(u20, yc, s=22, facecolors='white', edgecolors=MUTED, linewidths=1.0, zorder=4,
              label='LEAN-v20')
    c.scatter(u21, yc, s=24, color=BLUE, edgecolors='white', linewidths=1.0, zorder=5,
              label='LEAN-v21 (95% CI)')
    c.set_yticks(yc)
    c.set_yticklabels([f'{r["unit"]}  {SEC_SHORT[r["unit"]]}  ({r["raw_dims"]})' for r in secs])
    c.tick_params(axis='y', length=0)
    c.spines['left'].set_visible(False)
    hgrid(c, 'x')
    c.set_xlabel('Unique variance: % of full model')
    c.legend(loc='lower right', handletextpad=0.3)
    c.set_title('c  Unique variance by section', pad=4)

    # d: alone share vs dimensionality, groups
    d = fig.add_subplot(gs[1, 2])
    dims = np.array([int(r['raw_dims']) for r in grp])
    alone = np.array([pct(r, 'grid_alone_share') for r in grp])
    sig = np.array([float(r['grid_unique_lo']) > 0 for r in grp])
    d.scatter(dims[~sig], alone[~sig], s=14, color=CTX, linewidths=0, zorder=3,
              label='Unique not > 0')
    d.scatter(dims[sig], alone[sig], s=22, color=BLUE, edgecolors='white', linewidths=0.8,
              zorder=4, label='Unique > 0 (95% CI)')
    names = {r['unit']: (int(r['raw_dims']), pct(r, 'grid_alone_share')) for r in grp}
    for g, (dx, dy, ha) in {'spectral_flatness': (4, 26, 'left'), 'phonetic': (-14, 20, 'right'),
                            'gate_signals': (16, -18, 'left'), 'tonic': (-10, -18, 'right'),
                            'interact_v3e': (-12, 8, 'right'), 'scattering': (-14, 4, 'right'),
                            'stmf_mean_gabor': (10, 8, 'left'), 'hnr_dyn': (12, 2, 'left')}.items():
        x0, y0 = names[g]
        d.annotate(g, (x0, y0), xytext=(dx, dy), textcoords='offset points', fontsize=6,
                   color=INK2, ha=ha, va='center',
                   arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.5, shrinkA=0, shrinkB=2))
    d.set_xscale('log')
    d.set_xlim(0.7, 900)
    d.set_ylim(-3, 100)
    hgrid(d, 'y')
    d.set_xlabel('Raw dimensions in group (log)')
    d.set_ylabel('Alone: % of full model')
    d.legend(loc='upper left', handletextpad=0.2, borderaxespad=0.2)
    d.set_title('d  Group size vs. alone', pad=5)

    fig.suptitle(f'LEAN-v21 variance partitioning (125 grid electrodes, full r = {v21_grid:.4f})',
                 x=0.02, ha='left', fontsize=10, fontweight='bold', color=INK)
    return fig


def write_tables(trials):
    with open(os.path.join(HERE, 'fig1_data.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['panel', 'item', 'n_trials', 'best_r', 'all166_r', 'grid_r', 'value'])
        for lab, _ in BRANCHES_166:
            v = trials[lab]; w.writerow(['a', lab, len(v), f'{v.max():.5f}', '', '', ''])
        for lab, _ in BRANCHES_GRID:
            v = trials[lab]; w.writerow(['b', lab, len(v), f'{v.max():.5f}', '', '', ''])
        for name, a_, g_ in VERSIONS:
            w.writerow(['c', name, '', '', f'{a_:.5f}', '' if g_ is None else f'{g_:.5f}', ''])
        w.writerow(['c', 'Whisper', '', '', WHISPER['all'], WHISPER['grid'], ''])
        w.writerow(['c', 'v20+Whisper', '', '', COMBINED['all'], COMBINED['grid'], ''])
        for _, lab, n in REMOVED:
            w.writerow(['d', lab, '', '', '', '', n])


if __name__ == '__main__':
    trials = load_trials()
    write_tables(trials)
    for name, fig in (('fig1_progression', fig1(trials)), ('fig2_breakdown', fig2())):
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(HERE, f'{name}.{ext}'))
        plt.close(fig)
    print('n trials:', sum(len(v) for v in trials.values()))
    print('saved to', HERE)
