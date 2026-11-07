"""Figure 2, new held-out test: LEAN-v20 variance partitioning scored on the 317-chunk new test set
(runs-wave/varpart-v20-newtest), with the same fits on the old 23-chunk test (runs-wave/varpart-v20)
as open circles. Same layout and style as make_summary_figures.fig2."""
import os

import matplotlib.pyplot as plt
import numpy as np

from make_summary_figures import (BLUE, CTX, FAM_SHORT, HERE, INK, INK2, LBLUE, MUTED, SEC_SHORT,
                                  hgrid, load_varpart)

NEW, OLD = 'v20-newtest', 'v20'


def main():
    fam = load_varpart(NEW, 'family')
    fam_old = {r['unit']: r for r in load_varpart(OLD, 'family')}
    sec = {r['unit']: r for r in load_varpart(NEW, 'section')}
    sec_old = {r['unit']: r for r in load_varpart(OLD, 'section')}
    grp = load_varpart(NEW, 'group')
    full = float(fam[0]['grid_alone_R2']) / float(fam[0]['grid_alone_share'])
    full_old = float(fam_old[fam[0]['unit']]['grid_alone_R2']) / float(fam_old[fam[0]['unit']]['grid_alone_share'])
    pct = lambda r, k: 100 * float(r[k])

    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.75, 1.1], height_ratios=[1, 1.35],
                          hspace=0.5, wspace=0.5, left=0.225, right=0.975, top=0.9, bottom=0.08)

    # a: family alone vs unique (new test), old test unique as open circles
    fam_s = sorted(fam, key=lambda r: -float(r['grid_unique_share']))
    lab = [FAM_SHORT[r['unit']] for r in fam_s]
    y = np.arange(len(fam_s))
    a1 = fig.add_subplot(gs[0, 0])
    al = [pct(r, 'grid_alone_share') for r in fam_s]
    a1.barh(y, al, height=0.55, color=LBLUE, zorder=3)
    for i, v in enumerate(al):
        a1.text(v - 2, i, f'{v:.0f}%', va='center', ha='right', fontsize=6.5, color='white', fontweight='bold')
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
    un_old = [pct(fam_old[r['unit']], 'grid_unique_share') for r in fam_s]
    a2.axvline(0, color=CTX, linewidth=0.8)
    a2.hlines(y, lo, hi, color=BLUE, linewidth=1.2, zorder=3)
    a2.scatter(un_old, y, s=22, facecolors='white', edgecolors=MUTED, linewidths=1.0, zorder=4,
               label='old test')
    a2.scatter(un, y, s=24, color=BLUE, edgecolors='white', linewidths=1.0, zorder=5, label='new test')
    a2.legend(loc='lower right', fontsize=6, handletextpad=0.2, borderaxespad=0.1)
    a2.tick_params(axis='y', length=0, labelleft=False)
    a2.spines['left'].set_visible(False)
    a2.set_xlim(-1, 8.5)
    hgrid(a2, 'x')
    a2.set_xlabel('Unique: % of full model\n(95% CI over test chunks)')

    # b: unique vs shared by granularity (new test)
    b = fig.add_subplot(gs[0, 2])
    levels = [('7 families', 'family'), ('16 sections', 'section'), ('51 groups', 'group')]
    us = [sum(float(r['grid_unique_share']) for r in load_varpart(NEW, lv)) * 100 for _, lv in levels]
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

    # c: section unique, new test with CI, old test for comparison
    c = fig.add_subplot(gs[1, 0:2])
    secs = sorted(sec.values(), key=lambda r: float(r['grid_unique_share']))
    yc = np.arange(len(secs))
    u_new = [pct(r, 'grid_unique_share') for r in secs]
    lo_new = [100 * float(r['grid_unique_lo']) / full for r in secs]
    hi_new = [100 * float(r['grid_unique_hi']) / full for r in secs]
    u_old = [pct(sec_old[r['unit']], 'grid_unique_share') for r in secs]
    c.axvline(0, color=CTX, linewidth=0.8)
    c.hlines(yc, lo_new, hi_new, color=BLUE, linewidth=1.2, zorder=3)
    c.scatter(u_old, yc, s=22, facecolors='white', edgecolors=MUTED, linewidths=1.0, zorder=4,
              label='old 23-chunk test')
    c.scatter(u_new, yc, s=24, color=BLUE, edgecolors='white', linewidths=1.0, zorder=5,
              label='new 317-chunk test (95% CI)')
    c.set_yticks(yc)
    c.set_yticklabels([f'{r["unit"]}  {SEC_SHORT[r["unit"]]}  ({r["raw_dims"]})' for r in secs])
    c.tick_params(axis='y', length=0)
    c.spines['left'].set_visible(False)
    hgrid(c, 'x')
    c.set_xlabel('Unique variance: % of full model')
    c.legend(loc='lower right', handletextpad=0.3)
    c.set_title('c  Unique variance by section', pad=4)

    # d: alone share vs dimensionality, groups (new test)
    d = fig.add_subplot(gs[1, 2])
    dims = np.array([int(r['raw_dims']) for r in grp])
    alone = np.array([pct(r, 'grid_alone_share') for r in grp])
    sig = np.array([float(r['grid_unique_lo']) > 0 for r in grp])
    d.scatter(dims[~sig], alone[~sig], s=14, color=CTX, linewidths=0, zorder=3, label='Unique not > 0')
    d.scatter(dims[sig], alone[sig], s=22, color=BLUE, edgecolors='white', linewidths=0.8, zorder=4,
              label='Unique > 0 (95% CI)')
    names = {r['unit']: (int(r['raw_dims']), pct(r, 'grid_alone_share')) for r in grp}
    for g, (dx, dy, ha) in {'spectral_flatness': (0, 62, 'left'), 'gate_signals': (40, 28, 'left'),
                            'tonic': (-10, -18, 'right'), 'interact_v3e': (-12, 8, 'right'),
                            'scattering': (-14, 4, 'right')}.items():
        if g not in names:
            continue
        x0, y0 = names[g]
        d.annotate(g, (x0, y0), xytext=(dx, dy), textcoords='offset points', fontsize=6, color=INK2,
                   ha=ha, va='center',
                   arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.5, shrinkA=0, shrinkB=2))
    d.set_xscale('log')
    d.set_xlim(0.7, 900)
    d.set_ylim(-3, 100)
    hgrid(d, 'y')
    d.set_xlabel('Raw dimensions in group (log)')
    d.set_ylabel('Alone: % of full model')
    d.legend(loc='upper left', handletextpad=0.2, borderaxespad=0.2)
    d.set_title('d  Group size vs. alone', pad=5)

    r_full = float(np.sqrt(full))
    fig.suptitle(f'LEAN-v20 variance partitioning, new held-out test (grid; full R² {100 * full:.2f}%, '
                 f'old test {100 * full_old:.2f}%)',
                 x=0.02, ha='left', fontsize=9.5, fontweight='bold', color=INK)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'fig2_breakdown_newtest.{ext}'))
    print('full R2 new', round(100 * full, 3), 'old', round(100 * full_old, 3), 'sqrt', round(r_full, 4),
          '| unique sums', [round(u, 1) for u in us])


if __name__ == '__main__':
    main()
