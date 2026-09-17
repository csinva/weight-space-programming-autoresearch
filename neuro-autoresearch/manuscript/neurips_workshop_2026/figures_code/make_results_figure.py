"""Figure 3: selected baselines on individuals excluded from model selection.

Bars are equal-weight held-out individual means; NSD symbols show both subjects.
No uncertainty interval is inferred from two subjects or one monkey. Read the
exported evaluation files only; never mix search rewards with held-out scores.
Run with .venv/bin/python from the repository root. Outputs include source data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / 'neuro_autoresearch_latex' / 'figures'
INK, MUTED, GRID = '#202A32', '#64717B', '#E5E9EC'
REFERENCE = '#81786E'  # warm gray, distinct from the blue-gray model bars
ACCENT, UNTRAINED, PRETRAINED = '#CE691E', '#C4CDD3', '#536C7C'
SERIES = [
    ('champion', 'Discovered model', 0, ACCENT),
    ('random_projection', 'Random projection', None, REFERENCE),
    ('alexnet_untrained', 'AlexNet', 1.8, UNTRAINED),
    ('resnet50_untrained', 'ResNet-50', 2.8, UNTRAINED),
    ('alexnet_pretrained', 'AlexNet', 4.6, PRETRAINED),
    ('resnet50_pretrained', 'ResNet-50', 5.6, PRETRAINED),
    ('dinov2_pretrained', 'DINOv2', 6.6, PRETRAINED),
    ('clip_pretrained', 'CLIP', 7.6, PRETRAINED),
]
PANELS = [
    dict(dataset='nsd', letter='a', title='Human fMRI · NSD',
         subtitle='Discovered: multiscale color-opponent energy + texture statistics',
         cohort='Held-out subjects 5 and 7', regions=['V1', 'V2', 'V3', 'ventral'],
         metric='Noise-normalized explained variance', xmax=.70, ticks=[0, .3, .6],
         bottom=.605, top=.855, title_y=.972, subtitle_y=.927, cohort_y=.901),
    dict(dataset='tvsd', letter='b', title='Macaque spiking · TVSD',
         subtitle='Discovered: multiscale Gabor filters + texture and form pathways',
         cohort='Held-out monkey N', regions=['V1', 'V4', 'IT'],
         metric='Mean Pearson correlation', xmax=.95, ticks=[0, .4, .8],
         bottom=.108, top=.368, title_y=.488, subtitle_y=.444, cohort_y=.418),
]


def load_values(dataset):
    ev = json.loads((ROOT / 'runs' / 'best' / dataset / 'evaluation.json').read_text())
    expected = set(ev['cohorts']['heldout'])
    for key, _, _, _ in SERIES:
        subjects = ev['models'][key]['roi_scores']['heldout']['subjects']
        if set(subjects) != expected:
            raise ValueError(f'{dataset}/{key}: inconsistent held-out cohort')
    return ev


def draw_panel(fig, panel, rows):
    ev = load_values(panel['dataset'])
    left, right, gap = .223, .979, .032
    width = (right - left - gap * (len(panel['regions']) - 1)) / len(panel['regions'])
    fig.text(.015, panel['title_y'], panel['letter'], fontsize=11, weight='bold', va='top')
    fig.text(.055, panel['title_y'], panel['title'], fontsize=10, weight='bold', va='top')
    fig.text(.055, panel['subtitle_y'], panel['subtitle'], fontsize=8, color=ACCENT, va='top')
    fig.text(.055, panel['cohort_y'], panel['cohort'], fontsize=7.3, color=MUTED, va='top')
    for col, region in enumerate(panel['regions']):
        ax = fig.add_axes([left + col * (width + gap), panel['bottom'], width,
                           panel['top'] - panel['bottom']])
        ax.set_xlim(0, panel['xmax'])
        ax.set_ylim(8.2, -.9)
        for key, label, y, color in SERIES:
            model = ev['models'][key]
            subjects = model['roi_scores']['heldout']['subjects']
            vals = [float(subjects[s][region]) for s in ev['cohorts']['heldout']]
            if not np.isfinite(vals).all():
                raise ValueError(f'Nonfinite score: {panel["dataset"]}/{key}/{region}')
            mean = float(np.mean(vals))
            for subject, value in zip(ev['cohorts']['heldout'], vals):
                rows.append(dict(dataset=panel['dataset'], region=region, model=key,
                                 model_id=model.get('model_id', ''), cohort='heldout',
                                 subject=subject, score=value, plotted_mean=mean))
            if key == 'random_projection':
                ax.axvline(mean, color=REFERENCE, lw=1, ls=(0, (3, 2)), zorder=3)
                continue
            ax.barh(y, mean, height=.65, color=color, edgecolor='none', zorder=2)
            if panel['dataset'] == 'nsd':
                for j, value in enumerate(vals):
                    ax.plot(value, y, marker=['o', 'D'][j], ms=2.8,
                            mfc='white', mec=INK, mew=.55, ls='none', zorder=4)
            if key == 'champion':
                ax.text(max(vals) + .022 * panel['xmax'], y, f'{mean:.2f}',
                        va='center', fontsize=7.1, weight='bold', color=ACCENT)
            if col == 0:
                ax.text(-.065, y, label, transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=7.8,
                        color=ACCENT if key == 'champion' else INK,
                        weight='bold' if key == 'champion' else 'normal', clip_on=False)
        if col == 0:
            for y, label in [(1.02, 'UNTRAINED BASELINES'), (3.82, 'PRETRAINED ENCODERS')]:
                ax.text(-.065, y, label, transform=ax.get_yaxis_transform(), ha='right',
                        va='center', fontsize=6.5, color=MUTED, weight='bold', clip_on=False)
        ax.set_title('Ventral' if region == 'ventral' else region,
                     fontsize=8, weight='bold', pad=3)
        ax.set_yticks([])
        ax.set_xticks(panel['ticks'])
        ax.set_xticklabels(['0' if v == 0 else f'{v:.1f}' for v in panel['ticks']])
        ax.tick_params(axis='x', labelsize=9.75, length=2.5, width=.5, colors=MUTED, pad=2)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=GRID, linewidth=.5)
        for side in ['left', 'right', 'top']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.spines['bottom'].set_linewidth(.6)
    fig.text((left + right) / 2, panel['bottom'] - .067, panel['metric'],
             ha='center', va='top', fontsize=10.5, color=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--stem', default='fig3_results')
    parser.add_argument('--fmt', nargs='+', default=['pdf', 'png', 'svg'])
    args = parser.parse_args()
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next(f for f in ['Helvetica Neue', 'Arial', 'DejaVu Sans'] if f in available)
    plt.rcParams.update({'font.family': font, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
    fig = plt.figure(figsize=(7.2, 4.9), facecolor='white')
    rows = []
    for panel in PANELS:
        draw_panel(fig, panel, rows)
    handles = [Line2D([], [], ls='none', marker=m, ms=3.4, mfc='white', mec=INK,
                      mew=.6, label=f'Subject {s}') for m, s in [('o', 5), ('D', 7)]]
    handles.append(Line2D([], [], color=REFERENCE, lw=1, ls=(0, (3, 2)),
                          label='Random projection'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.985, .978),
               ncol=3, frameon=False, fontsize=6.8, handletextpad=.35,
               columnspacing=.9, borderaxespad=0)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in args.fmt:
        path = args.out / f'{args.stem}.{ext}'
        fig.savefig(path, dpi=300, facecolor='white')
        print(path)
    with (args.out / f'{args.stem}_source_data.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)


if __name__ == '__main__':
    main()
