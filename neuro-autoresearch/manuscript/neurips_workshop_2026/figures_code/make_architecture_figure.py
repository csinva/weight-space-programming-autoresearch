"""Publication schematic of hierarchical autoresearch; editable vector + PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
from matplotlib.path import Path as MPath
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'neuro_autoresearch_latex' / 'figures'
plt.rcParams.update({'font.family': 'Helvetica Neue', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans'})
INK, MUTED = '#25343E', '#64737C'
ORANGE, TEAL = '#B86735', '#257D82'
fig = plt.figure(figsize=(7.2, 2.95), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 7.2), ylim=(0, 2.95))
ax.set_axis_off()

def text(x, y, s, size=9, color=INK, weight='normal', ha='center', **kw):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight,
            ha=ha, va='center', **kw)

def box(x, y, w, h, fc, ec, radius=.07, lw=.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fc, edgecolor=ec, linewidth=lw))

def line(points, color, lw=.9):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round')

def arrow(points, color, lw=.95, scale=7):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>',
        mutation_scale=scale, color=color, linewidth=lw, joinstyle='round'))

# Level 1: orchestration. Only advice flows downward; loops remain self-paced.
box(2.06, 2.40, 2.28, .50, '#ECF3F3', '#96B6B8')
text(3.20, 2.74, 'Managing conductor', 10, weight='bold')
text(3.20, 2.54, 'Opus 4.8', 9.5, TEAL)
line([(3.2, 2.40), (3.2, 2.17)], MUTED)
line([(1.2, 2.17), (5.55, 2.17)], MUTED)
text(3.2, 2.28, 'Guides each loop', 8, MUTED,
     bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Level 2: model-diverse research loops; ellipsis denotes arbitrary population size.
for cx, label, model in [(1.2, 'Autoresearch loop 1', 'GPT-5.6'),
                          (3.2, 'Autoresearch loop 2', 'Opus 4.8'),
                          (5.55, r'Autoresearch loop $n$', 'Gemini 3.1 Pro')]:
    box(cx-.84, 1.09, 1.68, .80, 'white', '#D9C5B6', .055)
    arrow([(cx, 2.17), (cx, 1.91)], MUTED)
    text(cx, 1.70, label, 9, weight='bold')
    # Two open arcs make the same compact iteration glyph for every model.
    gx, gy, r = cx-.60, 1.35, .105
    for start in (30, 210):
        theta = np.deg2rad(np.linspace(start, start+145, 40))
        pts = list(zip(gx+r*np.cos(theta), gy+r*np.sin(theta)))
        arrow(pts, ORANGE, 1.15, 5.5)
    text(cx+.14, 1.35, model, 9.5, ORANGE)
text(4.375, 1.42, '···', 20, MUTED)

# Level 3: all loops write to one shared store. The collector is a single path.
for cx in (1.2, 3.2, 5.55):
    line([(cx, 1.09), (cx, .78)], ORANGE)
line([(1.2, .78), (5.55, .78)], ORANGE)
arrow([(3.2, .78), (3.2, .57)], ORANGE)
text(3.2, .81, 'Record experiments and results', 8, ORANGE,
     bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Restrained cylinder glyph, with text in the body rather than on its lid.
x, y, w, h, cap = 1.79, .15, 2.82, .32, .13
ax.add_patch(Ellipse((3.2, y), w, cap, fc='#EFF6F5', ec=TEAL, lw=.9))
ax.add_patch(Rectangle((x, y), w, h, fc='#EFF6F5', ec='none'))
line([(x, y), (x, y+h)], TEAL)
line([(x+w, y), (x+w, y+h)], TEAL)
ax.add_patch(Ellipse((3.2, y+h), w, cap, fc='#F9FCFB', ec=TEAL, lw=.9))
text(3.2, .295, 'Shared SQL database', 10, TEAL, 'bold')

# Population-level feedback closes the outer loop without crossing local loops.
arrow([(4.64, .30), (6.78, .30), (6.78, 2.62), (4.38, 2.62)], TEAL)
text(5.56, 2.77, 'Reviews shared progress', 8, TEAL)

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(OUT / f'fig2_architecture.{ext}', dpi=360, facecolor='white')
plt.close(fig)
