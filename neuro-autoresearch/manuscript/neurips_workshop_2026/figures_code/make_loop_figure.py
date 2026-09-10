"""Integrated, single-panel search schematic: PDF, SVG and PNG.

Run from the research repository with its Python environment. All glyphs and
response curves are schematic, not experimental data.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, FancyArrowPatch
from matplotlib.path import Path as MPath
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'neuro_autoresearch_latex' / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'mathtext.fontset': 'dejavusans'})
INK, MUTED, LINE = '#25343E', '#64737C', '#AAB6BC'
ORANGE, TEAL = '#B86735', '#257D82'
fig = plt.figure(figsize=(7.2, 3.45), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 9.6), ylim=(0, 4.6))
ax.set_axis_off()

def text(x, y, s, size=8, color=INK, weight='normal', ha='center', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   ha=ha, va='center', **kw)

def box(x, y, w, h, fc='white', ec=LINE, r=.07, lw=.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    return p

def arrow(points, color=LINE, lw=1.0, scale=7, dashed=False):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=scale,
                                color=color, linewidth=lw, joinstyle='round',
                                linestyle=(0, (3, 2)) if dashed else '-'))

def code(x, y, w=.42, color=ORANGE):
    box(x, y, w, .32, '#FCF5EF', '#D2A17C', .025, .6)
    for dy, indent, length in [(.24, .06, .24), (.16, .11, .21), (.08, .06, .18)]:
        ax.plot([x+indent, x+indent+length*w/.42], [y+dy]*2,
                lw=.9, color=color, solid_capstyle='round')

# One composition, two roles. There are no lettered or separate subpanels.
text(1.48, 4.34, 'Explore in parallel', 10, weight='bold')
box(3.69, 1.35, 5.70, 2.75, '#F3F6F7', 'none', .13)
text(6.54, 4.34, 'Evaluate each computational model', 10, weight='bold')

# Conductor advice is visually lighter than the proposal / score paths.
box(.35, 3.57, 2.28, .43, '#F2F5F5', '#C1CDCF', .07)
text(1.49, 3.785, 'Conductor', 8.5, weight='bold')
arrow([(1.49, 3.54), (1.49, 3.07)], MUTED, .8, dashed=True)
text(1.60, 3.31, 'Advises', 7, MUTED, ha='left')
box(.35, 1.39, 2.28, 1.65, '#FCF8F4', '#E3D2C3', .10, .7)
text(1.49, 2.85, 'Independent search islands', 7.2, weight='bold')
# Each row depicts one agent's own lineage, without implying a global parent.
for i, y in enumerate([2.45, 2.00, 1.55]):
    text(.88, y+.16, f'Agent {i+1}', 7.6, ORANGE, 'bold')
    arrow([(1.30, y+.16), (1.58, y+.16)], ORANGE, .8, 5)
    code(1.63, y)
    code(2.10, y)
# Collect proposals asynchronously.
ax.plot([2.73, 2.73], [1.71, 2.61], color=LINE, lw=.8)
for y in [2.61, 2.16, 1.71]:
    ax.plot([2.65, 2.73], [y, y], color=LINE, lw=.8)
arrow([(2.73, 2.61), (2.94, 2.61)], ORANGE, 1.0)
for x, y in [(2.98, 2.47), (3.03, 2.42), (3.08, 2.37)]:
    box(x, y, .37, .48, 'white', '#BFC8CC', .025, .65)
text(3.19, 2.12, 'Queue', 7.4, weight='bold')
# Submitted code enters the computational-model stage; images have a separate input.
arrow([(3.49, 2.61), (3.57, 2.61), (3.57, 3.48),
       (5.30, 3.48), (5.30, 3.02)], ORANGE, 1.05)
text(2.94, 3.19, 'Propose', 7.2, ORANGE)

# Common evaluation path, used by every fixed evaluator worker.
for x, y in [(3.89, 2.40), (3.94, 2.35), (3.99, 2.30)]:
    box(x, y, .43, .54, '#FFFFFF', '#B6C3C9', .015, .65)
ax.add_patch(Rectangle((4.01, 2.32), .39, .50, facecolor='#E5EFF1', edgecolor='none'))
ax.add_patch(Circle((4.31, 2.70), .043, facecolor='#DCA65C', edgecolor='none'))
ax.add_patch(Polygon([(4.01,2.32),(4.01,2.46),(4.16,2.68),(4.37,2.32)], facecolor='#8EA9A1', edgecolor='none'))
ax.add_patch(Polygon([(4.09,2.32),(4.29,2.56),(4.40,2.42),(4.40,2.32)], facecolor='#567D79', edgecolor='none'))
text(4.17, 2.06, 'Images', 7.6, weight='bold')
arrow([(4.51, 2.61), (4.82, 2.61)])

box(4.91, 2.27, .78, .68, '#FCF5EF', '#D2A17C', .06)
text(5.30, 2.61, '$f$', 19, ORANGE)
text(5.30, 2.06, 'Model', 7.6, weight='bold')
text(5.30, 1.79, 'Fixed weights', 6.8, MUTED)
arrow([(5.77, 2.61), (6.03, 2.61)])

rng = np.random.default_rng(8)
for row in range(6):
    for col in range(5):
        ax.add_patch(Rectangle((6.12+col*.093, 2.33+row*.093), .077, .077,
                               facecolor=TEAL, alpha=rng.uniform(.16,.82), edgecolor='none'))
text(6.35, 2.06, 'Features', 7.6, weight='bold')
arrow([(6.65, 2.61), (6.95, 2.61)])

for a in np.linspace(2.38, 2.84, 4):
    for b in np.linspace(2.43, 2.79, 3):
        ax.plot([7.05,7.55], [a,b], color='#BDCED2', lw=.6)
for x, ys in [(7.05,np.linspace(2.38,2.84,4)), (7.55,np.linspace(2.43,2.79,3))]:
    for y in ys:
        ax.add_patch(Circle((x,y), .032, facecolor='white', edgecolor=TEAL, lw=.8))
text(7.30, 2.06, 'Readout', 7.6, weight='bold')
text(7.30, 1.79, 'Ridge · train split', 6.8, MUTED)
arrow([(7.66,2.61), (7.98,2.61)])

t=np.linspace(0,1,120)
r=.045*np.sin(14*t)+.12*np.exp(-((t-.27)/.07)**2)+.15*np.exp(-((t-.70)/.08)**2)
ax.plot(8.08+t*1.02,2.46+r*1.8,color=INK,lw=1.05)
ax.plot(8.08+t*1.02,2.46+r*1.45+.034*np.cos(t*22),color=TEAL,lw=1.05,linestyle=(0,(3,2)))
ax.plot([8.04,8.16],[3.10]*2,color=INK,lw=.9)
text(8.20,3.10,'Brain',6.5,ha='left')
ax.plot([8.64,8.76],[3.10]*2,color=TEAL,lw=.9,linestyle=(0,(3,2)))
text(8.80,3.10,'Model',6.5,TEAL,ha='left')
text(8.60,2.06,'Compare',7.6,weight='bold')
text(8.60,1.79,'Held-out stimuli',6.8,MUTED)
text(6.54,1.52,'Same fixed evaluation procedure across workers',6.9,MUTED)

# Shared history sits on the return path. The caption explains what is stored
# and how the conductor reads it; no duplicate network of administrative arrows.
box(4.80,.35,2.10,.54,'#FCF8F4','#E3D2C3',.09,.7)
text(5.85,.70,'Shared history',7.7,weight='bold')
text(5.85,.47,'Code · scores · ancestry',6.6,MUTED)
path=MPath([(8.60,1.30),(8.60,.86),(8.60,.62),(8.33,.62),(6.99,.62)],
           [MPath.MOVETO,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO])
ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7,color=ORANGE,lw=1.2))
text(7.71,.94,'Score',8,ORANGE)
path=MPath([(4.71,.62),(1.73,.62),(1.49,.62),(1.49,.86),(1.49,1.32)],
           [MPath.MOVETO,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO])
ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7,color=ORANGE,lw=1.2))
text(3.18,.91,'Revise and repeat',8,ORANGE)

for ext in ('pdf','svg','png'):
    fig.savefig(OUT / f'fig1_loop.{ext}',dpi=300,facecolor='white')
plt.close(fig)
