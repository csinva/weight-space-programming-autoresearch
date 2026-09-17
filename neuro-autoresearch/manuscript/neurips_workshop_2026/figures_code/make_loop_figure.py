"""Single-agent autoresearch and neural evaluation schematic; PDF, SVG and PNG.
All glyphs and response curves are schematic, not experimental data.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, Ellipse, FancyArrowPatch
from matplotlib.path import Path as MPath
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'neuro_autoresearch_latex' / 'figures'
# Arial is embedded in PDF; use its italic face for matching math glyphs.
font_dir = Path('/System/Library/Fonts/Supplemental')
for face in ('Arial.ttf', 'Arial Italic.ttf', 'Arial Bold.ttf'):
    if (font_dir / face).exists():
        font_manager.fontManager.addfont(font_dir / face)
plt.rcParams.update({'font.family':'Arial', 'pdf.fonttype':42,
                     'svg.fonttype':'none', 'mathtext.fontset':'custom',
                     'mathtext.rm':'Arial', 'mathtext.it':'Arial:italic',
                     'mathtext.bf':'Arial:bold', 'mathtext.sf':'Arial',
                     'lines.solid_capstyle':'round'})
INK, MUTED, LINE = '#29343A', '#627078', '#A5B2B8'
ORANGE, TEAL = '#B86735', '#257D82'
fig=plt.figure(figsize=(8.0,2.74),facecolor='white')
ax=fig.add_axes([0,0,1,1],xlim=(0,8.0),ylim=(.30,3.04))
ax.set_axis_off()

def text(x,y,s,size=8,color=INK,weight='normal',ha='center',**kw):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',**kw)

def box(x,y,w,h,fc='white',ec=LINE,r=.07,lw=.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                 facecolor=fc,edgecolor=ec,linewidth=lw))

def arrow(points,color=LINE,lw=1,scale=7,dashed=False,both=False):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='<|-|>' if both else '-|>',
        mutation_scale=scale,color=color,linewidth=lw,joinstyle='round',
        linestyle=(0,(3,2)) if dashed else '-'))

# NSD illustration is generated and illustrative, not a dataset sample.
box(.07,1.02,1.35,1.97,'white','#C8D9D8',.08)
text(.745,2.79,'Natural Scenes\nDataset (NSD)',7.4,TEAL,'bold',linespacing=1.2)
text(.745,2.48,'Human 7 T fMRI',7.3,TEAL)
asset = OUT / 'nsd_illustration.png'
if asset.exists():
    ax.imshow(plt.imread(asset), extent=(.16,1.33,1.29,2.39), aspect='auto', zorder=2)
text(.745,1.13,'Natural images',7.5)
# Two visual levels: stimulus-to-prediction above, scalar feedback below.
# Group the readout and comparison to identify a single fixed evaluator.
box(3.72,1.01,4.13,1.34,'#F3F6F6','none',.07)
text(5.78,2.17,'Fixed evaluator',9,INK,'bold')
ax.plot([5.62,5.62],[1.18,1.93],color='#D8E2E3',lw=.7)

# Model and evaluator align along one horizontal stimulus pathway.
text(2.61,2.17,'Proposed model',9,INK,'bold')
box(2.02,1.26,1.18,.64,'#FCF7F2','#C99B7A',.06,.85)
text(2.61,1.68,'$f(x)$',16,ORANGE)
text(2.61,1.38,'Fixed weights',7.5,MUTED)
arrow([(1.48,1.58),(1.94,1.58)],LINE,1.0)
arrow([(3.27,1.58),(3.65,1.58)],LINE,1.0)

# Readout is a mathematical mapping rather than a neural-network pictogram.
text(4.69,1.60,r'$\hat{y}=Wf(x)$',14,INK)
text(4.69,1.19,'Ridge readout',7.5,MUTED)

# Schematic neural comparison, with a small direct legend.
t=np.linspace(0,1,140)
y=.055*np.sin(t*11)+.19*np.exp(-((t-.37)/.12)**2)+.12*np.exp(-((t-.76)/.13)**2)
ax.plot(5.91+t*1.60,1.45+y,color=TEAL,lw=1.3)
ax.plot(5.91+t*1.60,1.45+.84*y+.015*np.cos(t*17),color=INK,lw=1.15,ls=(0,(3,2)))
text(6.73,1.19,'Held-out responses',7.5,MUTED)
ax.plot([6.00,6.15],[1.88,1.88],color=TEAL,lw=1.2)
text(6.22,1.88,r'Data ($y$)',7.5,MUTED,ha='left')
ax.plot([6.74,6.89],[1.88,1.88],color=INK,lw=1.1,ls=(0,(3,2)))
text(6.96,1.88,r'Model ($\hat{y}$)',7.5,MUTED,ha='left')

# Recorded targets travel directly to the evaluator; never into model code.
arrow([(1.48,2.52),(6.73,2.52),(6.73,2.39)],TEAL,.95)
text(4.12,2.52,r'Recorded responses $y$',7.5,TEAL,
     bbox={'facecolor':'white','edgecolor':'none','pad':3})

# One compact feedback route, anchored to the model and evaluator.
box(2.02,.40,1.18,.43,'#FCF7F2','#C99B7A',.06,.85)
text(2.61,.615,'LLM agent',8.5,ORANGE,'bold')
arrow([(2.61,.87),(2.61,1.19)],ORANGE,1.1)
text(2.40,1.02,'Revise code',7.5,ORANGE,ha='right')
arrow([(6.73,.94),(6.73,.615),(3.28,.615)],ORANGE,1.1)
text(5.12,.83,'Reward: neural predictivity',8,ORANGE)
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'fig1_loop.{ext}',dpi=300,facecolor='white')
plt.close(fig)
