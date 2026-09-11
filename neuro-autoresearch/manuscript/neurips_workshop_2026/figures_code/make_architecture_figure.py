"""Single-panel hierarchy and neural evaluation schematic; PDF, SVG and PNG.
All glyphs and response curves are schematic, not experimental data.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, Ellipse, FancyArrowPatch
from matplotlib.path import Path as MPath
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'neuro_autoresearch_latex' / 'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                     'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
INK, MUTED, LINE = '#25343E', '#64737C', '#AAB6BC'
ORANGE, TEAL = '#B86735', '#257D82'
fig=plt.figure(figsize=(7.2,2.65),facecolor='white')
ax=fig.add_axes([0,0,1,1],xlim=(0,9.6),ylim=(1.12,5.47))
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

# A true hierarchy: one conductor branches to multiple self-paced loops.
box(3.52,4.83,2.56,.48,'#ECF2F2','#AEBFC2',.09)
text(4.80,5.07,'Conductor',10,weight='bold')
ax.plot([4.80,4.80],[4.81,4.51],color=MUTED,lw=.85,ls=(0,(3,2)))
ax.plot([1.80,7.80],[4.51,4.51],color=MUTED,lw=.85,ls=(0,(3,2)))
text(4.80,4.64,'Advisory directions',7,MUTED,
     bbox={'facecolor':'white','edgecolor':'none','pad':2})
for cx in [1.80,4.80,7.80]:
    arrow([(cx,4.51),(cx,4.24)],MUTED,.85,dashed=True)

for i,cx in enumerate([1.80,4.80,7.80],1):
    box(cx-1.16,3.10,2.32,1.11,'#FCF8F4','#DCC5B3',.10,.75)
    text(cx,4.01,f'Autoresearch loop {i}',8.2,weight='bold')
    # Agent-to-model proposal and an explicit local revision cycle.
    text(cx-.58,3.62,'LLM agent',7.2,ORANGE,'bold')
    box(cx+.36,3.43,.46,.38,'white','#D2A17C',.04,.7)
    text(cx+.59,3.62,'$f$',13,ORANGE)
    arrow([(cx-.13,3.62),(cx+.28,3.62)],ORANGE,.95,6)
    arrow([(cx+.59,3.40),(cx+.59,3.25),(cx-.59,3.25),(cx-.59,3.43)],ORANGE,.9,6)
    text(cx,3.25,'Score → revise',6.7,ORANGE,
         bbox={'facecolor':'#FCF8F4','edgecolor':'none','pad':2})

# The shared database is the queue and persistent memory, not another agent.
# Each loop has its own return channel. These pairs summarize API interaction.
for cx in [1.80,7.80]:
    arrow([(cx,3.04),(cx,2.87)],ORANGE,.9,6,both=True)
ax.plot([1.80,7.80],[2.87,2.87],color=ORANGE,lw=.9)
arrow([(4.80,3.04),(4.80,2.72)],ORANGE,.9,6,both=True)
text(2.42,2.66,'Proposals / returned scores',7.1,ORANGE)

# Familiar cylinder glyph, large enough to recognize at manuscript size.
ax.add_patch(Rectangle((3.64,2.12),2.32,.42,facecolor='#EAF3F2',edgecolor=TEAL,lw=.9))
ax.add_patch(Ellipse((4.80,2.12),2.32,.24,facecolor='#EAF3F2',edgecolor=TEAL,lw=.9))
ax.add_patch(Rectangle((3.65,2.12),2.30,.22,facecolor='#EAF3F2',edgecolor='none'))
ax.add_patch(Ellipse((4.80,2.54),2.32,.24,facecolor='#F7FBFA',edgecolor=TEAL,lw=.9))
text(4.80,2.30,'SQLite database',9.2,TEAL,'bold')
text(4.80,1.83,'Queue · code · scores · ancestry',7,MUTED)

# The conductor reads the population via the database; dashed advice never
# touches the evaluation path. Route this dependency outside the search loops.
arrow([(6.03,2.52),(9.23,2.52),(9.23,5.07),(6.15,5.07)],TEAL,.85,7)
text(7.76,5.19,'Reads population history',7,TEAL)

# Worker pool: evaluation mechanics are introduced in Figure 1.
box(.60,1.15,8.40,.55,'#F3F6F7','none',.09)
for cx in [1.8,4.8,7.8]:
    box(cx-.95,1.20,1.90,.43,'#F3F6F7','#B9C9CC',.07)
    text(cx,1.415,'Evaluator worker',8,weight='bold')
arrow([(3.58,2.29),(1.80,2.29),(1.80,1.68)],ORANGE,1.0)

arrow([(7.80,1.69),(7.80,2.20),(6.03,2.20)],ORANGE,1.0)
text(2.65,2.06,'Queued models',7,ORANGE)
text(7.37,2.05,'Scores',7,ORANGE)
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'fig2_architecture.{ext}',dpi=300,facecolor='white')
plt.close(fig)
