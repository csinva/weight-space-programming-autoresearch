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
fig=plt.figure(figsize=(7.2,2.30),facecolor='white')
ax=fig.add_axes([0,0,1,1],xlim=(0,7.2),ylim=(0,2.70))
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

# NSD provides paired natural images and human 7 T fMRI responses.
box(.12,1.02,1.00,1.54,'#EEF5F4','none',.08)
text(.62,2.37,'NSD',9,TEAL,'bold')
text(.62,2.15,'7 T fMRI',7,TEAL)
t=np.linspace(0,1,100)
r=.03*np.sin(14*t)+.12*np.exp(-((t-.35)/.10)**2)+.09*np.exp(-((t-.75)/.09)**2)
ax.plot(.30+t*.64,1.87+r,color=TEAL,lw=1)
arrow([(1.16,1.99),(1.40,1.99),(1.40,2.36),(6.95,2.36),(6.95,1.64)],TEAL,.85)
text(4.00,2.36,'Recorded responses',7,TEAL,
     bbox={'facecolor':'white','edgecolor':'none','pad':2})
for x,y in [(.28,1.22),(.33,1.17)]:
    box(x,y,.57,.48,'white','#B6C3C9',.015,.65)
ax.add_patch(Rectangle((.35,1.19),.53,.44,facecolor='#E5EFF1',edgecolor='none'))
ax.add_patch(Circle((.75,1.52),.043,facecolor='#DCA65C',edgecolor='none'))
ax.add_patch(Polygon([(.35,1.19),(.35,1.32),(.54,1.55),(.82,1.19)],facecolor='#8EA9A1',edgecolor='none'))
ax.add_patch(Polygon([(.46,1.19),(.73,1.42),(.88,1.28),(.88,1.19)],facecolor='#567D79',edgecolor='none'))
text(.62,1.10,'Natural images',6.8)
arrow([(1.16,1.45),(1.76,1.45)])
box(1.84,1.13,1.13,.65,'#FCF5EF','#D2A17C',.06)
text(2.405,1.45,'$f$',18,ORANGE)
text(2.405,1.96,'Computational model',8,weight='bold')
text(2.405,.94,'Fixed weights',7,MUTED)
arrow([(3.04,1.45),(3.36,1.45)])
rng=np.random.default_rng(8)
for row in range(6):
    for col in range(5):
        ax.add_patch(Rectangle((3.44+col*.085,1.20+row*.085),.069,.069,
                    facecolor=TEAL,alpha=rng.uniform(.16,.82),edgecolor='none'))
text(3.65,1.96,'Features',8,weight='bold')
arrow([(3.95,1.45),(4.30,1.45)])
for a in np.linspace(1.20,1.70,4):
    for b in np.linspace(1.25,1.65,3):
        ax.plot([4.42,4.98],[a,b],color='#BDCED2',lw=.6)
for x,ys in [(4.42,np.linspace(1.20,1.70,4)),(4.98,np.linspace(1.25,1.65,3))]:
    for y in ys:
        ax.add_patch(Circle((x,y),.03,facecolor='white',edgecolor=TEAL,lw=.8))
text(4.70,1.96,'Ridge readout',8,weight='bold')
text(4.70,.94,'Fit on train split',7,MUTED)
arrow([(5.10,1.45),(5.58,1.45)])
ax.plot(5.73+t*1.13,1.31+r*1.9,color=TEAL,lw=1.1)
ax.plot(5.73+t*1.13,1.31+r*1.6+.018*np.cos(t*20),color=INK,lw=1.1,ls=(0,(3,2)))
text(6.30,1.96,'Compare',8,weight='bold')
text(6.30,.94,'Held-out stimuli',7,MUTED)
box(1.84,.15,1.13,.43,'#FCF5EF','#D2A17C',.07)
text(2.405,.365,'LLM agent',8.5,ORANGE,'bold')
arrow([(1.81,.365),(1.55,.365),(1.55,1.15),(1.80,1.15)],ORANGE,1.1)
text(1.48,.73,'Revise',7,ORANGE,ha='right')
arrow([(6.30,.79),(6.30,.365),(3.07,.365)],ORANGE,1.1)
text(4.60,.61,'Neural predictivity score',8,ORANGE)
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'fig1_loop.{ext}',dpi=300,facecolor='white')
plt.close(fig)
