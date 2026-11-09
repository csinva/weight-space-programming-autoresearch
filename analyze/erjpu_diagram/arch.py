"""Architecture of the final LAION visual cortex model (erjpu), drawn from its feature atlas (atlas.json).

Feature counts per block are the model's own output layout. Nonlinearity badges follow the channel definitions
(atlas names, figures/fmri/d_families.py) and the model description in the paper appendix.
Usage: python arch.py atlas.json out.png
"""
import json, sys, textwrap
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'Downloads/neuro/neuro_autoresearch_latex/figures_code'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from palette import PRIMARY, SECONDARY, TERTIARY
for f in ('Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf'):
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/' + f)
plt.rcParams.update({'font.family': 'Arial'})
INK, MUTED, HAIR, PANEL = '#202A32', '#64717B', '#C9D0D6', '#F6F8FA'
A = json.load(open(sys.argv[1]))
W = {b['name']: b['width'] for b in A['blocks']}
assert len(W) == 36 and sum(W.values()) == 9996
NL = {'P': ('#8E5BB5', 'power-law compression'), 'Q': (PRIMARY, 'quadrature energy  √(even² + odd²)'),
      'R': (SECONDARY, 'half-wave rectification  max(±x, 0)'), 'D': (TERTIARY, 'divisive normalization  x / (σ + pool)'),
      'M': ('#D9822B', 'multiplication of channel pairs'), 'T': ('#B8860B', 'analytic tuning curves of the shape index'),
      'S': ('#5B6770', 'pooling: mean, spread (std), peak (max)')}

fig = plt.figure(figsize=(17, 14))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 140); ax.axis('off')
def box(x, y, w, h, fc='white', ec=HAIR, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec=ec, lw=lw))
def txt(x, y, s, **k):
    k.setdefault('fontsize', 9); k.setdefault('color', INK); k.setdefault('va', 'top'); ax.text(x, y, s, **k)
def arrow(p0, p1, c=MUTED, lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=9, color=c, lw=lw, shrinkA=0, shrinkB=0))
def badge(x, y, k):
    ax.text(x, y, k, fontsize=7.5, color='white', fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.22', fc=NL[k][0], ec='none'))
def num(x, y, n, fs=8.6, c=INK):
    txt(x, y, f'{n:,}', fontsize=fs, fontweight='bold', ha='right', color=c)

txt(2, 138.6, 'The final visual cortex model (erjpu): every stage, its channels, the features it emits and where the nonlinearities are', fontsize=14, fontweight='bold')
txt(2, 135.4, 'No learned weights: every filter is set analytically or from a fixed seed, and ridge regression is the only fitted step. Feature counts are the model’s own output '
    'layout (36 blocks, 9,996 features); nonlinearity badges follow its channel definitions and the description in the paper appendix.', fontsize=9, color=MUTED)

# input
box(2, 104, 20, 27, fc=PANEL)
txt(12, 129.6, 'Input', fontsize=11, fontweight='bold', ha='center')
txt(12, 126.2, 'RGB image\n(LAION or NSD photo)', fontsize=9, ha='center', linespacing=1.15)
txt(3.5, 120.5, 'Pixel compression', fontsize=9, fontweight='bold'); badge(19.6, 119.6, 'P')
txt(3.5, 117.6, 'intensity raised to a power', fontsize=8.3, color=MUTED)
txt(3.5, 113.6, 'Opponent channels', fontsize=9, fontweight='bold')
txt(3.5, 110.8, 'luminance L, red−green RG,\nblue−yellow BY (linear mix)', fontsize=8.3, color=MUTED, linespacing=1.15)

# V1-like stage
box(26, 80, 34, 51, fc='#EEF4FA', ec=PRIMARY, lw=1.3)
txt(43, 129.6, 'V1-like stage', fontsize=11, fontweight='bold', ha='center', color=PRIMARY)
txt(43, 126.4, '30 channels at 3 octaves (image pyramid: o0 fine, o1, o2)', fontsize=8.4, ha='center', color=MUTED)
rows = [('16', 'luminance Gabor energies', '8 orientations (0–157.5°) × 2 scales', 'Q'),
        ('2', 'luminance blobs', 'bright and dark, difference of Gaussians', 'R'),
        ('8', 'color Gabor energies', 'RG and BY × 4 orientations', 'Q'),
        ('4', 'color blobs', 'red, green, yellow, blue', 'R')]
for i, (n, a, b, k) in enumerate(rows):
    y = 121.8 - i * 7.4
    txt(29.6, y + 0.2, n, fontsize=12, fontweight='bold', ha='center', color=PRIMARY)
    txt(32.6, y, a, fontsize=9.2, fontweight='bold'); txt(32.6, y - 2.6, b, fontsize=8.2, color=MUTED); badge(57.4, y - 1.3, k)
txt(28, 91.4, 'Then each channel is divided', fontsize=9, fontweight='bold'); badge(57.4, 90.4, 'D')
txt(28, 88.6, 'by a pool of local energy', fontsize=9, fontweight='bold')
txt(28, 85.4, '(intermediate maps also keep small and large\nblob sizes, 36 maps in all)', fontsize=7.8, color=MUTED, linespacing=1.15)
arrow((22.2, 117), (25.8, 117))
txt(13, 79.6, 'opponent channels', fontsize=7.6, color=MUTED, va='bottom')

modules = [
 ('V1 maps pooled on foveated grids', 'V1', 'V1-like stage', '30 channels per grid cell; cells are finer near the centre of gaze.', 'S', PRIMARY,
  [('o0: mean on 9×9 grid', W['s1_cort_mean_o0']), ('o0: spread and peak on 5×5', W['s1_std_peak_o0']),
   ('o1: mean on 6×6', W['s1_cort_mean_o1']), ('o1: spread and peak on 3×3', W['s1_std_peak_o1']),
   ('o2: mean on 4×4', W['s1_cort_mean_o2']), ('o2: spread and peak on 2×2', W['s1_std_peak_o2']),
   ('whole image: mean and spread', W['s1_global_stats'])]),
 ('Raw color layout', 'IN', 'opponent channels', 'L, RG and BY averaged on a 6×6 grid, with no filtering.', 'S', '#7A8590',
  [('6×6 grid × 3 channels', W['color_grid'])]),
 ('Texture boundaries (filter–rectify–filter)', 'V1', 'V1 energy maps, one octave coarser',
  'A second Gabor bank applied to the energy maps and squared, so it responds where patterns of oriented contrast change.', 'QS', '#9C6ADE',
  [('3×3 grid', W['second_order_grid']), ('whole image: mean and spread', W['second_order_stats'])]),
 ('Orientation field and contour curvature', 'V1', 'V1 energies at 3 octaves',
  '13 channels: 2nd and 4th orientation harmonics and their strength (junctions), curvature split by sign into +x, −x, +y, −y, '
  '|curvature|, iso- and cross-orientation surround.', 'MRS', '#C2410C',
  [('o0 and o1: 4×4 grids', W['orient_field_o0'] + W['orient_field_o1']), ('o2: 2×2 grid, mean and spread', W['orient_field_o2'])]),
 ('Edge phase', 'V1', 'even and odd Gabor responses at 3 octaves',
  '9 channels: bright line, dark crease, line vs edge, phase coherence, fine vs coarse energy, brightness gradient split into +x, −x, +y, −y.', 'RS', '#A0522D',
  [('o0 and o1: 4×4 grids', W['phase_o0'] + W['phase_o1']), ('o2: 2×2 grid, mean and spread', W['phase_o2'])]),
 ('Surface geometry', 'IN', 'smoothed luminance derivatives (Hessian, structure tensor) at 5 octaves',
  '14 channels: 5 shading shapes from the shape index (cup, rut, saddle, ridge, dome), corner and edge strength, edge coherence, '
  'isophote curvature split by sign, |Lxx−Lyy|, |2Lxy|, red−green and blue−yellow 2D structure.', 'TRS', '#B8860B',
  [('o0: mean 4×4 + spread 3×3', W['geom_o0_mean'] + W['geom_o0_std']), ('o1: mean 4×4 + spread 3×3', W['geom_o1_mean'] + W['geom_o1_std']),
   ('o2: mean 3×3 + spread', W['geom_o2_mean'] + W['geom_o2_std']), ('o3: 2×2 grid, mean and spread', W['geom_o3']),
   ('o4: mean and spread', W['geom_o4'])]),
 ('Co-occurrences (texture statistics)', 'V1', 'normalized channels of the stages above',
  'Products of channel pairs averaged over space, in the spirit of Portilla–Simoncelli texture statistics.', 'MS', '#D9822B',
  [('V1 channels within each scale', W['cov_within_scale']), ('V1 channels within image quadrants', W['cov_quadrant']),
   ('curvature, phase, geometry, color', W['cov_joint']), ('persistence from octave to octave', W['octave_persist_diff']),
   ('fine channels × their coarse parents', W['octave_parent_child_cov']),
   ('geometry channels across fine scales', W['geom_rel_fine'] + W['geom_comp_fine']), ('geometry channels across coarse scales', W['geom_rel_coarse'])]),
 ('Global layout', 'V1', 'summary channels and a 32-pixel image',
  'Coarse paths of contrast, light and dark extrema that persist across thresholds, left–right symmetry and centre versus periphery.', 'RS', '#5B6770',
  [('coarse contrast paths', W['path']), ('persistent light and dark extrema', W['persistence']),
   ('symmetry, centre vs periphery', W['ecc_symmetry']), ('symmetry of geometry channels', W['geom_symmetry'])])]

y, total = 129.4, 0
txt(93, 130.2, 'Modules: intermediate maps', fontsize=10, fontweight='bold', ha='center', color=MUTED, va='bottom')
txt(146, 130.2, 'Pooled output features', fontsize=10, fontweight='bold', ha='center', color=MUTED, va='bottom')
for name, srckey, src, note, nls, c, blocks in modules:
    lines = textwrap.wrap(note, 74)
    h = max(6.6 + 2.15 * len(lines) + 0.8, 2.35 * len(blocks) + 4.6)
    top = y; y -= h + 1.0
    box(64, top - h, 58, h, ec=c, lw=1.1); box(126, top - h, 41, h, fc=PANEL)
    txt(65.5, top - 1.1, name, fontsize=9.6, fontweight='bold', color=c)
    for i, k in enumerate(nls):
        badge(120.2 - 2.7 * (len(nls) - 1 - i), top - 2.2, k)
    txt(65.5, top - 3.9, 'from: ' + src, fontsize=7.9, color=MUTED, style='italic')
    txt(65.5, top - 6.3, '\n'.join(lines), fontsize=8.1, linespacing=1.12)
    sub = sum(n for _, n in blocks); total += sub
    for i, (lab, n) in enumerate(blocks):
        txt(127.5, top - 1.2 - i * 2.35, lab, fontsize=8.1); num(165.5, top - 1.2 - i * 2.35, n)
    yb = top - h + 0.6
    ax.plot([127.5, 165.5], [yb + 2.5, yb + 2.5], color=HAIR, lw=0.6)
    txt(127.5, yb + 2.2, 'module total', fontsize=8, color=MUTED); num(165.5, yb + 2.3, sub, fs=9.4, c=c)
    if srckey == 'V1':
        arrow((60.3, 100), (63.8, top - h / 2))
    else:  # from the opponent channels: down the input box, under the V1 box, then right
        mid = top - h / 2
        ax.plot([12, 12, 62.3, 62.3], [104, 78, 78, mid], color=MUTED, lw=1.0, ls=(0, (3, 2)))
        arrow((62.3, mid), (63.8, mid))
    arrow((122.2, top - h / 2), (125.8, top - h / 2))
assert total == 9996, total
assert y > 1, y

# output panel
box(2, 2, 58, 74, fc=PANEL)
txt(4, 74, 'Output and ridge regression', fontsize=11, fontweight='bold')
txt(4, 70.6, 'Each block is normalized separately, then all 36 blocks are\nconcatenated into one vector per image:', fontsize=9, linespacing=1.15)
txt(31, 63.6, '9,996 features', fontsize=22, fontweight='bold', ha='center', color=PRIMARY)
txt(4, 56.6, 'By computational family (each feature traced to its source channels):', fontsize=8.6, color=MUTED)
for i, (f, n) in enumerate([('Edges', 3562), ('Texture statistics', 2280), ('Color', 2148), ('Corners and contours', 935),
                            ('Luminance curvature', 670), ('Mixed', 243), ('Global layout', 158)]):
    txt(6, 53.4 - i * 2.5, f, fontsize=8.6); num(30, 53.4 - i * 2.5, n)
txt(34, 53.4, 'Ridge regression', fontsize=10, fontweight='bold')
txt(34, 50.4, 'one linear map from the 9,996\nfeatures to each voxel, fit on each\nsubject’s unique images', fontsize=8.4, color=MUTED, linespacing=1.15)
txt(34, 41.6, 'Order of operations', fontsize=9, fontweight='bold')
txt(34, 38.8, 'compression → energy or\nrectification → normalization →\nproducts, tuning → pooling →\nlinear ridge regression', fontsize=8.4, color=MUTED, linespacing=1.15)
txt(4, 30, 'Nonlinearity badges', fontsize=9.5, fontweight='bold')
for i, (k, (c, lab)) in enumerate(NL.items()):
    yy = 26.4 - i * 3.0
    badge(6, yy, k); txt(8.4, yy + 1.0, lab, fontsize=8.6)
txt(4, 4.6, 'Grids are foveated: cell size grows with eccentricity. "Spread" is the spatial standard deviation within a cell.', fontsize=7.8, color=MUTED)
fig.savefig(sys.argv[2], dpi=170, facecolor='white')
fig.savefig(sys.argv[2].replace('.png', '.pdf'), facecolor='white')
print('ok', total, 'bottom', round(y, 1))
