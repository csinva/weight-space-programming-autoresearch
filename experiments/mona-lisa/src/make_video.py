"""Explainer video for the small Mona Lisa convnet.
usage: python src/make_video.py [--preview]      -> figures/monalisa_explainer.mp4"""
import os, sys, json, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(__file__))
from mnist import load

root = os.path.join(os.path.dirname(__file__), '..')
Z = np.load(os.path.join(root, 'cnn_small_weights.npz'))
meta = json.load(open(os.path.join(root, 'cnn_small_weights.json')))
order = meta['order']; C, R = meta['cols'], meta['rows']; TH, TW = meta['tile']; T = TH*TW
PER = meta['per_class']; S = meta['S']
flat = np.concatenate([Z[nm].ravel() for nm in order]); lo, hi = flat.min(), flat.max()
PIC = flat.reshape(R, C, TH, TW).transpose(0, 2, 1, 3).reshape(R*TH, C*TW)
PH, PW = PIC.shape
sizes = [Z[nm].size for nm in order]; starts = np.cumsum([0] + sizes[:-1])
g8 = lambda a: (np.clip((a - lo)/(hi - lo), 0, 1)*255).astype(np.uint8)
PIC8 = g8(PIC)

VW, VH, FPS = 1280, 720, 30
BG = (14, 14, 18); FG = (235, 235, 240); DIM = (150, 150, 160)
ACC = {1: (230, 80, 90), 2: (245, 150, 60), 3: (90, 130, 240), 'tree': (80, 200, 110), 'gold': (240, 200, 90)}
F = '/System/Library/Fonts/Supplemental/'
def font(sz, bold=False, mono=False):
    if mono: return ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', sz)
    return ImageFont.truetype(F + ('Arial Bold.ttf' if bold else 'Arial.ttf'), sz)
ease = lambda x: 0.5 - 0.5*np.cos(np.pi*np.clip(x, 0, 1))
fade = lambda t, a, d=0.5: float(np.clip((t - a)/d, 0, 1))

def canvas(): return Image.new('RGB', (VW, VH), BG)
def text(img, xy, s, sz=26, col=FG, alpha=1.0, bold=False, mono=False, anchor='la'):
    if alpha <= 0: return
    c = tuple(int(BG[i] + (col[i] - BG[i])*alpha) for i in range(3))
    ImageDraw.Draw(img).text(xy, s, font=font(sz, bold, mono), fill=c, anchor=anchor)
def paste(img, arr_or_img, xy, scale=1, alpha=1.0, resample=Image.NEAREST, border=None):
    if alpha <= 0: return
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    im = im.convert('RGB')
    if scale != 1: im = im.resize((int(im.width*scale), int(im.height*scale)), resample)
    if alpha < 1:
        base = img.crop((xy[0], xy[1], xy[0] + im.width, xy[1] + im.height))
        im = Image.blend(base, im, alpha)
    img.paste(im, xy)
    if border: ImageDraw.Draw(img).rectangle([xy[0]-2, xy[1]-2, xy[0]+im.width+1, xy[1]+im.height+1], outline=border, width=2)
    return im.size
def arrow(img, x0, x1, y, alpha=1.0, col=DIM):
    if alpha <= 0: return
    c = tuple(int(BG[i] + (col[i] - BG[i])*alpha) for i in range(3)); d = ImageDraw.Draw(img)
    d.line([x0, y, x1, y], fill=c, width=3); d.polygon([(x1, y), (x1-10, y-7), (x1-10, y+7)], fill=c)
def magma(a01):
    import matplotlib.cm as cm
    return (cm.magma(np.clip(a01, 0, 1))[..., :3]*255).astype(np.uint8)

# ---------------------------------------------------------------- data for the story
_, _, Xte, yte = load()
q = 0                                                         # test image 0 (a 7)
x = Xte[q].reshape(28, 28)
K1 = Z['conv1'][0, 0]
xp = np.pad(x, 3); y15 = np.einsum('hwij,ij->hw', np.lib.stride_tricks.sliding_window_view(xp, (6, 6)), K1)[::2, ::2]
pairs = np.empty((S, 2*S)); pairs[:, 0::2] = 1 + y15; pairs[:, 1::2] = 1 - y15
mem = Z['conv3_memory'][:, 0]; bm = Z['conv3_memory.bias']
win = [pairs[a:a+TH, 2*b:2*b+TW].ravel() for a in range(2) for b in range(2)]
scores = np.max(np.stack([mem.reshape(len(mem), -1)@w for w in win]), 0) + bm
cls_best = scores.reshape(10, PER).max(1); pred = int(cls_best.argmax())
best_f = int(pred*PER + scores.reshape(10, PER)[pred].argmax())
def digit_of(i):
    f = mem[i]; d = 0.5*(f[:, 0::2] - f[:, 1::2]); return (d - d.min())/(d.max() - d.min() + 1e-12)

# ---------------------------------------------------------------- scene 1: how it works (no training)
def scene1(t):
    img = canvas()
    text(img, (VW//2, 48), 'How a network can read digits with zero training', 38, bold=True, anchor='mm', alpha=fade(t, 0))
    text(img, (VW//2, 92), 'every weight is written down, nothing is learned by gradient descent', 22, DIM, anchor='mm', alpha=fade(t, 0.4))
    cy = 250; xs = [60, 290, 535, 800, 1060]
    steps = [
        (0.9, 'input digit', '28 x 28 pixels', lambda: paste(img, (np.clip(x, 0, 1)*255).astype(np.uint8), (xs[0], cy - 84), 6, fade(t, 0.9), border=DIM)),
        (2.0, 'layer 1: blur + pool', 'a hand-written kernel', lambda: paste(img, (np.clip(y15/ y15.max(), 0, 1)*255).astype(np.uint8), (xs[1], cy - 84), 11, fade(t, 2.0), border=ACC[1])),
        (3.1, 'layer 2: pairs (1+y, 1-y)', 'each pair always sums to 2', lambda: paste(img, (np.clip(pairs/2, 0, 1)*255).astype(np.uint8), (xs[2], cy - 84), 7, fade(t, 3.1), border=ACC[2])),
        (4.2, 'layer 3: 10,240 filters', 'painting + tiny stored digit', lambda: paste(img, g8(mem[best_f]), (xs[3], cy - 84), 7, fade(t, 4.2), border=ACC[3])),
        (5.3, 'layers 4-16: max-tree', 'best match per class', None),
    ]
    for k, (ta, l1, l2, draw) in enumerate(steps):
        a = fade(t, ta)
        if draw: draw()
        text(img, (xs[k], cy + 100), l1, 21, FG, a, bold=True); text(img, (xs[k], cy + 128), l2, 18, DIM, a)
        if k: arrow(img, xs[k] - 42, xs[k] - 10, cy, a)
    a = fade(t, 5.3)
    if a > 0:   # bar chart of the 10 class scores
        cb = cls_best - cls_best.min(); cb = cb/cb.max()
        d = ImageDraw.Draw(img)
        for c in range(10):
            h = int(150*(0.08 + 0.92*cb[c])); x0 = xs[4] + c*18
            col = ACC['gold'] if c == pred else (90, 90, 100)
            col = tuple(int(BG[i] + (col[i] - BG[i])*a) for i in range(3))
            d.rectangle([x0, cy + 75 - h, x0 + 13, cy + 75], fill=col)
            text(img, (x0 + 7, cy + 80), str(c), 13, DIM, a, anchor='ma')
        text(img, (xs[4] + 90, cy - 108), f'answer: {pred}', 26, ACC['gold'], fade(t, 5.8), bold=True, anchor='ma')
    a = fade(t, 7.0, 0.7)
    text(img, (VW//2, 470), 'why the painting does not get in the way', 24, ACC['gold'], a, bold=True, anchor='mm')
    text(img, (VW//2, 520), 'weights on a pair:  P + e*c   and   P - e*c        inputs on a pair:  1 + y   and   1 - y', 21, FG, a, mono=True, anchor='mm')
    text(img, (VW//2, 560), 'filter output  =  2 * sum(P)  +  2e * (c . y)', 23, FG, fade(t, 8.0, 0.7), mono=True, anchor='mm')
    text(img, (VW//2, 600), 'the painting P adds the same constant for every input (removed by the bias);', 20, DIM, fade(t, 8.8, 0.7), anchor='mm')
    text(img, (VW//2, 628), 'only the stored digit c is compared with the input y', 20, DIM, fade(t, 8.8, 0.7), anchor='mm')
    return img

# ---------------------------------------------------------------- scene 2: what the weights store
def scene2(t):
    img = canvas()
    text(img, (VW//2, 48), 'What a single weight tile stores', 38, bold=True, anchor='mm', alpha=fade(t, 0))
    f = mem[best_f]; s = 15
    tile8 = g8(f)
    paste(img, tile8, (70, 140), s, fade(t, 0.3), border=ACC[3])
    text(img, (70, 140 + TH*s + 14), f'layer-3 filter {best_f}: {T} weights', 20, FG, fade(t, 0.5))
    text(img, (70, 140 + TH*s + 40), 'looks like a piece of the painting', 18, DIM, fade(t, 0.8))
    a = fade(t, 2.2, 0.8); sl = ease((t - 2.2)/1.2)
    L8, R8 = g8(f[:, 0::2]), g8(f[:, 1::2])
    xL = int(70 + TW*s + 40 + 10*sl); xR = int(xL + 14*s + 30 + 20*sl)
    paste(img, L8, (xL, 140), s, a); paste(img, R8, (xR, 140), s, a)
    text(img, (xL, 140 + TH*s + 14), 'left weight of each pair', 18, DIM, a)
    text(img, (xR, 140 + TH*s + 14), 'right weight of each pair', 18, DIM, a)
    text(img, (xR - 17, 140 + TH*s//2), '-', 34, FG, fade(t, 4.0), bold=True, anchor='mm')
    a = fade(t, 4.4, 1.0)
    xD = xR + 14*s + 50
    text(img, (xD - 25, 140 + TH*s//2), '=', 34, FG, fade(t, 4.2), bold=True, anchor='mm')
    paste(img, magma(digit_of(best_f)), (xD, 140), s, a, border=ACC['gold'] if a > 0.5 else None)
    text(img, (xD, 140 + TH*s + 14), 'a stored training digit', 18, ACC['gold'], a, bold=True)
    text(img, (VW//2, 420), 'the two weights of a pair differ by only 3% of the brightness range (magnified here)', 20, DIM, fade(t, 5.4), anchor='mm')
    # one example per class
    a0 = 6.4
    for c in range(10):
        i = c*PER + 37 + c; ac = fade(t, a0 + 0.18*c)
        x0 = 90 + c*112
        paste(img, g8(mem[i]), (x0, 470), 3.5, ac)
        paste(img, magma(digit_of(i)), (x0 + 24, 530), 3.5, ac)
    text(img, (VW//2, 650), '10,240 filters = 1,024 training digits per class, copied in, not learned', 22, FG, fade(t, 8.4), anchor='mm')
    return img

# ---------------------------------------------------------------- scene 3: 3D zoom out
def tex_of(L):
    w = Z[order[L]]
    if L == 0: a = w[:, 0].reshape(14, 14, 6, 6).transpose(0, 2, 1, 3).reshape(84, 84)
    elif L == 1: a = w.reshape(TH, -1)
    elif L == 2: return None
    else:
        v = w.ravel(); side = int(np.ceil(np.sqrt(v.size))); v = np.concatenate([v, np.full(side*side - v.size, v.mean())])
        a = v.reshape(side, side)
    im = Image.fromarray(g8(a)).convert('RGB')
    if im.width < 64: im = im.resize((64, int(64*im.height/im.width)), Image.NEAREST)
    return im
TEX = [tex_of(L) for L in range(16)]
PYR = [Image.fromarray(PIC8).convert('RGB')]
while PYR[-1].width > 120: PYR.append(PYR[-1].resize((PYR[-1].width//2, PYR[-1].height//2), Image.LANCZOS))
plates = []; z = 0.0
for L in range(16):
    if L == 2: h = 3.0; w = h*PW/PH
    else:
        n = sizes[L]; side = 0.25 + 0.17*np.log10(n)
        h = side; w = side*(TEX[L].width/TEX[L].height)
    plates.append(dict(L=L, z=z, w=w, h=h)); z += 0.55 if L != 2 else 0.8
    if L == 1: z += 0.25
ZC = plates[2]['z']
def persp_coeffs(dst, src):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]]; B += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(B, float))
def camera(t):
    s = ease(t)
    face = np.array([plates[2]['w']*(0.405 - 0.5), 3.0*(0.5 - 0.228), ZC])   # her eye
    mid = np.array([0.0, 0.0, (plates[0]['z'] + plates[-1]['z'])/2])
    target = face + (mid - face)*ease((t - 0.45)/0.55)
    d = np.exp(np.log(0.09) + (np.log(15.0) - np.log(0.09))*ease(t*1.05))
    yaw = np.radians(62*ease((t - 0.42)/0.58)); pitch = np.radians(18*ease((t - 0.42)/0.58))
    fwd = np.array([np.sin(yaw)*np.cos(pitch), -np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    pos = target - fwd*d
    r = np.cross(fwd, [0, 1, 0]); r /= np.linalg.norm(r); up = np.cross(r, fwd)
    return pos, fwd, r, up
def proj(p, cam, f=900):
    pos, fwd, r, up = cam; v = p - pos; zc = v@fwd
    return (VW/2 + f*(v@r)/zc, VH/2 - f*(v@up)/zc, zc)
def scene3(t, dur):
    img = canvas(); u = t/dur; cam = camera(u)
    items = []
    for P in plates:
        cx, cz, w, h = 0.0, P['z'], P['w'], P['h']
        corners = [np.array([cx - w/2, h/2, cz]), np.array([cx + w/2, h/2, cz]),
                   np.array([cx + w/2, -h/2, cz]), np.array([cx - w/2, -h/2, cz])]
        pr = [proj(c, cam) for c in corners]
        if min(p[2] for p in pr) < 0.02: continue
        items.append((np.mean([p[2] for p in pr]), P, pr))
    for _, P, pr in sorted(items, key=lambda a: -a[0]):
        quad = [(p[0], p[1]) for p in pr]
        xs_, ys_ = [q[0] for q in quad], [q[1] for q in quad]
        if max(xs_) < -50 or min(xs_) > VW + 50 or max(ys_) < -50 or min(ys_) > VH + 50:
            if not (min(xs_) < 0 and max(xs_) > VW): continue
        if P['L'] == 2:
            span = max(np.hypot(quad[1][0]-quad[0][0], quad[1][1]-quad[0][1]), 1)
            lev = int(np.clip(np.floor(np.log2(PW/span)), 0, len(PYR)-1)); tex = PYR[lev]
            rs = Image.NEAREST if span > 2*PW else Image.BILINEAR
        else: tex = TEX[P['L']]; rs = Image.BILINEAR
        src = [(0, 0), (tex.width, 0), (tex.width, tex.height), (0, tex.height)]
        try: cf = persp_coeffs(quad, src)
        except np.linalg.LinAlgError: continue
        warped = tex.transform((VW, VH), Image.PERSPECTIVE, tuple(cf), rs, fillcolor=BG)
        mask = Image.new('L', (VW, VH), 0); ImageDraw.Draw(mask).polygon(quad, fill=255)
        img.paste(warped, (0, 0), mask)
        col = ACC[3] if P['L'] == 2 else (ACC[1] if P['L'] == 0 else ACC[2] if P['L'] == 1 else ACC['tree'])
        ImageDraw.Draw(img).polygon(quad, outline=col, width=2 if u > 0.3 else 1)
        a = fade(u, 0.8, 0.12)
        if a > 0:
            top = min(quad, key=lambda q: q[1])
            lab = {0: '1', 1: '2', 2: '3: memory', 15: '16'}.get(P['L'], str(P['L'] + 1))
            text(img, (top[0], top[1] - 8), lab, 15 if P['L'] != 2 else 20, col, a, bold=True, anchor='md')
    text(img, (VW//2, 44), 'the whole network', 34, FG, fade(u, 0.55, 0.15), bold=True, anchor='mm')
    text(img, (VW//2, 84), '16 layers  |  4,045,440 weights  |  each plate shows one layer\'s weights', 20, DIM, fade(u, 0.65, 0.15), anchor='mm')
    a = fade(u, 0.85, 0.12)
    text(img, (VW - 40, VH - 70), 'layer 1: blur + pool   layer 2: pairs', 18, ACC[1], a, anchor='ra')
    text(img, (VW - 40, VH - 44), 'layer 3: 10,240 memory filters   layers 4-16: max-tree', 18, ACC['tree'], a, anchor='ra')
    if u < 0.12: text(img, (40, VH - 50), 'zooming out from one weight tile...', 20, FG, 1 - fade(u, 0.08, 0.04))
    return img

# ---------------------------------------------------------------- scene 4: assembly
DH = 640; DS = DH/PH; DW = int(round(PW*DS)); X0, Y0 = 90, 50
DISP = np.asarray(Image.fromarray(PIC8).resize((DW, DH), Image.LANCZOS), np.float32)
yy, xx = np.mgrid[0:DH, 0:DW]
tile_id = ((yy/DS).astype(int)//TH)*C + (xx/DS).astype(int)//TW
NT = C*R
def tile_time(i, dur): return dur*(i/NT)**0.62
def scene4(t, dur):
    img = canvas(); fill = dur - 3.5
    tt = tile_time(tile_id, fill)
    shown = tt <= t
    glow = np.where(shown, np.exp(-(t - tt)/0.25), 0)
    frame = np.where(shown, DISP, 26.0)
    rgb = np.stack([frame]*3, -1)
    gl = glow[..., None]*np.array([255, 210, 120])[None, None]
    rgb = np.clip(rgb*(1 - 0.6*glow[..., None]) + 0.6*gl, 0, 255).astype(np.uint8)
    img.paste(Image.fromarray(rgb), (X0, Y0))
    ImageDraw.Draw(img).rectangle([X0 - 2, Y0 - 2, X0 + DW + 1, Y0 + DH + 1], outline=(60, 60, 70), width=2)
    # progress
    placed = int(np.searchsorted(tile_time(np.arange(NT), fill), t, side='right'))
    wts = min(placed*T, flat.size)
    lay = int(np.searchsorted(starts, max(wts - 1, 0), side='right')) - 1 if placed else -1
    RX = X0 + DW + 70
    text(img, (RX, 70), 'assembling the weights, in layer order', 30, FG, fade(t, 0), bold=True)
    text(img, (RX, 112), f'{C} x {R} tiles of {TH} x {TW} weights, filled left to right, top to bottom', 19, DIM, fade(t, 0.2))
    rows = [(0, 'layer 1   blur + pool kernels', ACC[1]), (1, 'layer 2   pair kernels', ACC[2]),
            (2, 'layer 3   10,240 memory filters', ACC[3]), (3, 'layers 4-16   max-tree', ACC['tree'])]
    for k, (L, s, col) in enumerate(rows):
        a = fade(t, 0.4 + 0.1*k)
        Lend = 15 if L == 3 else L
        done = wts >= starts[Lend] + sizes[Lend]; active = (lay >= L and not done) if L < 3 else (lay >= 3 and not done)
        c = col if (active or done) else (80, 80, 90)
        text(img, (RX, 190 + 44*k), ('> ' if active else '   ') + s, 23, c, a, bold=active)
        if done: text(img, (RX + 470, 190 + 44*k), 'done', 19, col, a, anchor='la')
    text(img, (RX, 400), f'{wts:,}', 54, ACC['gold'], fade(t, 0.3), bold=True)
    text(img, (RX, 462), 'weights placed', 20, DIM, fade(t, 0.3))
    a = fade(t, fill + 0.4, 0.8)
    text(img, (RX, 540), 'every pixel is a weight', 28, FG, a, bold=True)
    text(img, (RX, 580), '96.17% on MNIST test digits', 24, ACC['gold'], fade(t, fill + 0.9, 0.8), bold=True)
    text(img, (RX, 614), 'zero training', 24, ACC['gold'], fade(t, fill + 1.3, 0.8), bold=True)
    return img

# ---------------------------------------------------------------- render
SCENES = [(scene1, 11.0), (scene2, 10.5), (lambda t: scene3(t, 9.0), 9.0), (lambda t: scene4(t, 12.0), 12.0)]
def frames():
    for k, (fn, dur) in enumerate(SCENES):
        n = int(dur*FPS)
        for i in range(n):
            t = i/FPS; im = fn(t)
            a = min(1.0, t/0.35, (dur - t)/0.35) if k < len(SCENES) - 1 else min(1.0, t/0.35)
            if a < 1: im = Image.blend(canvas(), im, max(a, 0))
            yield im

if __name__ == '__main__':
    out = os.path.join(root, 'figures', 'monalisa_explainer.mp4')
    if '--preview' in sys.argv:
        for k, (fn, dur) in enumerate(SCENES):
            for frac in [0.97] if k != 2 else [0.02, 0.3, 0.5, 0.7]:
                fn(frac*dur).save(os.path.join(os.environ.get('CLAUDE_JOB_DIR', root), 'tmp', f'prev_{k}_{int(frac*100)}.png'))
        print('previews saved'); sys.exit()
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{VW}x{VH}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                          '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for i, im in enumerate(frames()):
        p.stdin.write(im.tobytes())
        if i % 150 == 0: print('frame', i, flush=True)
    p.stdin.close(); p.wait(); print('saved', out)
