"""Two-step animation: (1) the built weights, (2) the weight tiles flying in to form the Mona Lisa.
A virtual camera follows the first tiles up close, then zooms out as the tiles speed up.
usage: uv run experiments/mona-lisa/src/make_assembly_anim.py
needs figures/cnn_small_explained.png (from cnn_explained.py) and ffmpeg."""
import os, json, subprocess, numpy as np
from PIL import Image

root = os.path.join(os.path.dirname(__file__), '..')
Z = np.load(os.path.join(root, 'cnn_small_weights.npz')); meta = json.load(open(os.path.join(root, 'cnn_small_weights.json')))
order = meta['order']; C, R = meta['cols'], meta['rows']; TH, TW = meta['tile']; T = TH*TW; N = C*R
flat = np.concatenate([Z[nm].ravel() for nm in order]); lo, hi = flat.min(), flat.max()
tiles = ((flat.reshape(N, TH, TW) - lo)/(hi - lo)*255).clip(0, 255).astype(np.uint8)        # native 14 x 28
sizes = [Z[nm].size for nm in order]; starts = np.cumsum([0] + sizes[:-1])
layer_of = np.searchsorted(starts, np.arange(N)*T, side='right') - 1
group = np.where(layer_of == 0, 0, np.where(layer_of == 1, 1, np.where(layer_of == 2, 2, 3)))  # layer 1, 2, 3, 4-16
COLS = np.array([(209, 73, 91), (224, 122, 31), (51, 102, 204), (46, 157, 87)], np.uint8)

VW, VH, FPS = 1600, 900, 30
BGV = 255
fig = np.asarray(Image.open(os.path.join(root, 'figures', 'cnn_small_explained.png')).convert('RGB'))
# the built weights, with the figure's title, "1 Build" header and rule replaced by one sentence
from PIL import ImageDraw, ImageFont
CROP_TOP, CAP_H = 160, 230
CAPTION = 'Opus 5.5 hand-wrote these weights\nthat achieve 96.17% MNIST test accuracy'
part1 = np.full((CAP_H + 835 - CROP_TOP, fig.shape[1], 3), 255, np.uint8)
part1[CAP_H:] = fig[CROP_TOP:835]
_im = Image.fromarray(part1); _d = ImageDraw.Draw(_im)
_d.multiline_text((part1.shape[1]//2, CAP_H//2 + 15), CAPTION, fill=(20, 20, 20), anchor='mm', align='center', spacing=14,
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 70))
part1 = np.asarray(_im); P1H, P1W = part1.shape[:2]; P1img = Image.fromarray(part1)
S1, X1 = VW/P1W, 0.0; Y1 = (VH - P1H*S1)/2                        # step 1 layout (world units = screen px at zoom 1)
S2, X2 = 940/P1W, 20.0; Y2 = (VH - P1H*S2)/2                      # step 2 layout: shrunk to the left
TWc, THc = 10, 5; CW, CH = C*TWc, R*THc; CX, CY = 980.0, (VH - CH)/2.0
NAT = TH/THc                                                      # native canvas pixels per world unit
canvas_nat = tiles.reshape(R, C, TH, TW).transpose(0, 2, 1, 3).reshape(R*TH, C*TW)
gy, gx = np.mgrid[0:R*TH, 0:C*TW]; tile_id_nat = (gy//TH)*C + gx//TW
grid_nat = np.where((gy % TH == 0) | (gx % TW == 0), 228, 243).astype(np.uint8)

# sources: where each layer's weights sit in the figure (pixels of the 2400 x 1860 figure)
SRC = {0: (72, 344, 360, 632), 1: (492, 446, 708, 530), 2: (888, 370, 1368, 610), 3: (1764, 368, 1944, 618)}
rng = np.random.default_rng(0)
src = np.zeros((N, 2)); dst = np.zeros((N, 2))
for i in range(N):
    x0, y0, x1, y1 = SRC[group[i]]
    src[i] = (X2 + S2*rng.uniform(x0, x1), Y2 + S2*(rng.uniform(y0, y1) - CROP_TOP + CAP_H))
    r_, c_ = divmod(i, C); dst[i] = (CX + c_*TWc + TWc/2, CY + r_*THc + THc/2)

# timing: step 1, shrink, then tiles launched on a log schedule (slow first, then faster and faster)
T1, TR = 1.3, 0.5; S0 = T1 + TR
L0, TF, KS = 0.25, 4.0, 3.0
A = TF/np.log1p((N - 1)/KS)
launch = S0 + L0 + A*np.log1p(np.arange(N)/KS)    # first gaps ~0.15 s, then ever faster
dur = 0.35 + 0.35*np.exp(-np.arange(N)/6)                        # early flights a little slower
END = (launch + dur).max(); TOTAL = END + 1.2
ease = lambda x: 0.5 - 0.5*np.cos(np.pi*np.clip(x, 0, 1))
def flight(i, t):
    p = ease((t - launch[i])/dur[i])
    ctrl = (src[i] + dst[i])/2 + np.array([0, -140])
    pos = (1 - p)**2*src[i] + 2*(1 - p)*p*ctrl + p**2*dst[i]
    w = 24 + (TWc - 24)*p; h = 12 + (THc - 12)*p
    return pos, w, h

# the camera: (centre x, centre y, zoom) in world units
HOME = np.array([VW/2, VH/2]); Z0 = 3.6
corner = np.array([CX + 95, CY + 45])
def camera(t):
    """wide while the first tiles launch, zoom into the corner as tile 0 lands, then back out"""
    s = t - S0
    land0 = L0 + dur[0]
    z_in0, z_in1 = land0 - 0.45, land0 + 0.05                  # zoom in at the end of tile 0's flight
    hold1 = z_in1 + 0.6                                          # watch the first row fill
    out1 = hold1 + 0.9                                           # zoom out as the tiles speed up
    if s < z_in0: return HOME, 1.0
    if s < z_in1:
        u = ease((s - z_in0)/(z_in1 - z_in0)); return HOME + (corner - HOME)*u, np.exp(np.log(Z0)*u)
    if s < hold1: return corner, Z0
    if s < out1:
        u = ease((s - hold1)/(out1 - hold1)); return corner + (HOME - corner)*u, np.exp(np.log(Z0)*(1 - u))
    return HOME, 1.0

def place(frame, img_pil, wx, wy, ww, wh, cam, resample):
    """draw a world-space image rectangle through the camera, cropping to what is visible"""
    (cx, cy), z = cam
    sx0, sy0 = (wx - cx)*z + VW/2, (wy - cy)*z + VH/2; sw, sh = ww*z, wh*z
    vx0, vy0, vx1, vy1 = max(sx0, 0), max(sy0, 0), min(sx0 + sw, VW), min(sy0 + sh, VH)
    if vx1 - vx0 < 1 or vy1 - vy0 < 1: return
    W_, H_ = img_pil.size; kx, ky = W_/sw, H_/sh
    box = ((vx0 - sx0)*kx, (vy0 - sy0)*ky, (vx1 - sx0)*kx, (vy1 - sy0)*ky)
    out = img_pil.resize((int(round(vx1)) - int(round(vx0)), int(round(vy1)) - int(round(vy0))), resample, box=box)
    a = np.asarray(out); a = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    frame[int(round(vy0)):int(round(vy0)) + a.shape[0], int(round(vx0)):int(round(vx0)) + a.shape[1]] = a

def render(t):
    frame = np.full((VH, VW, 3), BGV, np.uint8)
    if t < T1:                                                    # step 1: the built weights
        a = min(1.0, t/0.3); cam = (HOME, 1.0)
        place(frame, P1img, X1, Y1, P1W*S1, P1H*S1, cam, Image.LANCZOS)
        return (BGV + (frame.astype(float) - BGV)*a).astype(np.uint8)
    u = ease((t - T1)/TR)                                         # shrink to the left, canvas fades in
    s_ = S1 + (S2 - S1)*u
    cam = camera(t)
    place(frame, P1img, X1 + (X2 - X1)*u, Y1 + (Y2 - Y1)*u, P1W*s_, P1H*s_, cam, Image.LANCZOS)
    landed = (launch + dur) <= t
    cv = np.where(landed[tile_id_nat], canvas_nat, grid_nat)
    if u < 1: cv = (BGV + (cv.astype(float) - BGV)*u).astype(np.uint8)
    z = cam[1]
    place(frame, Image.fromarray(cv), CX, CY, CW, CH, cam, Image.NEAREST if z*TWc/TW > 1.2 else Image.BOX)
    (cx, cy) = cam[0]
    for i in np.where((launch <= t) & (t < launch + dur))[0]:   # tiles in flight
        pos, w, h = flight(i, t)
        sw, sh = max(int(round(w*z)), 2), max(int(round(h*z)), 1)
        x0 = int(round((pos[0] - cx)*z + VW/2 - sw/2)); y0 = int(round((pos[1] - cy)*z + VH/2 - sh/2))
        if x0 >= VW or y0 >= VH or x0 + sw <= 0 or y0 + sh <= 0: continue
        ti = tiles[i][np.linspace(0, TH - 1e-6, sh).astype(int)][:, np.linspace(0, TW - 1e-6, sw).astype(int)]
        im = np.repeat(ti[..., None], 3, 2)
        if sw >= 14:
            bw = max(1, sw//40); col = COLS[group[i]]
            im[:bw] = col; im[-bw:] = col; im[:, :bw] = col; im[:, -bw:] = col
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + sw, VW), min(y0 + sh, VH)
        frame[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0]
    return frame

if __name__ == '__main__':
    import sys
    if '--preview' in sys.argv:
        for tt in [float(a) for a in sys.argv[sys.argv.index('--preview') + 1:]]:
            Image.fromarray(render(tt)).save(os.path.join(os.environ.get('CLAUDE_JOB_DIR', root), 'tmp', f'cam_{tt:.2f}.png'))
        sys.exit()
    out = os.path.join(root, 'figures', 'assembly_animation.mp4')
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{VW}x{VH}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                          '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for f in range(int(TOTAL*FPS)):
        p.stdin.write(render(f/FPS).tobytes())
    p.stdin.close(); p.wait()
    gif = out.replace('.mp4', '.gif')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out, '-vf',
                    'fps=20,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse', gif], check=True)
    print('saved', out, 'and', gif, f'({TOTAL:.1f} s)')
