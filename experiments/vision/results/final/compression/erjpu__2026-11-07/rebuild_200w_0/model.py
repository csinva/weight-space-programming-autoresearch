"""
Fixed, training-free image feature model for fMRI encoding (early + ventral visual cortex).

Pipeline (all parameters analytic; nothing learned, no pretrained weights):
  * compressive gamma ((x+0.04)^0.45, rescaled) -> orthonormal opponent space (L, RG, BY)
  * foveally magnified "cortical" copy: each axis warped by u -> 0.58u + 0.42u^3
  * 3 octaves: 7x7 quadrature Gabors (8 orient x 2 freq) + 2 DoGs on luminance,
    coarse quadrature Gabors (4 orient) on chromatic channels; divisive normalisation
    by local 3x3 + 0.5 * global mean energy, log1p compression
  * cortical-grid means (9/6/4), retinal-grid std + peaks (4/3/2)
  * derived maps (orientation harmonics, curvature, pop-out, phase congruency,
    bar/edge polarity, 2nd-order modulation, channel correlations, cross-octave
    persistence/differences, fovea vs periphery, mirror correlations)
  * five-octave surface geometry (Hessian shape-index x curvedness, structure tensor,
    isophote curvature, chromatic corners) on shrinking grids, weights 1.5 -> 0.35,
    plus cross-octave correlations
  * extras: coarse colour grids, anchor-distance ring maps, local-extremum densities
Every block is separately L2-normalised and weighted.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

_CHUNK = 64
_EPS = 1e-8
_CACHE = {}


# ----------------------------------------------------------------------------- filters
def _gabor_pair(size, lam, sig, theta):
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    xr = x * math.cos(theta) + y * math.sin(theta)
    env = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    ev = env * np.cos(2 * math.pi * xr / lam)
    ev = ev - env * (ev.sum() / env.sum())          # exact zero DC
    od = env * np.sin(2 * math.pi * xr / lam)
    ev /= np.sqrt((ev ** 2).sum())
    od /= np.sqrt((od ** 2).sum())
    return ev, od


def _dog(size, sc, ss):
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    rr = x ** 2 + y ** 2
    gc = np.exp(-rr / (2 * sc ** 2)); gc /= gc.sum()
    gs = np.exp(-rr / (2 * ss ** 2)); gs /= gs.sum()
    d = gc - gs
    d -= d.mean()
    return d / np.sqrt((d ** 2).sum())


def _params(device):
    key = str(device)
    if key in _CACHE:
        return _CACHE[key]
    lum = []
    for lam, sig in ((3.0, 1.35), (5.5, 2.0)):
        for k in range(8):
            ev, od = _gabor_pair(7, lam, sig, math.pi * k / 8)
            lum += [ev, od]
    lum.append(_dog(7, 0.8, 2.0))
    lum.append(_dog(7, 1.4, 3.0))
    chrm = []
    for k in range(4):
        ev, od = _gabor_pair(7, 6.5, 2.3, math.pi * k / 4)
        chrm += [ev, od]
    lum = torch.tensor(np.stack(lum)[:, None], dtype=torch.float32, device=device)
    chrm = torch.tensor(np.stack(chrm)[:, None], dtype=torch.float32, device=device)
    b = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32, device=device) / 16.0
    th8 = torch.arange(8, dtype=torch.float32, device=device) * math.pi / 8
    P = {
        "lum": lum,                                   # (34,1,7,7)
        "chr": chrm,                                  # (8,1,7,7)
        "chr2": torch.cat([chrm, chrm], 0),           # (16,1,7,7) for groups=2
        "bx": b.view(1, 1, 1, 5), "by": b.view(1, 1, 5, 1),
        "c2": torch.cos(2 * th8).view(1, 8, 1, 1), "s2": torch.sin(2 * th8).view(1, 8, 1, 1),
        "c4": torch.cos(4 * th8).view(1, 8, 1, 1), "s4": torch.sin(4 * th8).view(1, 8, 1, 1),
    }
    _CACHE[key] = P
    return P


# ----------------------------------------------------------------------------- helpers
def _blur(x, P):
    B, C, H, W = x.shape
    y = x.reshape(B * C, 1, H, W)
    y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="reflect"), P["bx"])
    y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="reflect"), P["by"])
    return y.view(B, C, H, W)


def _down(x, P):
    return _blur(x, P)[:, :, ::2, ::2]


def _grad(x):
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) * 0.5
    gy = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) * 0.5
    return gx, gy


def _d2(x):
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    c = p[..., 1:-1, 1:-1]
    dxx = p[..., 1:-1, 2:] - 2 * c + p[..., 1:-1, :-2]
    dyy = p[..., 2:, 1:-1] - 2 * c + p[..., :-2, 1:-1]
    return dxx, dyy


def _smean(x):
    return x.mean(dim=(2, 3), keepdim=True)


def _pool(x, g):
    return F.adaptive_avg_pool2d(x, g).flatten(1)


def _pool_std(x, g):
    m = F.adaptive_avg_pool2d(x, g)
    m2 = F.adaptive_avg_pool2d(x * x, g)
    return torch.sqrt((m2 - m * m).clamp_min(0) + 1e-12).flatten(1)


def _pool_max(x, g):
    return F.adaptive_max_pool2d(x, g).flatten(1)


def _corr_spatial(a, b, e=1e-6):
    a = a - _smean(a)
    b = b - _smean(b)
    cov = (a * b).mean(dim=(2, 3))
    va = (a * a).mean(dim=(2, 3))
    vb = (b * b).mean(dim=(2, 3))
    return cov / torch.sqrt((va + e) * (vb + e))


def _corr_offdiag(S, e=1e-6):
    B, C = S.shape[:2]
    Z = S.flatten(2)
    Z = Z - Z.mean(-1, keepdim=True)
    cov = Z @ Z.transpose(1, 2) / Z.shape[-1]
    d = torch.diagonal(cov, dim1=1, dim2=2) + e
    cor = cov / torch.sqrt(d[:, :, None] * d[:, None, :])
    iu = torch.triu_indices(C, C, offset=1, device=S.device)
    return cor[:, iu[0], iu[1]]


def _divnorm(m):
    S = m.mean(1, keepdim=True)
    loc = F.avg_pool2d(S, 3, stride=1, padding=1, count_include_pad=False)
    den = loc + 0.5 * _smean(S) + 1e-4
    return torch.log1p(m / den)


def _warp_axis(x, dim, n):
    u = torch.linspace(-1.0, 1.0, n, device=x.device)
    s = 0.58 * u + 0.42 * u ** 3
    p = (s + 1) * 0.5 * (n - 1)
    i0 = p.floor().clamp(0, n - 2).long()
    w = (p - i0.float()).clamp(0, 1)
    a = x.index_select(dim, i0)
    b = x.index_select(dim, i0 + 1)
    shape = [1] * x.dim(); shape[dim] = n
    w = w.view(shape)
    return a * (1 - w) + b * w


def _cortical(x):
    H, W = x.shape[-2:]
    return _warp_axis(_warp_axis(x, 2, H), 3, W)


def _coords(H, W, device):
    yy = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)
    return yy, xx


# ----------------------------------------------------------------------------- octave stage
def _octave(opp, P):
    B, _, H, W = opp.shape
    L = opp[:, 0:1]
    C = opp[:, 1:3]
    r = F.conv2d(F.pad(L, (3, 3, 3, 3), mode="reflect"), P["lum"])
    g = r[:, :32].reshape(B, 2, 8, 2, H, W)
    ev, od = g[:, :, :, 0], g[:, :, :, 1]                 # (B,2,8,H,W)
    E = torch.sqrt(ev * ev + od * od + 1e-12)
    dog = r[:, 32:34]
    dogr = torch.cat([F.relu(dog), F.relu(-dog)], 1)       # on0,on1,off0,off1
    cr = F.conv2d(F.pad(C, (3, 3, 3, 3), mode="reflect"), P["chr2"], groups=2)
    cr = cr.reshape(B, 2, 4, 2, H, W)
    CE = torch.sqrt(cr[:, :, :, 0] ** 2 + cr[:, :, :, 1] ** 2 + 1e-12)  # (B,2ch,4,H,W)
    lumN = _divnorm(torch.cat([E.reshape(B, 16, H, W), dogr], 1))      # (B,20)
    chrN = _divnorm(CE.reshape(B, 8, H, W))                             # (B,8)
    return {"ev": ev, "od": od, "E": E, "lumN": lumN, "chrN": chrN}


def _summary(o):
    lumN, chrN = o["lumN"], o["chrN"]
    B, _, H, W = lumN.shape
    Eo = lumN[:, :16].reshape(B, 2, 8, H, W).mean(1)
    S = torch.cat([
        Eo[:, 0::2],
        lumN[:, 16:18].mean(1, keepdim=True),
        lumN[:, 18:20].mean(1, keepdim=True),
        chrN[:, :4].mean(1, keepdim=True),
        chrN[:, 4:].mean(1, keepdim=True)], 1)               # (B,8,H,W)
    return Eo, S


def _derived(o, Eo, S, P):
    out = []
    lumN, chrN = o["lumN"], o["chrN"]
    B, _, H, W = lumN.shape
    # orientation harmonics
    T = Eo.sum(1, keepdim=True) + 1e-6
    harm = torch.cat([(Eo * P[k]).sum(1, keepdim=True) / T for k in ("c2", "s2", "c4", "s4")], 1)
    out.append(("harm", _pool(harm, 4), 0.6))
    # orientation-field curvature
    vx = _blur((Eo * P["c2"]).sum(1, keepdim=True), P)
    vy = _blur((Eo * P["s2"]).sum(1, keepdim=True), P)
    mag = torch.sqrt(vx * vx + vy * vy + 1e-12)
    ux, uy = vx / (mag + 1e-3), vy / (mag + 1e-3)
    gxx, gxy = _grad(ux)
    gyx, gyy = _grad(uy)
    curv = torch.sqrt(gxx ** 2 + gxy ** 2 + gyx ** 2 + gyy ** 2 + 1e-12)
    coh = mag / T
    out.append(("curv", _pool(torch.cat([curv * coh, coh], 1), 4), 0.5))
    # surround pop-out (orientation + colour)
    sur = F.avg_pool2d(Eo, 9, stride=1, padding=4, count_include_pad=False)
    cE = chrN.mean(1, keepdim=True)
    csur = F.avg_pool2d(cE, 9, stride=1, padding=4, count_include_pad=False)
    pop = torch.cat([F.relu(Eo - sur).mean(1, keepdim=True), F.relu(cE - csur)], 1)
    out.append(("pop", _pool(pop, 4), 0.5))
    # phase congruency across the two frequencies
    ev, od, E = o["ev"], o["od"], o["E"]
    num = torch.sqrt((ev[:, 0] + ev[:, 1]) ** 2 + (od[:, 0] + od[:, 1]) ** 2 + 1e-12).sum(1, keepdim=True)
    den = (E[:, 0] + E[:, 1]).sum(1, keepdim=True)
    pc = num / (den + 0.05 * _smean(den) + 1e-4)
    out.append(("pc", _pool(pc, 4), 0.5))
    # bar-vs-edge and bar polarity
    se2 = (ev * ev).sum((1, 2)).unsqueeze(1)
    so2 = (od * od).sum((1, 2)).unsqueeze(1)
    tot = se2 + so2
    tot = tot + 0.05 * _smean(tot) + 1e-8
    be = (se2 - so2) / tot
    pol = (ev * ev.abs()).sum((1, 2)).unsqueeze(1) / tot
    out.append(("baredge", _pool(torch.cat([be, pol], 1), 4), 0.5))
    # second-order (contrast-envelope) modulation
    env = _down(lumN.mean(1, keepdim=True), P)
    sr = F.conv2d(F.pad(env, (3, 3, 3, 3), mode="reflect"), P["chr"])
    se = torch.sqrt(sr[:, 0::2] ** 2 + sr[:, 1::2] ** 2 + 1e-12)
    out.append(("second", _pool(torch.log1p(4 * se), 3), 0.5))
    # off-diagonal channel correlations (global + quadrants)
    h2, w2 = H // 2, W // 2
    cors = [_corr_offdiag(S)]
    for ys in (slice(0, h2), slice(h2, H)):
        for xs in (slice(0, w2), slice(w2, W)):
            cors.append(_corr_offdiag(S[:, :, ys, xs]))
    out.append(("corr", torch.cat(cors, 1), 0.5))
    # fovea vs periphery
    yy, xx = _coords(H, W, lumN.device)
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    mc = (rr < 0.35).float(); mp = (rr > 0.65).float()
    allc = torch.cat([lumN, chrN], 1)
    fc = (allc * mc).sum((2, 3)) / mc.sum()
    fp = (allc * mp).sum((2, 3)) / mp.sum()
    out.append(("fovper", fc - fp, 0.4))
    # mirror correlations
    mir = torch.cat([_corr_spatial(S, S.flip(-1)), _corr_spatial(S, S.flip(-2)),
                     _corr_spatial(S, S.flip(-1).flip(-2))], 1)
    out.append(("mirror", mir, 0.4))
    return out


# ----------------------------------------------------------------------------- surface geometry
def _surface(opp, P):
    Ls = _blur(opp[:, 0:1], P)
    Lx, Ly = _grad(Ls)
    Lxx, Lyy = _d2(Ls)
    _, Lxy = _grad(Lx)
    tr = Lxx + Lyy
    d = torch.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy ** 2 + 1e-14)
    si = (2 / math.pi) * torch.atan2(tr, d)
    cv = torch.sqrt((tr * tr + d * d) * 0.25)
    cvn = torch.log1p(cv / (_smean(cv) + 1e-5))
    centers = torch.linspace(-1, 1, 6, device=opp.device).view(1, 6, 1, 1)
    bins = F.relu(1 - (si - centers).abs() / 0.4) * cvn         # (B,6)
    # structure tensor
    Jxx = _blur(Lx * Lx, P); Jyy = _blur(Ly * Ly, P); Jxy = _blur(Lx * Ly, P)
    jt = Jxx + Jyy
    jd = torch.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2 + 1e-16)
    l1 = (jt + jd) * 0.5
    l2 = ((jt - jd) * 0.5).clamp_min(0)
    m1 = _smean(l1) + 1e-8
    l1n = torch.log1p(l1 / m1)
    l2n = torch.log1p(l2 / m1)
    coh = (jd / (jt + 0.01 * _smean(jt) + 1e-10)) ** 2
    # isophote curvature
    g2 = Lx * Lx + Ly * Ly
    kn = Lx * Lx * Lyy - 2 * Lx * Ly * Lxy + Ly * Ly * Lxx
    kap = torch.tanh(kn / (g2 ** 1.5 + 1e-9))
    gm = torch.sqrt(g2 + 1e-14)
    gw = torch.log1p(gm / (_smean(gm) + 1e-6))
    iso_s = kap * gw
    iso_a = kap.abs() * gw
    # chromatic (RG/BY) Harris corner strength
    Cx, Cy = _grad(_blur(opp[:, 1:3], P))
    Kxx = _blur((Cx * Cx).sum(1, keepdim=True), P)
    Kyy = _blur((Cy * Cy).sum(1, keepdim=True), P)
    Kxy = _blur((Cx * Cy).sum(1, keepdim=True), P)
    kt = Kxx + Kyy
    R = Kxx * Kyy - Kxy * Kxy - 0.04 * kt * kt
    corner = torch.log1p(F.relu(R) / (_smean(kt) ** 2 + 1e-8))
    return torch.cat([bins, cvn, l1n, l2n, coh, iso_s, iso_a, corner], 1)   # (B,13)


# ----------------------------------------------------------------------------- full features
def _bl2(v, w):
    return v / (v.norm(dim=1, keepdim=True) + _EPS) * w


def _features(X, P):
    B = X.shape[0]
    S0 = int(round(math.sqrt(X.shape[1] / 3)))
    img = X.reshape(B, S0, S0, 3).permute(0, 3, 1, 2).clamp(0, 1)
    g0 = 0.04 ** 0.45
    img = ((img + 0.04) ** 0.45 - g0) / (1.04 ** 0.45 - g0)
    R, G, Bc = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    opp = torch.cat([(R + G + Bc) / math.sqrt(3), (R - G) / math.sqrt(2),
                     (R + G - 2 * Bc) / math.sqrt(6)], 1)
    cort = _cortical(opp)

    blocks = []
    ret_pyr = [opp]; cor_pyr = [cort]
    for _ in range(2):
        ret_pyr.append(_down(ret_pyr[-1], P))
        cor_pyr.append(_down(cor_pyr[-1], P))

    cgrid = (9, 6, 4); rgrid = (4, 3, 2)
    summaries = []
    oct_maps = []
    for k in range(3):
        oc = _octave(cor_pyr[k], P)
        blocks.append(_bl2(_pool(oc["lumN"], cgrid[k]), (1.0, 1.0, 0.9)[k]))
        blocks.append(_bl2(_pool(oc["chrN"], cgrid[k]), 0.7))
        del oc
        o = _octave(ret_pyr[k], P)
        allc = torch.cat([o["lumN"], o["chrN"]], 1)
        blocks.append(_bl2(_pool_std(allc, rgrid[k]), 0.6))
        blocks.append(_bl2(_pool_max(allc, rgrid[k]), 0.6))
        Eo, S = _summary(o)
        for _, v, w in _derived(o, Eo, S, P):
            blocks.append(_bl2(v, w))
        summaries.append(S)
        oct_maps.append((o["lumN"].mean(1, keepdim=True), o["chrN"].mean(1, keepdim=True)))
        del o, allc

    # cross-octave persistence and differences
    pers, diffs = [], []
    for k in range(2):
        up = F.interpolate(summaries[k + 1], size=summaries[k].shape[-2:], mode="bilinear",
                           align_corners=False)
        pers.append(_corr_spatial(summaries[k], up))
        diffs.append(_pool(summaries[k] - up, 4))
    blocks.append(_bl2(torch.cat(pers, 1), 0.4))
    blocks.append(_bl2(torch.cat(diffs, 1), 0.5))

    # five-octave surface geometry
    sgrid = (8, 6, 4, 3, 2)
    sw = (1.5, 1.2, 0.9, 0.6, 0.35)
    lvl = opp
    smaps = []
    for k in range(5):
        if k > 0:
            lvl = _down(lvl, P)
        sm = _surface(lvl, P)
        blocks.append(_bl2(_pool(sm, sgrid[k]), sw[k]))
        smaps.append(sm)
    xc = []
    for i in range(5):
        for j in range(i + 1, 5):
            a = F.adaptive_avg_pool2d(smaps[i], smaps[j].shape[-2:])
            xc.append(_corr_spatial(a, smaps[j]))
    blocks.append(_bl2(torch.cat(xc, 1), 0.6))
    del smaps

    # extras: coarse colour grids
    blocks.append(_bl2(torch.cat([_pool(opp, 8), _pool_std(opp, 4)], 1), 0.6))

    # extras: anchor-distance ring maps (16x16)
    amap = torch.cat([F.adaptive_avg_pool2d(oct_maps[1][0], 16),
                      F.adaptive_avg_pool2d(oct_maps[1][1], 16),
                      F.adaptive_avg_pool2d(ret_pyr[2][:, 0:1], 16)], 1)   # (B,3,16,16)
    yy, xx = _coords(16, 16, X.device)
    edges = (0.0, 0.4, 0.8, 1.2, 1.7, 3.0)
    feats = []
    for ay in (-1.0, 0.0, 1.0):
        for ax in (-1.0, 0.0, 1.0):
            dist = torch.sqrt((yy - ay) ** 2 + (xx - ax) ** 2)
            for e0, e1 in zip(edges[:-1], edges[1:]):
                m = ((dist >= e0) & (dist < e1)).float()
                feats.append((amap * m).sum((2, 3)) / (m.sum() + 1e-6))
    blocks.append(_bl2(torch.cat(feats, 1), 0.4))

    # extras: local-extremum densities (luminance and energy)
    ext = []
    for k in range(3):
        Lk = _blur(ret_pyr[k][:, 0:1], P) if k == 0 else ret_pyr[k][:, 0:1]
        for x, thr in ((Lk, 0.02), (oct_maps[k][0], 0.05)):
            mx = F.max_pool2d(x, 3, stride=1, padding=1)
            mn = -F.max_pool2d(-x, 3, stride=1, padding=1)
            rng = mx - mn
            ismax = ((x >= mx) & (rng > thr)).float()
            ismin = ((x <= mn) & (rng > thr)).float()
            ext.append(_pool(torch.cat([ismax, ismin], 1), 4))
    blocks.append(_bl2(torch.cat(ext, 1), 0.4))

    out = torch.cat(blocks, 1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def MODEL(X):
    X = torch.as_tensor(X)
    dev = X.device
    P = _params(dev)
    outs = []
    for i in range(0, X.shape[0], _CHUNK):
        outs.append(_features(X[i:i + _CHUNK].float(), P))
    return torch.cat(outs, 0)
