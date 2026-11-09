"""
Fixed, training-free image feature model for fMRI encoding (early + ventral visual cortex).

Pipeline (all parameters analytic; < 3k materialized filter weights):
  * compressive gamma ((x+0.04)^0.45, rescaled to [0,1]) -> orthonormal opponent space
    (luminance, red-green, blue-yellow)
  * a foveally magnified "cortical" copy (each axis warped by 0.58u + 0.42u^3)
  * 3 octaves of 7x7 quadrature Gabors (8 orientations x 2 frequencies) + 2 DoGs on luminance,
    coarse quadrature Gabors on the chromatic channels; divisive normalisation by
    (local 3x3 + 0.5 * global) mean energy, log1p compression
  * means pooled on cortical grids (9/6/4), std + peaks on retinal grids
  * derived maps (orientation harmonics, orientation-field curvature, pop-out, phase congruency,
    bar/edge polarity, second-order modulation, channel correlations, cross-octave persistence,
    fovea/periphery contrast, mirror correlations)
  * five-octave surface geometry (Hessian shape index x curvedness, structure tensor,
    isophote curvature, chromatic corners) on shrinking grids, weights 1.5 -> 0.35
  * extras: coarse colour grids, anchor path maps, local-extremum counts
Every block is L2-normalised and weighted. Output dimension ~9.2k (< 10,000).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

IMG = 128
CHUNK = 32
EPS = 1e-6


# ----------------------------------------------------------------------------- filter design
def _coords(k):
    r = (k - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x.astype(np.float64), y.astype(np.float64)


def _gabor(k, f, theta, sigma, odd):
    x, y = _coords(k)
    xr = x * math.cos(theta) + y * math.sin(theta)
    yr = -x * math.sin(theta) + y * math.cos(theta)
    env = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2))
    g = env * (np.sin(2 * math.pi * f * xr) if odd else np.cos(2 * math.pi * f * xr))
    g = g - env * (g.sum() / env.sum())  # zero DC, keeps envelope shape
    return g / (np.linalg.norm(g) + 1e-12)


def _gauss2d(k, s):
    x, y = _coords(k)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))
    return g / g.sum()


def _dog(k, sc, ss):
    d = _gauss2d(k, sc) - _gauss2d(k, ss)
    d = d - d.mean()
    return d / (np.linalg.norm(d) + 1e-12)


def _gauss1d(k, s):
    x = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-x ** 2 / (2 * s ** 2))
    return g / g.sum()


# ----------------------------------------------------------------------------- small helpers
def _conv(x, w, groups=1):
    p = w.shape[-1] // 2
    return F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"), w, groups=groups)


def _blur(x, g):
    n, c, h, w = x.shape
    k = g.numel()
    p = k // 2
    y = x.reshape(n * c, 1, h, w)
    y = F.conv2d(F.pad(y, (p, p, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    y = F.conv2d(F.pad(y, (0, 0, p, p), mode="reflect"), g.view(1, 1, k, 1))
    return y.view(n, c, h, w)


def _dx(x):
    p = F.pad(x, (1, 1, 0, 0), mode="replicate")
    return 0.5 * (p[..., :, 2:] - p[..., :, :-2])


def _dy(x):
    p = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return 0.5 * (p[..., 2:, :] - p[..., :-2, :])


def _dxx(x):
    p = F.pad(x, (1, 1, 0, 0), mode="replicate")
    return p[..., :, 2:] - 2 * x + p[..., :, :-2]


def _dyy(x):
    p = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return p[..., 2:, :] - 2 * x + p[..., :-2, :]


def _eig2(a, b, c):
    """Eigenvalues of symmetric [[a, c], [c, b]] (l1 >= l2)."""
    m = 0.5 * (a + b)
    d = torch.sqrt((0.5 * (a - b)) ** 2 + c ** 2 + 1e-20)
    return m + d, m - d


def _pool(x, g):
    return F.adaptive_avg_pool2d(x, g).flatten(1)


def _pool_std(x, g):
    m = F.adaptive_avg_pool2d(x, g)
    m2 = F.adaptive_avg_pool2d(x * x, g)
    return torch.sqrt((m2 - m * m).clamp_min(0) + 1e-10).flatten(1)


def _pool_max(x, g):
    return F.adaptive_max_pool2d(x, g).flatten(1)


def _offdiag_corr(m):
    n, c = m.shape[:2]
    x = m.flatten(2)
    x = x - x.mean(2, keepdim=True)
    cov = x @ x.transpose(1, 2) / x.shape[2]
    sd = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0) + 1e-8)
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    iu = torch.triu_indices(c, c, 1, device=m.device)
    return corr[:, iu[0], iu[1]]


def _chan_corr(a, b):
    a = a.flatten(2)
    b = b.flatten(2)
    a = a - a.mean(2, keepdim=True)
    b = b - b.mean(2, keepdim=True)
    num = (a * b).sum(2)
    den = torch.sqrt((a * a).sum(2) * (b * b).sum(2) + 1e-10)
    return num / den


def _quadrants(x):
    h, w = x.shape[-2] // 2, x.shape[-1] // 2
    return [x[..., :h, :w], x[..., :h, w:], x[..., h:, :w], x[..., h:, w:]]


def _l2(b):
    return b / (b.norm(dim=1, keepdim=True) + 1e-8)


# ----------------------------------------------------------------------------- model
class SurfaceGeometryV1:
    def __init__(self):
        k = 7
        freqs = [(0.30, 1.5), (0.15, 2.5)]
        self.thetas = np.arange(8) * math.pi / 8
        ev, od = [], []
        for f, s in freqs:
            for t in self.thetas:
                ev.append(_gabor(k, f, t, s, False))
                od.append(_gabor(k, f, t, s, True))
        dogs = [_dog(k, 0.8, 1.6), _dog(k, 1.5, 3.0)]
        lum = np.stack(ev + od + dogs)[:, None]  # (34,1,7,7)
        cth = np.arange(4) * math.pi / 4
        ch = np.stack([_gabor(k, 0.12, t, 2.5, False) for t in cth] +
                      [_gabor(k, 0.12, t, 2.5, True) for t in cth])[:, None]  # (8,1,7,7)
        self._cpu = {
            "lum": torch.tensor(lum, dtype=torch.float32),
            "ch": torch.tensor(ch, dtype=torch.float32),
            "g1": torch.tensor(_gauss1d(5, 1.0), dtype=torch.float32),
            "g15": torch.tensor(_gauss1d(7, 1.5), dtype=torch.float32),
        }
        self.num_parameters = int(sum(v.numel() for v in self._cpu.values()))
        self._dev = {}

    # ---------------------------------------------------------------- device cache
    def _w(self, device):
        key = str(device)
        if key not in self._dev:
            d = {k: v.to(device) for k, v in self._cpu.items()}
            d["ch2"] = torch.cat([d["ch"], d["ch"]], 0)  # view-like copy for grouped conv
            self._dev[key] = d
        return self._dev[key]

    # ---------------------------------------------------------------- front end
    @staticmethod
    def _opponent(rgb):
        lo = 0.04 ** 0.45
        g = ((rgb.clamp(0, 1) + 0.04) ** 0.45 - lo) / (1.04 ** 0.45 - lo)
        r, gg, b = g[:, 0:1], g[:, 1:2], g[:, 2:3]
        L = (r + gg + b) / math.sqrt(3)
        RG = (r - gg) / math.sqrt(2)
        BY = (r + gg - 2 * b) / math.sqrt(6)
        return torch.cat([L, RG, BY], 1)

    @staticmethod
    def _cortical(x):
        n, _, h, w = x.shape
        u = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / (h / 2) - 1
        s = 0.58 * u + 0.42 * u ** 3
        gy, gx = torch.meshgrid(s, s, indexing="ij")
        grid = torch.stack([gx, gy], -1)[None].expand(n, h, w, 2)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    @staticmethod
    def _divnorm(r, sigma=0.005):
        p = r.mean(1, keepdim=True)
        loc = F.avg_pool2d(p, 3, 1, 1, count_include_pad=False)
        glob = p.mean((2, 3), keepdim=True)
        return torch.log1p(r / (sigma + loc + 0.5 * glob))

    def _pathway(self, opp, w):
        L, C = opp[:, 0:1], opp[:, 1:3]
        r = _conv(L, w["lum"])
        e, o, d = r[:, :16], r[:, 16:32], r[:, 32:34]
        E = torch.sqrt(e * e + o * o + 1e-12)
        lum = torch.cat([E, F.relu(d), F.relu(-d)], 1)
        rc = _conv(C, w["ch2"], groups=2)
        ce = torch.sqrt(rc[:, 0:4] ** 2 + rc[:, 4:8] ** 2 + 1e-12)
        cb = torch.sqrt(rc[:, 8:12] ** 2 + rc[:, 12:16] ** 2 + 1e-12)
        ch = torch.cat([ce, cb], 1)
        return {"e": e, "o": o, "d": d, "E": E,
                "lumN": self._divnorm(lum), "chN": self._divnorm(ch)}

    # ---------------------------------------------------------------- geometry
    def _geometry(self, opp, w):
        grids = [8, 6, 4, 3, 2]
        weights = [1.5, 1.15, 0.85, 0.6, 0.35]
        centers = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=opp.device).view(1, 5, 1, 1)
        blocks, maps_all, edge32 = [], [], None
        x = opp
        for oc in range(5):
            if oc > 0:
                x = F.avg_pool2d(x, 2)
            s = _blur(x, w["g1"])
            Ls = s[:, 0:1]
            Lx, Ly = _dx(Ls), _dy(Ls)
            Lxx, Lyy, Lxy = _dxx(Ls), _dyy(Ls), _dy(_dx(Ls))
            # Hessian shape index weighted by normalised curvedness
            k1, k2 = _eig2(Lxx, Lyy, Lxy)
            si = (2 / math.pi) * torch.atan2(k1 + k2, k1 - k2)
            cv = torch.sqrt(0.5 * (k1 * k1 + k2 * k2) + 1e-20)
            cm = cv.mean((2, 3), keepdim=True) + 1e-6
            cn = cv / (cv + cm)
            bins = F.relu(1 - (si - centers).abs() / 0.5) * cn
            curved = torch.log1p(cv / cm)
            # structure tensor
            Jxx = _blur(Lx * Lx, w["g15"])
            Jyy = _blur(Ly * Ly, w["g15"])
            Jxy = _blur(Lx * Ly, w["g15"])
            m1, m2 = _eig2(Jxx, Jyy, Jxy)
            m1, m2 = m1.clamp_min(0), m2.clamp_min(0)
            trm = (Jxx + Jyy).mean((2, 3), keepdim=True) + 1e-10
            coh = ((m1 - m2) / (m1 + m2 + 1e-3 * trm + 1e-10)) ** 2
            st = torch.cat([torch.log1p(m1 / trm), torch.log1p(m2 / trm), coh], 1)
            # isophote curvature
            g2 = Lx * Lx + Ly * Ly
            gm = g2.mean((2, 3), keepdim=True)
            num = Lx * Lx * Lyy - 2 * Lx * Ly * Lxy + Ly * Ly * Lxx
            kap = torch.tanh(-2.0 * num / (g2 + 0.1 * gm + 1e-10) ** 1.5)
            gmag = torch.sqrt(g2 + 1e-20)
            wg = gmag / (gmag + torch.sqrt(gm + 1e-20) + 1e-6)
            iso = torch.cat([wg * F.relu(kap), wg * F.relu(-kap)], 1)
            # chromatic (colour-tensor) corner strength
            Cx, Cy = _dx(s), _dy(s)
            Cxx = _blur((Cx * Cx).sum(1, keepdim=True), w["g15"])
            Cyy = _blur((Cy * Cy).sum(1, keepdim=True), w["g15"])
            Cxy = _blur((Cx * Cy).sum(1, keepdim=True), w["g15"])
            ctr = Cxx + Cyy
            ctm = ctr.mean((2, 3), keepdim=True) + 1e-10
            corner = ((Cxx * Cyy - Cxy * Cxy).clamp_min(0) / (ctr + 1e-3 * ctm + 1e-10))
            corner = torch.log1p(corner / ctm)
            maps = torch.cat([bins, curved, st, iso, corner], 1)  # 12 maps
            maps_all.append(maps)
            blocks.append((_pool(maps, grids[oc]), weights[oc]))
            if oc == 2:
                edge32 = gmag / (gmag.mean((2, 3), keepdim=True) + 1e-6)
        xc = [_chan_corr(F.avg_pool2d(maps_all[i], 2), maps_all[i + 1]) for i in range(4)]
        blocks.append((torch.cat(xc, 1), 0.5))
        return blocks, edge32

    # ---------------------------------------------------------------- derived
    def _derived(self, P, w, oc):
        lumN, chN, e, o, E = P["lumN"], P["chN"], P["e"], P["o"], P["E"]
        n, _, h, ww = lumN.shape
        dev = lumN.device
        th = torch.tensor(self.thetas, dtype=lumN.dtype, device=dev).view(1, 8, 1, 1)
        Eo = lumN[:, :16].view(n, 2, 8, h, ww).sum(1)
        S = Eo.sum(1, keepdim=True) + EPS
        c2, s2, c4, s4 = torch.cos(2 * th), torch.sin(2 * th), torch.cos(4 * th), torch.sin(4 * th)
        vx, vy = (Eo * c2).sum(1, keepdim=True), (Eo * s2).sum(1, keepdim=True)
        harm = torch.cat([vx / S, vy / S, (Eo * c4).sum(1, keepdim=True) / S,
                          (Eo * s4).sum(1, keepdim=True) / S], 1)
        # orientation-field curvature
        bx = F.avg_pool2d(vx, 3, 1, 1, count_include_pad=False)
        by = F.avg_pool2d(vy, 3, 1, 1, count_include_pad=False)
        mag = torch.sqrt(bx * bx + by * by + 1e-12)
        ux, uy = bx / mag, by / mag
        curv = torch.sqrt(_dx(ux) ** 2 + _dy(ux) ** 2 + _dx(uy) ** 2 + _dy(uy) ** 2 + 1e-12)
        curv = curv.clamp_max(2.0) * (mag / S)
        # orientation pop-out
        pop = F.relu(Eo - F.avg_pool2d(Eo, 9, 1, 4, count_include_pad=False)).mean(1, keepdim=True)
        # phase congruency across frequencies
        er, orr = e.view(n, 2, 8, h, ww), o.view(n, 2, 8, h, ww)
        num = torch.sqrt(er.sum(1) ** 2 + orr.sum(1) ** 2 + 1e-12).sum(1, keepdim=True)
        Es = E.sum(1, keepdim=True)
        pc = num / (Es + 1e-4)
        # bar vs edge, polarity
        ae, ao = e.abs(), o.abs()
        b = (ae - ao) / (ae + ao + 1e-6)
        bar = (E * b).sum(1, keepdim=True) / (Es + 1e-4)
        pol = e.sum(1, keepdim=True) / (Es + 1e-4)
        maps9 = torch.cat([harm, curv, pop, pc, bar, pol], 1)
        out = [(_pool(maps9, 4), 0.6)]
        # second-order (filter-rectify-filter)
        T = lumN[:, :16].mean(1, keepdim=True)
        r2 = _conv(F.avg_pool2d(T, 2), w["ch"])
        so = torch.sqrt(r2[:, :4] ** 2 + r2[:, 4:] ** 2 + 1e-12)
        out.append((_pool(so, 3), 0.4))
        # compact channel set
        dogN = lumN[:, 16:20]
        comp = torch.cat([Eo / 2, dogN[:, :2].sum(1, keepdim=True), dogN[:, 2:].sum(1, keepdim=True),
                          chN[:, :4].sum(1, keepdim=True), chN[:, 4:].sum(1, keepdim=True)], 1)
        out.append((_offdiag_corr(comp), 0.5))
        if oc < 2:
            out.append((torch.cat([_offdiag_corr(q) for q in _quadrants(comp)], 1), 0.4))
        # fovea vs periphery
        all28 = torch.cat([lumN, chN], 1)
        yy = (torch.arange(h, device=dev, dtype=lumN.dtype) + 0.5) / (h / 2) - 1
        xx = (torch.arange(ww, device=dev, dtype=lumN.dtype) + 0.5) / (ww / 2) - 1
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        rr = torch.sqrt(gx ** 2 + gy ** 2)
        rings = [(rr < 0.3), (rr >= 0.3) & (rr < 0.65), (rr >= 0.65)]
        ms = []
        for m in rings:
            m = m.to(lumN.dtype)
            ms.append((all28 * m).sum((2, 3)) / m.sum())
        out.append((torch.cat([ms[0] - ms[1], ms[1] - ms[2]], 1), 0.4))
        # mirror correlations
        mir = torch.cat([_chan_corr(comp, comp.flip(-1)), _chan_corr(comp, comp.flip(-2))], 1)
        out.append((mir, 0.3))
        return out, Eo, T

    # ---------------------------------------------------------------- extras
    @staticmethod
    def _paths(edge):
        n = edge.shape[0]
        dev, dt = edge.device, edge.dtype
        anchors = torch.tensor([[0, 0], [-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]], device=dev, dtype=dt)
        t = (torch.arange(16, device=dev, dtype=dt) + 0.5) / 8 - 1
        ty, tx = torch.meshgrid(t, t, indexing="ij")
        tgt = torch.stack([tx, ty], -1).view(1, 256, 2)
        K = 12
        sk = ((torch.arange(K, device=dev, dtype=dt) + 0.5) / K).view(1, 1, K, 1)
        a = anchors.view(5, 1, 1, 2)
        pts = a + sk * (tgt.view(1, 256, 1, 2) - a)  # (5,256,K,2)
        dist = (tgt.view(1, 256, 2) - anchors.view(5, 1, 2)).norm(dim=-1)  # (5,256)
        grid = pts.reshape(1, 5 * 256, K, 2).expand(n, -1, -1, -1)
        smp = F.grid_sample(edge, grid, mode="bilinear", padding_mode="border", align_corners=False)
        smp = smp.view(n, 5, 256, K)
        integ = (smp.mean(-1) * dist[None]).view(n, 5, 16, 16)
        peak = smp.amax(-1).view(n, 5, 16, 16)
        return torch.cat([_pool(integ, 4), _pool(peak, 4)], 1)

    @staticmethod
    def _extrema(d, T):
        feats = []

        def regions(m):
            return [m.mean((2, 3))] + [q.mean((2, 3)) for q in _quadrants(m)]

        sd = d.flatten(2).std(2).view(d.shape[0], d.shape[1], 1, 1)
        thr = 0.25 * sd
        mx = ((d >= F.max_pool2d(d, 3, 1, 1)) & (d > thr)).to(d.dtype)
        mn = ((-d >= F.max_pool2d(-d, 3, 1, 1)) & (-d > thr)).to(d.dtype)
        tm = ((T >= F.max_pool2d(T, 3, 1, 1)) & (T > T.mean((2, 3), keepdim=True))).to(d.dtype)
        for m in (mx, mn, tm):
            feats += regions(m)
        return torch.cat(feats, 1)

    # ---------------------------------------------------------------- forward
    def _features(self, x):
        w = self._w(x.device)
        n = x.shape[0]
        rgb = x.reshape(n, IMG, IMG, 3).permute(0, 3, 1, 2).contiguous()
        opp = self._opponent(rgb)
        cort = self._cortical(opp)
        blocks = []

        ret_pyr, cor_pyr = [opp], [cort]
        for _ in range(2):
            ret_pyr.append(F.avg_pool2d(ret_pyr[-1], 2))
            cor_pyr.append(F.avg_pool2d(cor_pyr[-1], 2))

        Eos, Ts, ext = [], [], []
        for oc, g in enumerate([9, 6, 4]):
            Pc = self._pathway(cor_pyr[oc], w)
            blocks.append((_pool(Pc["lumN"], g), 1.0))
            blocks.append((_pool(Pc["chN"], g), 0.7))
            del Pc
            P = self._pathway(ret_pyr[oc], w)
            a28 = torch.cat([P["lumN"], P["chN"]], 1)
            blocks.append((_pool_std(a28, 3), 0.6))
            blocks.append((_pool_max(a28, 3), 0.6))
            dv, Eo, T = self._derived(P, w, oc)
            blocks += dv
            Eos.append(Eo)
            Ts.append(T)
            ext.append(self._extrema(P["d"], T))
            del P, a28

        # cross-octave persistence / differences
        xo = []
        for i in range(2):
            a, b = F.avg_pool2d(Eos[i], 2), Eos[i + 1]
            xo += [_pool(torch.minimum(a, b), 3), _pool(a - b, 3)]
            ta, tb = F.avg_pool2d(Ts[i], 2), Ts[i + 1]
            xo += [_pool(torch.minimum(ta, tb), 4), _pool(ta - tb, 4)]
        blocks.append((torch.cat(xo, 1), 0.5))

        geo, edge32 = self._geometry(opp, w)
        blocks += geo

        blocks.append((_pool(opp, 8), 0.6))
        blocks.append((_pool_std(opp, 4), 0.4))
        blocks.append((self._paths(edge32), 0.4))
        blocks.append((torch.cat(ext, 1), 0.3))

        out = torch.cat([_l2(b) * wt for b, wt in blocks], 1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X)
        if X.dim() == 1:
            X = X[None]
        X = X.float()
        outs = [self._features(X[i:i + CHUNK]) for i in range(0, X.shape[0], CHUNK)]
        return torch.cat(outs, 0)


MODEL = SurfaceGeometryV1()
