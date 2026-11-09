"""
Fixed, training-free image feature model for fMRI encoding (early + ventral visual cortex).

Input : torch float tensor X of shape (N, 128*128*3), flattened HWC RGB in [0, 1].
Output: finite float tensor of shape (N, D), D < 10,000.

All filters are analytic; no learned or pretrained weights. Each image is processed
independently (all "global" statistics are per image). Images are processed in chunks.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

_H = 128
_CHUNK = 32
_EPS = 1e-6
_SQ2, _SQ3, _SQ6 = math.sqrt(2.0), math.sqrt(3.0), math.sqrt(6.0)


# ----------------------------------------------------------------------------- filters
def _gabor_bank(size, freq_sigmas, n_orient):
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    evens, odds = [], []
    for f, s in freq_sigmas:
        for k in range(n_orient):
            th = math.pi * k / n_orient
            xr = x * math.cos(th) + y * math.sin(th)
            g = np.exp(-(x ** 2 + y ** 2) / (2.0 * s * s))
            e = g * np.cos(2 * math.pi * f * xr)
            e = e - g * (e.sum() / g.sum())          # remove DC
            o = g * np.sin(2 * math.pi * f * xr)
            o = o - o.mean()
            evens.append(e / np.linalg.norm(e))
            odds.append(o / np.linalg.norm(o))
    return np.array(evens), np.array(odds)


def _dog(size, s1, s2):
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g1 = np.exp(-(x ** 2 + y ** 2) / (2 * s1 * s1))
    g2 = np.exp(-(x ** 2 + y ** 2) / (2 * s2 * s2))
    d = g1 / g1.sum() - g2 / g2.sum()
    d = d - d.mean()
    return d / (np.abs(d).sum() / 2.0)


def _gk(s, device, dtype):
    r = max(1, int(math.ceil(3 * s)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-t ** 2 / (2 * s * s))
    return k / k.sum()


def _blur(x, s):
    k = _gk(s, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    C = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous(), groups=C)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous(), groups=C)
    return x


def _dx(x):
    p = F.pad(x, (1, 1, 0, 0), mode="replicate")
    return (p[..., 2:] - p[..., :-2]) * 0.5


def _dy(x):
    p = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return (p[..., 2:, :] - p[..., :-2, :]) * 0.5


def _dxx(x):
    p = F.pad(x, (1, 1, 0, 0), mode="replicate")
    return p[..., 2:] - 2 * p[..., 1:-1] + p[..., :-2]


def _dyy(x):
    p = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return p[..., 2:, :] - 2 * p[..., 1:-1, :] + p[..., :-2, :]


def _coords(h, device, dtype):
    lin = torch.linspace(-1 + 1.0 / h, 1 - 1.0 / h, h, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return yy, xx


def _corr_offdiag(Z, iu):
    # Z: (n, C, P) -> (n, C*(C-1)/2)
    Z = Z - Z.mean(2, keepdim=True)
    cov = Z @ Z.transpose(1, 2) / Z.shape[2]
    d = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0) + 1e-8)
    c = cov / (d[:, :, None] * d[:, None, :])
    return c[:, iu[0], iu[1]]


def _corr_pair(A, B):
    # A, B: (n, C, h, w) -> per-channel spatial correlation (n, C)
    A = A.flatten(2)
    B = B.flatten(2)
    A = A - A.mean(2, keepdim=True)
    B = B - B.mean(2, keepdim=True)
    num = (A * B).mean(2)
    den = torch.sqrt((A * A).mean(2) * (B * B).mean(2) + 1e-8)
    return num / den


def _pool_std(x, g):
    m1 = F.adaptive_avg_pool2d(x, g)
    m2 = F.adaptive_avg_pool2d(x * x, g)
    return torch.sqrt(F.relu(m2 - m1 * m1) + 1e-8)


# ----------------------------------------------------------------------------- model
class _FixedVisualModel:
    def __init__(self):
        ev, od = _gabor_bank(7, [(0.25, 1.4), (0.125, 2.0)], 8)      # 16 + 16
        d1 = _dog(7, 0.8, 1.6)
        d2 = _dog(7, 1.4, 2.8)
        self._klum = torch.tensor(np.concatenate([ev, od, d1[None], d2[None]], 0),
                                  dtype=torch.float32).unsqueeze(1)          # 34
        cev, cod = _gabor_bank(7, [(0.1, 2.2)], 4)                    # coarse, 4 + 4
        self._kchrom = torch.tensor(np.concatenate([cev, cod, d2[None]], 0),
                                    dtype=torch.float32).unsqueeze(1)        # 9
        self._k2 = torch.tensor(np.concatenate([cev, cod], 0),
                                dtype=torch.float32).unsqueeze(1)            # 8
        th = torch.arange(8, dtype=torch.float32) * math.pi / 8
        self._c2 = torch.cos(2 * th).view(1, 8, 1, 1)
        self._s2 = torch.sin(2 * th).view(1, 8, 1, 1)
        self._c4 = torch.cos(4 * th).view(1, 8, 1, 1)
        self._s4 = torch.sin(4 * th).view(1, 8, 1, 1)
        self._iu10 = torch.triu_indices(10, 10, 1)
        self._cache = {}
        self.n_params = sum(t.numel() for t in
                            [self._klum, self._kchrom, self._k2,
                             self._c2, self._s2, self._c4, self._s4])

    def _dev(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = {
                "klum": self._klum.to(device), "kchrom": self._kchrom.to(device),
                "k2": self._k2.to(device), "c2": self._c2.to(device),
                "s2": self._s2.to(device), "c4": self._c4.to(device),
                "s4": self._s4.to(device), "iu10": self._iu10.to(device),
            }
        return self._cache[key]

    # divisive normalization: local 3x3 + half global mean energy, then log1p
    @staticmethod
    def _norm(E):
        pool = E.mean(1, keepdim=True)
        loc = F.avg_pool2d(pool, 3, 1, 1, count_include_pad=False)
        glob = pool.mean((2, 3), keepdim=True)
        return torch.log1p(E / (loc + 0.5 * glob + 1e-3))

    def _pathway(self, opp, K):
        B, _, H, W = opp.shape
        r = F.conv2d(F.pad(opp[:, 0:1], (3, 3, 3, 3), mode="reflect"), K["klum"])
        ev, od = r[:, :16], r[:, 16:32]
        E = torch.sqrt(ev * ev + od * od + 1e-8)
        dog = r[:, 32:34]
        lum = torch.cat([E, F.relu(dog), F.relu(-dog)], 1)                    # 20
        lumn = self._norm(lum)
        C = F.pad(opp[:, 1:3].reshape(B * 2, 1, H, W), (3, 3, 3, 3), mode="reflect")
        rc = F.conv2d(C, K["kchrom"])
        Ec = torch.sqrt(rc[:, :4] ** 2 + rc[:, 4:8] ** 2 + 1e-8)
        dc = rc[:, 8:9]
        ch = torch.cat([Ec, F.relu(dc), F.relu(-dc)], 1)                     # 2B,6
        chn = self._norm(ch).view(B, 12, H, W)
        return torch.cat([lumn, chn], 1), ev, od                             # 32 ch

    def _derived(self, nm, ev, od, K):
        n, _, H, W = nm.shape
        O = nm[:, :16].view(n, 2, 8, H, W).sum(1)                             # (n,8,H,W)
        S = O.sum(1, keepdim=True) + 1e-4
        re1 = (O * K["c2"]).sum(1, keepdim=True) / S
        im1 = (O * K["s2"]).sum(1, keepdim=True) / S
        h2 = torch.sqrt(((O * K["c4"]).sum(1, keepdim=True)) ** 2 +
                        ((O * K["s4"]).sum(1, keepdim=True)) ** 2 + 1e-10) / S
        # orientation-field curvature
        mag = torch.sqrt(re1 ** 2 + im1 ** 2 + 1e-8)
        u, v = re1 / mag, im1 / mag
        curv = torch.sqrt(_dx(u) ** 2 + _dy(u) ** 2 + _dx(v) ** 2 + _dy(v) ** 2 + 1e-8)
        curv = (curv * mag).clamp(max=2.0)
        # surround pop-out
        sur = F.avg_pool2d(O, 9, 1, 4, count_include_pad=False)
        pop = (O - sur).abs().sum(1, keepdim=True) / (sur.sum(1, keepdim=True) + 1e-4)
        # phase congruency across the two frequencies
        e0, e1 = ev[:, :8], ev[:, 8:]
        o0, o1 = od[:, :8], od[:, 8:]
        a0 = torch.sqrt(e0 ** 2 + o0 ** 2 + 1e-10)
        a1 = torch.sqrt(e1 ** 2 + o1 ** 2 + 1e-10)
        A = a0 + a1
        pc_t = torch.sqrt((e0 + e1) ** 2 + (o0 + o1) ** 2 + 1e-10) / (A + 1e-4)
        pc = (pc_t * A).sum(1, keepdim=True) / (A.sum(1, keepdim=True) + 1e-4)
        # bar vs edge, and bar polarity
        ae, ao = ev.abs(), od.abs()
        bar = (ae - ao).sum(1, keepdim=True) / ((ae + ao).sum(1, keepdim=True) + 1e-4)
        pol = ev.sum(1, keepdim=True) / (ae.sum(1, keepdim=True) + 1e-4)
        # second-order (contrast-modulation) energy
        T = S / (S.mean((2, 3), keepdim=True) + 1e-4)
        r2 = F.conv2d(F.pad(T, (3, 3, 3, 3), mode="reflect"), K["k2"])
        so = torch.log1p(torch.sqrt(r2[:, :4] ** 2 + r2[:, 4:] ** 2 + 1e-8))
        maps = torch.cat([re1, im1, h2, curv, pop, pc, bar, pol, so], 1)      # 12
        R10 = torch.cat([O, nm[:, 16:20].sum(1, keepdim=True),
                         nm[:, 20:32].sum(1, keepdim=True)], 1)
        return maps, O, R10

    @staticmethod
    def _sym_eig(a, b, c):
        # eigenvalues of [[a, c], [c, b]]
        tr = 0.5 * (a + b)
        disc = torch.sqrt(0.25 * (a - b) ** 2 + c * c + 1e-12)
        return tr + disc, tr - disc, disc

    def _geometry(self, P):
        # P: (n,3,h,w) opponent image at one octave -> (n,15,h,w)
        Ls = _blur(P[:, 0:1], 1.0)
        Cs = _blur(P[:, 1:3], 1.0)
        Lx, Ly = _dx(Ls), _dy(Ls)
        Lxx, Lyy, Lxy = _dxx(Ls), _dyy(Ls), _dy(_dx(Ls))
        k1, k2, disc = self._sym_eig(Lxx, Lyy, Lxy)
        si = -(2.0 / math.pi) * torch.atan2(k1 + k2, 2 * disc)
        cv = torch.sqrt(0.5 * (k1 ** 2 + k2 ** 2) + 1e-12)
        w = torch.log1p(cv / (cv.mean((2, 3), keepdim=True) + 1e-4))
        centers = torch.linspace(-1, 1, 8, device=P.device, dtype=P.dtype).view(1, 8, 1, 1)
        bins = F.relu(1 - (si - centers).abs() / (2.0 / 7)) * w                # 8
        # structure tensor
        Jxx, Jyy, Jxy = _blur(Lx * Lx, 1.5), _blur(Ly * Ly, 1.5), _blur(Lx * Ly, 1.5)
        l1, l2, _ = self._sym_eig(Jxx, Jyy, Jxy)
        l1, l2 = l1.clamp_min(0), l2.clamp_min(0)
        gm = (l1 + l2).mean((2, 3), keepdim=True) + 1e-6
        coh = ((l1 - l2) / (l1 + l2 + 1e-6)) ** 2
        st = torch.cat([torch.log1p(l1 / gm), torch.log1p(l2 / gm), coh], 1)  # 3
        # isophote curvature
        g2 = Lx * Lx + Ly * Ly
        g = torch.sqrt(g2 + 1e-12)
        kap = -(Lxx * Ly * Ly - 2 * Lxy * Lx * Ly + Lyy * Lx * Lx) / (g2 + 1e-6) ** 1.5
        gw = g / (g + g.mean((2, 3), keepdim=True) + 1e-4)
        tk = torch.tanh(2.0 * kap)
        iso = torch.cat([tk * gw, tk.abs() * gw], 1)                          # 2
        # chromatic corner strength
        Cx, Cy = _dx(Cs), _dy(Cs)
        cxx = _blur((Cx * Cx).sum(1, keepdim=True), 1.5)
        cyy = _blur((Cy * Cy).sum(1, keepdim=True), 1.5)
        cxy = _blur((Cx * Cy).sum(1, keepdim=True), 1.5)
        _, c2c, _ = self._sym_eig(cxx, cyy, cxy)
        _, c2f, _ = self._sym_eig(cxx + Jxx, cyy + Jyy, cxy + Jxy)
        cmc = (cxx + cyy).mean((2, 3), keepdim=True) + 1e-6
        cmf = cmc + gm * 2
        cor = torch.cat([torch.log1p(c2c.clamp_min(0) / cmc),
                         torch.log1p(c2f.clamp_min(0) / cmf)], 1)              # 2
        return torch.cat([bins, st, iso, cor], 1), Ls, g

    def _chunk(self, x):
        K = self._dev(x.device)
        n = x.shape[0]
        dev, dt = x.device, torch.float32
        img = x.reshape(n, _H, _H, 3).permute(0, 3, 1, 2).to(dt).clamp(0, 1)
        g0 = 0.04 ** 0.45
        gimg = ((img + 0.04) ** 0.45 - g0) / (1.04 ** 0.45 - g0)
        R, G, B = gimg[:, 0], gimg[:, 1], gimg[:, 2]
        opp = torch.stack([(R + G + B) / _SQ3, (R - G) / _SQ2, (R + G - 2 * B) / _SQ6], 1)

        # foveally magnified cortical copy
        lin = torch.linspace(-1 + 1.0 / _H, 1 - 1.0 / _H, _H, device=dev, dtype=dt)
        wl = 0.58 * lin + 0.42 * lin ** 3
        gy, gx = torch.meshgrid(wl, wl, indexing="ij")
        grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(n, -1, -1, -1)
        cort = F.grid_sample(opp, grid, mode="bilinear", padding_mode="border",
                             align_corners=False)

        blocks = []

        def add(v, w):
            v = v.flatten(1)
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            blocks.append(v * (w / (v.norm(dim=1, keepdim=True) + 1e-8)))

        cur = torch.cat([opp, cort], 0)
        Os, R10s, El1, Ec1 = [], [], None, None
        for o in range(3):
            if o > 0:
                cur = F.avg_pool2d(cur, 2)
            nm, ev, od = self._pathway(cur, K)
            nm_r, nm_c = nm[:n], nm[n:]
            ev_r, od_r = ev[:n], od[:n]
            H = nm.shape[-1]
            # means on cortical grids
            cm = F.adaptive_avg_pool2d(nm_c, [9, 6, 4][o])
            add(cm[:, :20], 1.0)
            add(cm[:, 20:], 0.7)
            # std + peaks on retinal grids
            gs = [3, 2, 1][o]
            add(_pool_std(nm_r, gs), 0.5)
            add(F.adaptive_max_pool2d(nm_r, gs), 0.5)
            # derived maps
            maps, O, R10 = self._derived(nm_r, ev_r, od_r, K)
            add(F.adaptive_avg_pool2d(maps, 4), 0.6)
            Os.append(O)
            R10s.append(R10)
            # channel correlations: global + quadrants
            cs = [_corr_offdiag(R10.flatten(2), K["iu10"])]
            h2 = H // 2
            for qy in range(2):
                for qx in range(2):
                    q = R10[:, :, qy * h2:(qy + 1) * h2, qx * h2:(qx + 1) * h2]
                    cs.append(_corr_offdiag(q.flatten(2), K["iu10"]))
            add(torch.cat(cs, 1), 0.45)
            # foveal vs peripheral contrast
            yy, xx = _coords(H, dev, dt)
            rr = torch.sqrt(yy ** 2 + xx ** 2)
            fm = (rr < 0.35).to(dt)
            pm = (rr > 0.65).to(dt)
            fv = (nm_r * fm).sum((2, 3)) / fm.sum()
            pv = (nm_r * pm).sum((2, 3)) / pm.sum()
            add(torch.cat([fv - pv, torch.log((fv + 1e-3) / (pv + 1e-3))], 1), 0.4)
            # mirror correlations
            mc = [_corr_pair(R10, torch.flip(R10, [3])), _corr_pair(R10, torch.flip(R10, [2]))]
            if o == 0:
                L0 = opp[:, 0:1]
                mc += [_corr_pair(L0, torch.flip(L0, [3])), _corr_pair(L0, torch.flip(L0, [2]))]
            add(torch.cat(mc, 1), 0.3)
            if o == 1:
                El1 = nm_r[:, :16].sum(1, keepdim=True)
                Ec1 = nm_r[:, 20:32].sum(1, keepdim=True)
            del nm, ev, od, nm_r, nm_c, maps

        # cross-octave persistence and differences
        for k in range(2):
            a = F.avg_pool2d(Os[k], 2)
            b = Os[k + 1]
            add(F.adaptive_avg_pool2d(torch.sqrt(a * b + 1e-8), 3), 0.5)
            add(F.adaptive_avg_pool2d(a - b, 3), 0.5)

        # five-octave surface geometry (most heavily weighted)
        pyr = [opp]
        for _ in range(4):
            pyr.append(F.avg_pool2d(_blur(pyr[-1], 1.0), 2))
        gw_oct = [1.5, 1.1, 0.8, 0.55, 0.35]
        grids = [8, 6, 4, 3, 2]
        groups = [(0, 8, 1.0), (8, 11, 0.8), (11, 13, 0.6), (13, 15, 0.6)]
        Gs, Lss, gms = [], [], []
        for k in range(5):
            Gm, Ls, gmag = self._geometry(pyr[k])
            pooled = F.adaptive_avg_pool2d(Gm, grids[k])
            for a0, a1, wg in groups:
                add(pooled[:, a0:a1], gw_oct[k] * wg)
            Gs.append(Gm)
            Lss.append(Ls)
            gms.append(gmag)
        xc = [_corr_pair(F.avg_pool2d(Gs[k], 2), Gs[k + 1]) for k in range(4)]
        add(torch.cat(xc, 1), 0.5)
        stats = [torch.cat([G.mean((2, 3)), G.flatten(2).std(2)], 1) for G in Gs]
        add(torch.cat(stats, 1), 0.5)

        # coarse color grids
        add(F.adaptive_avg_pool2d(opp, 8), 0.6)
        add(_pool_std(opp, 4), 0.4)

        # anchor-distance maps (rings around anchors) and anchor-to-anchor paths
        h1 = El1.shape[-1]
        yy, xx = _coords(h1, dev, dt)
        anchors = [(0.0, 0.0), (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
        edges = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 3.0]
        EM = torch.cat([El1, Ec1], 1)
        EMn = EM / (EM.mean((2, 3), keepdim=True) + 1e-4)
        rings = []
        for ay, ax in anchors:
            d = torch.sqrt((yy - ay) ** 2 + (xx - ax) ** 2)
            for i in range(len(edges) - 1):
                m = ((d >= edges[i]) & (d < edges[i + 1])).to(dt)
                rings.append((EMn * m).sum((2, 3)) / (m.sum() + 1e-6))
        add(torch.cat(rings, 1), 0.4)
        ts = torch.linspace(0.06, 0.94, 8, device=dev, dtype=dt)
        pts = []
        for i in range(5):
            for j in range(i + 1, 5):
                ya, xa = anchors[i]
                yb, xb = anchors[j]
                pts.append(torch.stack([xa + (xb - xa) * ts, ya + (yb - ya) * ts], -1))
        pts = torch.cat(pts, 0).view(1, 1, -1, 2).expand(n, -1, -1, -1)
        smp = F.grid_sample(_blur(EMn[:, 0:1], 1.5), pts, mode="bilinear",
                            padding_mode="border", align_corners=False)
        add(smp, 0.35)

        # local-extremum counts (densities, global + quadrants), octaves 0..3
        ext = []
        for k in range(4):
            Ls, gmag = Lss[k], gms[k]
            dl = Ls - F.avg_pool2d(Ls, 5, 1, 2, count_include_pad=False)
            thr = dl.flatten(2).std(2).view(n, 1, 1, 1)
            mx = ((Ls >= F.max_pool2d(Ls, 3, 1, 1)) & (dl > thr)).to(dt)
            mn = ((-Ls >= F.max_pool2d(-Ls, 3, 1, 1)) & (dl < -thr)).to(dt)
            gthr = (gmag.mean((2, 3)) + gmag.flatten(2).std(2)).view(n, 1, 1, 1)
            gx_ = ((gmag >= F.max_pool2d(gmag, 3, 1, 1)) & (gmag > gthr)).to(dt)
            M = torch.cat([mx, mn, gx_], 1)
            ext.append(M.mean((2, 3)))
            ext.append(F.adaptive_avg_pool2d(M, 2).flatten(1))
        add(torch.cat(ext, 1), 0.3)

        out = torch.cat(blocks, 1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        outs = []
        for i in range(0, X.shape[0], _CHUNK):
            outs.append(self._chunk(X[i:i + _CHUNK].float()))
        return torch.cat(outs, 0)


MODEL = _FixedVisualModel()
