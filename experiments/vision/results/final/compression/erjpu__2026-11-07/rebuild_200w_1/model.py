"""Fixed, training-free image feature model for fMRI encoding (early + ventral visual cortex).

MODEL(X): X is (N, 128*128*3) float tensor, flattened HWC RGB in [0, 1].
Returns (N, D) finite float32 features, D < 10,000. Every image is processed
independently; all filters are analytic (no learned / pretrained weights).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

IMG = 128
CHUNK = 32


# ----------------------------------------------------------------------------- helpers
def _gauss1d(sigma, device):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _blur(x, sigma):
    C = x.shape[1]
    g = _gauss1d(sigma, x.device)
    k = g.numel()
    r = k // 2
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(C, 1, 1, k).contiguous(), groups=C)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, g.view(1, 1, k, 1).expand(C, 1, k, 1).contiguous(), groups=C)


def _conv(x, w, groups=1):
    r = w.shape[-1] // 2
    return F.conv2d(F.pad(x, (r, r, r, r), mode="replicate"), w, groups=groups)


def _gabor_bank(freqs, sigmas, n_orient, radius=3):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    ks = []
    for f, s in zip(freqs, sigmas):
        env = np.exp(-(x ** 2 + y ** 2) / (2.0 * s ** 2))
        for k in range(n_orient):
            th = k * np.pi / n_orient
            xr = x * np.cos(th) + y * np.sin(th)
            ev = env * np.cos(2 * np.pi * f * xr)
            ev = ev - env * (ev.sum() / env.sum())  # zero DC
            od = env * np.sin(2 * np.pi * f * xr)
            ks += [ev / np.linalg.norm(ev), od / np.linalg.norm(od)]
    return torch.tensor(np.stack(ks)[:, None], dtype=torch.float32)


def _divnorm(E):
    """Divisive normalisation by local 3x3 + half global mean energy, then log1p."""
    m = E.mean(1, keepdim=True)
    loc = F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), 3, 1)
    glob = m.mean((2, 3), keepdim=True)
    return torch.log1p(E / (loc + 0.5 * glob + 1e-3))


def _mean_grid(x, g):
    return F.adaptive_avg_pool2d(x, g).flatten(1)


def _std_grid(x, g):
    m = F.adaptive_avg_pool2d(x, g)
    m2 = F.adaptive_avg_pool2d(x * x, g)
    return torch.sqrt(F.relu(m2 - m * m) + 1e-10).flatten(1)


def _max_grid(x, g):
    return F.adaptive_max_pool2d(x, g).flatten(1)


def _grad(x):
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    gy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    return gx, gy


def _derivs(I):
    p = F.pad(I, (1, 1, 1, 1), mode="replicate")
    Ix = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    Iy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    Ixx = p[..., 1:-1, 2:] - 2 * I + p[..., 1:-1, :-2]
    Iyy = p[..., 2:, 1:-1] - 2 * I + p[..., :-2, 1:-1]
    Ixy = 0.25 * (p[..., 2:, 2:] - p[..., 2:, :-2] - p[..., :-2, 2:] + p[..., :-2, :-2])
    return Ix, Iy, Ixx, Iyy, Ixy


def _pcorr(a, b):
    a = a.flatten(2)
    b = b.flatten(2)
    a = a - a.mean(2, keepdim=True)
    b = b - b.mean(2, keepdim=True)
    return (a * b).sum(2) / (a.norm(dim=2) * b.norm(dim=2) + 1e-6)


def _corr_offdiag(x, iu):
    """x: (N, C, P) -> off-diagonal correlations (N, C*(C-1)/2)."""
    x = x - x.mean(2, keepdim=True)
    x = x / (x.norm(dim=2, keepdim=True) + 1e-6)
    c = x @ x.transpose(1, 2)
    return c[:, iu[0], iu[1]]


def _l2(v):
    return v / (v.norm(dim=1, keepdim=True) + 1e-8)


# ----------------------------------------------------------------------------- model
class _FixedVisualModel:
    def __init__(self):
        self._cache = {}

    # fixed analytic constants, built once per device
    def _consts(self, device):
        key = str(device)
        if key in self._cache:
            return self._cache[key]
        c = {}
        c["gab"] = _gabor_bank([0.25, 0.125], [1.4, 2.0], 8).to(device)          # 32 x 1 x 7 x 7
        coarse = _gabor_bank([0.125], [2.0], 4)                                   # 8 x 1 x 7 x 7
        c["coarse"] = coarse.to(device)
        c["coarse2"] = coarse.repeat(2, 1, 1, 1).to(device)                       # for RG, BY
        th = torch.arange(8, dtype=torch.float32, device=device) * math.pi / 8
        c["harm"] = torch.stack([torch.cos(2 * th), torch.sin(2 * th),
                                 torch.cos(4 * th), torch.sin(4 * th)]).view(4, 8, 1, 1)
        u = torch.linspace(-1, 1, IMG, device=device)
        s = 0.58 * u + 0.42 * u ** 3
        gy, gx = torch.meshgrid(s, s, indexing="ij")
        c["warp"] = torch.stack([gx, gy], -1)[None]                              # 1 x H x W x 2
        c["si_centers"] = (-1 + (2 * torch.arange(8, dtype=torch.float32, device=device) + 1) / 8).view(1, 8, 1, 1)
        for h in (128, 64, 32):
            v = torch.linspace(-1, 1, h, device=device)
            yy, xx = torch.meshgrid(v, v, indexing="ij")
            r = torch.sqrt(xx ** 2 + yy ** 2)
            c[f"fov{h}"] = (r < 0.35).float().view(1, 1, h, h)
            c[f"per{h}"] = (r > 0.65).float().view(1, 1, h, h)
        mask = torch.zeros(1, 10, 32, 32, dtype=torch.bool, device=device)
        anchors = [(16, 16), (0, 0), (0, 31), (31, 0), (31, 31)]
        for j in range(2):
            for a, (yy_, xx_) in enumerate(anchors):
                mask[0, j * 5 + a, yy_, xx_] = True
        c["anchor_mask"] = mask
        c["iu12"] = torch.triu_indices(12, 12, 1, device=device)
        self._cache[key] = c
        return c

    # --------------------------------------------------------------- octave front end
    def _octave(self, img, c):
        N, _, h, w = img.shape
        L = img[:, 0:1]
        r = _conv(L, c["gab"]).view(N, 2, 8, 2, h, w)
        e, o = r[:, :, :, 0], r[:, :, :, 1]                                      # N,2,8,h,w
        E = torch.sqrt(e * e + o * o + 1e-12)
        gab = _divnorm(E.reshape(N, 16, h, w))
        b08, b16, b32 = _blur(L, 0.8), _blur(L, 1.6), _blur(L, 3.2)
        d1, d2 = b08 - b16, b16 - b32
        dog = _divnorm(torch.cat([F.relu(d1), F.relu(-d1), F.relu(d2), F.relu(-d2)], 1))
        rc = _conv(img[:, 1:3], c["coarse2"], groups=2).view(N, 2, 4, 2, h, w)
        Ec = torch.sqrt(rc[:, :, :, 0] ** 2 + rc[:, :, :, 1] ** 2 + 1e-12).reshape(N, 8, h, w)
        chrom = _divnorm(Ec)
        Eo = gab.view(N, 2, 8, h, w).sum(1)
        summ = torch.cat([Eo, dog[:, 0:1] + dog[:, 2:3], dog[:, 1:2] + dog[:, 3:4],
                          chrom[:, :4].sum(1, keepdim=True), chrom[:, 4:].sum(1, keepdim=True)], 1)
        return dict(e=e, o=o, Eo=Eo, d1=d1, Lb=b08, chrom=chrom, summ=summ,
                    stack=torch.cat([gab, dog, chrom], 1))

    # --------------------------------------------------------------- derived maps
    def _derived_maps(self, f, c):
        N = f["Eo"].shape[0]
        Eo, e, o = f["Eo"], f["e"], f["o"]
        s = Eo.sum(1, keepdim=True) + 1e-6
        harm = F.conv2d(Eo, c["harm"]) / s                                       # c2,s2,c4,s4
        gx1, gy1 = _grad(_blur(harm[:, 0:2], 1.0))
        curv = torch.sqrt((gx1 ** 2 + gy1 ** 2).sum(1, keepdim=True) + 1e-12)
        pop = F.relu(Eo - _blur(Eo, 3.0)).amax(1, keepdim=True)
        amp = torch.sqrt(e * e + o * o + 1e-12)
        epsA = 0.05 * amp.mean((1, 2, 3, 4), keepdim=True).squeeze(1) + 1e-6   # N,1,1,1
        es, os_ = e.sum(1), o.sum(1)
        pc = torch.sqrt(es ** 2 + os_ ** 2 + 1e-12).sum(1, keepdim=True) / (amp.sum((1, 2)).unsqueeze(1) + epsA)
        Ae = e.abs().sum((1, 2)).unsqueeze(1)
        Ao = o.abs().sum((1, 2)).unsqueeze(1)
        pol1 = (Ae - Ao) / (Ae + Ao + epsA)
        pol2 = e[:, 0].sum(1, keepdim=True) / (e[:, 0].abs().sum(1, keepdim=True) + epsA)
        maps = torch.cat([harm, curv, pop, pc, pol1, pol2], 1)
        # second-order (filter-rectify-filter) modulation
        Td = F.avg_pool2d(s, 2)
        hh, ww = Td.shape[-2:]
        r2 = _conv(Td, c["coarse"]).view(N, 4, 2, hh, ww)
        E2 = torch.sqrt(r2[:, :, 0] ** 2 + r2[:, :, 1] ** 2 + 1e-12)
        E2 = torch.log1p(E2 / (Td.mean((1, 2, 3), keepdim=True) + 1e-3))
        return _mean_grid(maps, 4), _mean_grid(E2, 3)

    # --------------------------------------------------------------- surface geometry
    def _geometry(self, img, c):
        S = _blur(img, 1.0)
        Ix, Iy, Ixx, Iyy, Ixy = _derivs(S)
        lx, ly, lxx, lyy, lxy = Ix[:, :1], Iy[:, :1], Ixx[:, :1], Iyy[:, :1], Ixy[:, :1]
        # Hessian shape index & curvedness
        t = 0.5 * (lxx + lyy)
        dd = torch.sqrt((0.5 * (lxx - lyy)) ** 2 + lxy ** 2 + 1e-12)
        l1, l2 = t + dd, t - dd
        si = torch.atan2(l1 + l2, l1 - l2) * (2.0 / math.pi)
        Cv = torch.sqrt(0.5 * (l1 ** 2 + l2 ** 2) + 1e-12)
        Cn = Cv / (Cv + Cv.mean((2, 3), keepdim=True) + 1e-5)
        shp = F.relu(1.0 - (si - c["si_centers"]).abs() / 0.25) * Cn          # 8 maps
        # structure tensor
        J = _blur(torch.cat([lx * lx, lx * ly, ly * ly], 1), 1.5)
        a, b, cc = J[:, 0:1], J[:, 1:2], J[:, 2:3]
        tr = a + cc
        dt = torch.sqrt((0.5 * (a - cc)) ** 2 + b ** 2 + 1e-16)
        mu1, mu2 = 0.5 * tr + dt, F.relu(0.5 * tr - dt)
        ref = tr.mean((2, 3), keepdim=True) + 1e-8
        coh = ((mu1 - mu2) / (mu1 + mu2 + 0.01 * ref)) ** 2
        st = torch.cat([torch.log1p(mu1 / ref), torch.log1p(mu2 / ref), coh], 1)
        # isophote curvature
        g2 = lx * lx + ly * ly
        g = torch.sqrt(g2 + 1e-12)
        gm = g.mean((2, 3), keepdim=True) + 1e-5
        kap = -(ly * ly * lxx - 2 * lx * ly * lxy + lx * lx * lyy) / (g2 + (0.1 * gm) ** 2) ** 1.5
        wk = g / (g + gm)
        tk = torch.tanh(kap)
        iso = torch.cat([wk * tk, wk * tk.abs()], 1)
        # chromatic corner strength (Harris on colour structure tensor)
        corners = []
        for sl in (slice(0, 3), slice(1, 3)):
            X_, Y_ = Ix[:, sl], Iy[:, sl]
            Jc = _blur(torch.cat([(X_ * X_).sum(1, keepdim=True), (X_ * Y_).sum(1, keepdim=True),
                                  (Y_ * Y_).sum(1, keepdim=True)], 1), 1.5)
            trc = Jc[:, 0:1] + Jc[:, 2:3]
            Hc = Jc[:, 0:1] * Jc[:, 2:3] - Jc[:, 1:2] ** 2 - 0.04 * trc ** 2
            ref2 = trc.mean((2, 3), keepdim=True) ** 2 + 1e-12
            corners.append(torch.sign(Hc) * torch.log1p(Hc.abs() / ref2))
        return torch.cat([shp, st, iso] + corners, 1)                           # 15 maps

    # --------------------------------------------------------------- extras
    @staticmethod
    def _extrema(x):
        p = F.pad(x, (1, 1, 1, 1), mode="replicate")
        mx = F.max_pool2d(p, 3, 1)
        mn = -F.max_pool2d(-p, 3, 1)
        loc = F.avg_pool2d(p, 3, 1)
        thr = 0.02 * x.flatten(2).std(2).view(x.shape[0], x.shape[1], 1, 1) + 1e-6
        ismax = ((x >= mx) & (x > loc + thr)).float()
        ismin = ((x <= mn) & (x < loc - thr)).float()
        return torch.cat([ismax, ismin], 1)

    def _paths(self, f, c):
        T = f["Eo"].sum(1, keepdim=True)
        Cc = f["chrom"].sum(1, keepdim=True)
        ca = 0.05 + T / (T.mean((2, 3), keepdim=True) + 1e-3)
        ch = 0.05 + Cc / (Cc.mean((2, 3), keepdim=True) + 0.05)
        cost = torch.cat([ca.expand(-1, 5, -1, -1), ch.expand(-1, 5, -1, -1)], 1)
        d = torch.where(c["anchor_mask"], torch.zeros_like(cost), torch.full_like(cost, 1e4))
        for _ in range(48):  # geodesic distance transform (min-plus propagation)
            d = torch.minimum(d, -F.max_pool2d(-d, 3, 1, 1) + cost)
        d = torch.log1p(d / (32.0 * cost.mean((2, 3), keepdim=True)))
        return _mean_grid(d, 4)

    # --------------------------------------------------------------- forward
    def _features(self, X, c):
        N = X.shape[0]
        rgb = X.reshape(N, IMG, IMG, 3).permute(0, 3, 1, 2).clamp(0, 1)
        rgb = (rgb + 0.04) ** 0.45
        R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        img = torch.cat([(R + G + B) / math.sqrt(3), (R - G) / math.sqrt(2),
                         (R + G - 2 * B) / math.sqrt(6)], 1)
        cort = F.grid_sample(img, c["warp"].expand(N, -1, -1, -1).contiguous(), mode="bilinear",
                             padding_mode="border", align_corners=True)

        blocks = []
        add = lambda v, wt: blocks.append(wt * _l2(v))

        # retinal & cortical pyramids
        ret_imgs = [img, F.avg_pool2d(img, 2), F.avg_pool2d(img, 4)]
        cor_imgs = [cort, F.avg_pool2d(cort, 2), F.avg_pool2d(cort, 4)]
        ret = [self._octave(x, c) for x in ret_imgs]

        # A: cortical means (9, 6, 4)
        for x, g, wt in zip(cor_imgs, (9, 6, 4), (1.0, 0.9, 0.8)):
            add(_mean_grid(self._octave(x, c)["stack"], g), wt)
        # B: retinal std + peaks
        for f, g in zip(ret, (3, 2, 2)):
            add(_std_grid(f["stack"], g), 0.5)
            add(_max_grid(f["stack"], g), 0.5)

        # derived maps
        for f in ret:
            m, so = self._derived_maps(f, c)
            add(m, 0.5)
            add(so, 0.35)
        iu = c["iu12"]
        s0 = ret[0]["summ"]
        add(_corr_offdiag(s0.flatten(2), iu), 0.5)
        h2 = s0.shape[-1] // 2
        quads = [s0[..., :h2, :h2], s0[..., :h2, h2:], s0[..., h2:, :h2], s0[..., h2:, h2:]]
        add(torch.cat([_corr_offdiag(q.flatten(2), iu) for q in quads], 1), 0.5)
        add(torch.cat([_corr_offdiag(ret[k]["summ"].flatten(2), iu) for k in (1, 2)], 1), 0.45)
        cross = []
        for k in range(2):
            a = ret[k]["Eo"]
            b = F.interpolate(ret[k + 1]["Eo"], size=a.shape[-2:], mode="bilinear", align_corners=False)
            cross += [_mean_grid(torch.sqrt(F.relu(a) * F.relu(b) + 1e-12), 2), _mean_grid(a - b, 2)]
        add(torch.cat(cross, 1), 0.4)
        fov = []
        for f in ret:
            st = f["stack"]
            h = st.shape[-1]
            mf, mp = c[f"fov{h}"], c[f"per{h}"]
            mc = (st * mf).sum((2, 3)) / mf.sum()
            mpp = (st * mp).sum((2, 3)) / mp.sum()
            fov += [mc - mpp, torch.log((mc + 1e-3) / (mpp + 1e-3))]
        add(torch.cat(fov, 1), 0.4)
        mir = []
        for f in ret:
            s_ = f["summ"]
            mir += [_pcorr(s_, s_.flip(-1)), _pcorr(s_, s_.flip(-2))]
        add(torch.cat(mir, 1), 0.35)

        # surface geometry over five octaves (most heavily weighted)
        geo_imgs = [img]
        for _ in range(4):
            geo_imgs.append(F.avg_pool2d(geo_imgs[-1], 2))
        geo = [self._geometry(x, c) for x in geo_imgs]
        for gm, g, wt in zip(geo, (10, 7, 5, 3, 2), (1.5, 1.2, 0.9, 0.6, 0.35)):
            add(_mean_grid(gm, g), wt)
        gc = []
        for k in range(4):
            up = F.interpolate(geo[k + 1], size=geo[k].shape[-2:], mode="bilinear", align_corners=False)
            gc.append(_pcorr(geo[k], up))
        add(torch.cat(gc, 1), 0.6)

        # extras
        add(torch.cat([_mean_grid(img, 6), _std_grid(img, 3)], 1), 0.5)
        add(self._paths(ret[2], c), 0.4)
        add(torch.cat([_mean_grid(torch.cat([self._extrema(f["Lb"]), self._extrema(f["d1"])], 1), 4)
                       for f in ret], 1), 0.4)

        out = torch.cat(blocks, 1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X)
        if X.dim() != 2:
            X = X.reshape(X.shape[0], -1)
        X = X.float()
        c = self._consts(X.device)
        outs = [self._features(X[i:i + CHUNK], c) for i in range(0, X.shape[0], CHUNK)]
        return torch.cat(outs, 0)


MODEL = _FixedVisualModel()
