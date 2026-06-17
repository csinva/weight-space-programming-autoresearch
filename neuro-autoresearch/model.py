"""CAPACITY-SCALED VENTRAL CASCADE @128px (evo-gO3-c5): wide random-feature
expansion on top of the proven gating+competition Carandini-Heeger cascade.

The front-end is the elite01 backbone (0.3201 ventral): color-opponent LGN ->
V1 Gabor energy -> random-orthonormal V2/V4/IT convs, each deep stage with
per-image cross-channel sigmoid gating + soft channel competition + divisive
norm. The eccentricity windows + windowed grams + offset co-occurrence are
KEPT (they carry the proven signal).

THE NEW LEVER (capacity scaling): at the deep stages (V2/V4/IT) we add a WIDE
RANDOM-FEATURE EXPANSION. Each location's channel vector is projected through a
much wider fixed-random matrix and passed through ReLU (an arc-cosine kernel /
random-ReLU expansion that approximates a powerful nonlinear kernel), then
spatial-pyramid pooled. This dramatically enlarges the untrained feature space
to test whether a richer random nonlinear basis generalizes better on the
held-out ventral ROI. The per-target ridge regularizes; we report the
train-test gap to see whether the extra capacity overfits or genuinely helps.

ANTI-HACKING: every filter is ANALYTIC (Gabor/DoG) or random-but-FIXED (fixed
seeds, QR-orthonormalized or unit-normed). Nothing is trained, Y is never seen.
Every normalization (divisive norm, eccentricity windows, random-feature scale,
final row-norm) is strictly PER-SAMPLE / data-independent -- no statistic
crosses samples, so no test structure can leak.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Analytic front-end filters
# ---------------------------------------------------------------------------
def _dog_bank(ksize, sigma_c, ratio, device, dtype):
    half = (ksize - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(ksize, device=device, dtype=dtype) - half,
        torch.arange(ksize, device=device, dtype=dtype) - half,
        indexing="ij",
    )
    r2 = xs ** 2 + ys ** 2
    sc, ss = sigma_c, sigma_c * ratio
    center = torch.exp(-r2 / (2 * sc * sc)); center = center / center.sum()
    surround = torch.exp(-r2 / (2 * ss * ss)); surround = surround / surround.sum()
    return (center - surround).view(1, 1, ksize, ksize)


def _gabor_quad(ksize, wavelength, n_orient, device, dtype):
    filters = []
    sigma = 0.5 * wavelength
    half = (ksize - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(ksize, device=device, dtype=dtype) - half,
        torch.arange(ksize, device=device, dtype=dtype) - half,
        indexing="ij",
    )
    for o in range(n_orient):
        theta = o * math.pi / n_orient
        xr = xs * math.cos(theta) + ys * math.sin(theta)
        env = torch.exp(-(xs ** 2 + ys ** 2) / (2.0 * sigma ** 2))
        phase = 2.0 * math.pi * xr / wavelength
        for func in (torch.cos, torch.sin):
            f = env * func(phase); f = f - f.mean()
            n = f.abs().sum()
            if n > 0:
                f = f / n
            filters.append(f)
    return torch.stack(filters).unsqueeze(1)


def _ortho_filters(out_ch, in_ch, k, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    fan = in_ch * k * k
    W = torch.randn(out_ch, fan, generator=g, dtype=torch.float32)
    if out_ch <= fan:
        q, _ = torch.linalg.qr(W.t())
        W = q.t().contiguous()
    else:
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return W.view(out_ch, in_ch, k, k).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Canonical computation pieces
# ---------------------------------------------------------------------------
def _divnorm(x, k, sigma):
    energy = (x * x).mean(dim=1, keepdim=True)
    local = F.avg_pool2d(energy, kernel_size=k, stride=1, padding=k // 2)
    return x / torch.sqrt(local + sigma * sigma + 1e-6)


def _halfwave(y):
    return torch.cat([F.relu(y), F.relu(-y)], dim=1)


def _spp(x, levels):
    N = x.shape[0]
    return torch.cat([F.adaptive_avg_pool2d(x, l).reshape(N, -1) for l in levels], 1)


# ---------------------------------------------------------------------------
# NOVEL INTERACTIONS (kept from elite01)
# ---------------------------------------------------------------------------
def _cross_channel_gate(x, P, b):
    N, C, H, W = x.shape
    flat = x.reshape(N, C, H * W)
    logits = torch.einsum("oc,ncp->nop", P, flat)
    g = torch.sigmoid(logits + b.view(1, C, 1))
    return (flat * g).reshape(N, C, H, W)


def _soft_competition(x, beta):
    N, C, H, W = x.shape
    w = torch.softmax(beta * x.reshape(N, C, H * W), dim=1).reshape(N, C, H, W)
    w.mul_(C).add_(1.0)
    return x * w


# ---------------------------------------------------------------------------
# WIDE RANDOM-FEATURE EXPANSION -- the capacity-scaling lever.
# Project each location's channel vector through a wide fixed-random matrix and
# apply ReLU (arc-cosine / random-ReLU kernel features), then spatial-pyramid
# pool. Per-image: the expansion is a fixed data-independent random map.
# ---------------------------------------------------------------------------
def _random_relu_expand(x, W, levels):
    """x: (N,C,H,W). W: (M, C) wide random projection (M >> C). Returns the
    spatial-pyramid-pooled ReLU random features -> (N, M*sum(l^2))."""
    N, C, H, Wd = x.shape
    flat = x.reshape(N, C, H * Wd)                      # (N,C,P)
    z = torch.einsum("mc,ncp->nmp", W, flat)            # (N,M,P) wide projection
    z = F.relu(z)                                       # arc-cosine (order 1) feature
    z = z.reshape(N, W.shape[0], H, Wd)
    return _spp(z, levels)


def _rff_expand(x, W, b, levels):
    """RBF random-Fourier features: cos(W x + b) with W ~ N(0, gamma^2 I), b ~
    U(0,2pi) -> unbiased estimate of an RBF kernel (a different, often stronger
    kernel than arc-cosine). x: (N,C,H,W). W: (M,C). b: (M,). Per-image; the
    projection is a fixed data-independent random map. Spatial-pyramid pooled.
    A per-location L2 normalization makes the RBF scale data-independent per
    sample (no cross-sample statistic)."""
    N, C, H, Wd = x.shape
    flat = x.reshape(N, C, H * Wd)                      # (N,C,P)
    nrm = flat.norm(dim=1, keepdim=True) + 1e-6        # per-location norm (per-image)
    flat = flat / nrm
    z = torch.einsum("mc,ncp->nmp", W, flat) + b.view(1, -1, 1)
    z = torch.cos(z) * math.sqrt(2.0 / W.shape[0])     # RFF scaling
    z = z.reshape(N, W.shape[0], H, Wd)
    return _spp(z, levels)


# ---------------------------------------------------------------------------
# ECCENTRICITY WINDOWS
# ---------------------------------------------------------------------------
def _ecc_windows(H, W, device, dtype):
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(gx ** 2 + gy ** 2)
    fov = torch.exp(-(r ** 2) / (2.0 * 0.35 ** 2))
    mid = torch.exp(-((r - 0.65) ** 2) / (2.0 * 0.28 ** 2))
    per = 1.0 - torch.exp(-(r ** 2) / (2.0 * 0.6 ** 2))
    fov = fov / (fov.sum() + 1e-8)
    mid = mid / (mid.sum() + 1e-8)
    per = per / (per.sum() + 1e-8)
    return fov.view(1, 1, H, W), mid.view(1, 1, H, W), per.view(1, 1, H, W)


def _windowed_pool(x, win, grid):
    N, C, H, W = x.shape
    wx = win * x
    num = F.adaptive_avg_pool2d(wx, grid)
    den = F.adaptive_avg_pool2d(win.expand(N, 1, H, W), grid)
    pooled = num / (den + 1e-8)
    return pooled.reshape(N, -1)


class CapacityScaledCascade:
    def __init__(self):
        self._dev = None
        self._n_orient = 12
        self._chans = (96, 128, 160)
        self._ksize = (3, 5, 5)
        self._seeds = (211, 322, 433)
        self._dn_k = (5, 5, 5, 7, 7)
        self._dn_sigma = (0.3, 0.5, 0.5, 0.5, 0.5)
        self._gram_k = 48
        self._cooc_k = 22
        self._cooc_offsets = [(2, 0), (0, 2), (2, 2)]
        self._soft_beta = 2.0
        self._fov_grid = {"lgn": 3, "v1": 3, "v2": 3, "v4": 2, "it": 2}
        self._mid_grid = {"lgn": 2, "v1": 2, "v2": 2, "v4": 2, "it": 1}
        self._per_grid = {"lgn": 2, "v1": 2, "v2": 2, "v4": 1, "it": 1}
        self._spp_levels = {"lgn": [2, 3], "v1": [2, 3], "v2": [1, 2],
                            "v4": [1, 2], "it": [1, 2]}
        # WIDE RANDOM-FEATURE expansion config (capacity scaling).
        # M = expansion width per stage; pooled at these spatial-pyramid levels.
        self._rf_enabled = False   # arc-cosine random-ReLU expansion (tested: no gain)
        self._rf_width = {"v2": 384, "v4": 512, "it": 768}
        self._rf_levels = {"v2": [2], "v4": [2], "it": [2]}
        self._rf_seed = {"v2": 9011, "v4": 9013, "it": 9017}
        # RBF random-Fourier expansion (capacity lever v2: a different kernel).
        self._rff_enabled = True
        self._rff_width = {"v2": 256, "v4": 384, "it": 640}
        self._rff_levels = {"v2": [2], "v4": [2], "it": [2]}
        self._rff_gamma = {"v2": 4.0, "v4": 4.0, "it": 4.0}
        self._rff_seed = {"v2": 8011, "v4": 8013, "it": 8017}

    def _build(self, device, dtype):
        no = self._n_orient
        self._dog = [_dog_bank(7, 1.0, 2.0, device, dtype),
                     _dog_bank(13, 2.0, 2.0, device, dtype)]
        self._gabor = [_gabor_quad(7, 4.0, no, device, dtype),
                       _gabor_quad(13, 8.0, no, device, dtype)]
        v1_ch = 3 * 2 * no
        Ws = []; prev = v1_ch
        for oc, k, sd in zip(self._chans, self._ksize, self._seeds):
            Ws.append(_ortho_filters(oc, prev, k, sd, device, dtype))
            prev = 2 * oc
        self._W = Ws
        self._gate = {}
        for nm, oc, sd in zip(["v2", "v4", "it"], self._chans, (1301, 1303, 1307)):
            C = 2 * oc
            gg = torch.Generator(device="cpu").manual_seed(sd)
            P = torch.randn(C, C, generator=gg, dtype=torch.float32)
            P = P / (P.norm(dim=1, keepdim=True) + 1e-8) * math.sqrt(C) * 0.5
            b = torch.zeros(C, dtype=torch.float32)
            self._gate[nm] = (P.to(device=device, dtype=dtype),
                              b.to(device=device, dtype=dtype))
        self._gram_proj = {}
        self._cooc_proj = {}
        for nm, ch, sd in zip(["v2", "v4", "it"], self._chans, (701, 709, 717)):
            hi_ch = 2 * ch
            gp = torch.Generator(device="cpu").manual_seed(sd)
            P = torch.randn(hi_ch, self._gram_k, generator=gp, dtype=torch.float32)
            P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
            self._gram_proj[nm] = P.to(device=device, dtype=dtype)
            cp = torch.Generator(device="cpu").manual_seed(sd + 5000)
            Pc = torch.randn(hi_ch, self._cooc_k, generator=cp, dtype=torch.float32)
            Pc = Pc / (Pc.norm(dim=0, keepdim=True) + 1e-8)
            self._cooc_proj[nm] = Pc.to(device=device, dtype=dtype)
        # WIDE RANDOM-FEATURE projection matrices (M, C) per deep stage.
        self._rf_proj = {}
        for nm, ch in zip(["v2", "v4", "it"], self._chans):
            C = 2 * ch
            M = self._rf_width[nm]
            rg = torch.Generator(device="cpu").manual_seed(self._rf_seed[nm])
            Wr = torch.randn(M, C, generator=rg, dtype=torch.float32)
            Wr = Wr / (Wr.norm(dim=1, keepdim=True) + 1e-8) * math.sqrt(C) * 0.5
            self._rf_proj[nm] = Wr.to(device=device, dtype=dtype)
        # RBF RANDOM-FOURIER projection matrices + phase biases per deep stage.
        self._rff_W = {}
        self._rff_b = {}
        for nm, ch in zip(["v2", "v4", "it"], self._chans):
            C = 2 * ch
            M = self._rff_width[nm]
            gamma = self._rff_gamma[nm]
            wg = torch.Generator(device="cpu").manual_seed(self._rff_seed[nm])
            Wr = torch.randn(M, C, generator=wg, dtype=torch.float32) * gamma
            bg = torch.Generator(device="cpu").manual_seed(self._rff_seed[nm] + 1)
            bb = torch.rand(M, generator=bg, dtype=torch.float32) * (2 * math.pi)
            self._rff_W[nm] = Wr.to(device=device, dtype=dtype)
            self._rff_b[nm] = bb.to(device=device, dtype=dtype)
        self._win_cache = {}
        self._dev = device

    def _get_windows(self, H, W, device, dtype):
        key = (H, W)
        if key not in self._win_cache:
            self._win_cache[key] = _ecc_windows(H, W, device, dtype)
        return self._win_cache[key]

    def _gram_windowed(self, h, nm, win, device):
        Nb = h.shape[0]
        flat = (win * h).reshape(Nb, h.shape[1], -1)
        z = torch.einsum("ncp,ck->nkp", flat, self._gram_proj[nm])
        z = z - z.mean(dim=-1, keepdim=True)
        gram = torch.matmul(z, z.transpose(1, 2)) / z.shape[-1]
        K = self._gram_k
        iu = torch.triu_indices(K, K, offset=0, device=device)
        gram = gram[:, iu[0], iu[1]]
        return torch.sign(gram) * torch.sqrt(gram.abs() + 1e-6)

    def _offset_cooccur(self, h, nm, device):
        Nb, C, H, W = h.shape
        flat = h.reshape(Nb, C, H * W)
        z = torch.einsum("ncp,ck->nkp", flat, self._cooc_proj[nm])
        K = z.shape[1]
        z = (z - z.mean(dim=-1, keepdim=True)).reshape(Nb, K, H, W)
        feats = []
        for dy, dx in self._cooc_offsets:
            zs = torch.roll(z, shifts=(dy, dx), dims=(2, 3))
            cc = torch.einsum("nkhw,nlhw->nkl", z, zs) / (H * W)
            cc = cc.reshape(Nb, K * K)
            feats.append(torch.sign(cc) * torch.sqrt(cc.abs() + 1e-6))
        return torch.cat(feats, dim=1)

    def _lgn(self, channels):
        outs = []
        for ch in range(channels.shape[1]):
            cmap = channels[:, ch:ch + 1]
            for dog in self._dog:
                r = F.conv2d(cmap, dog, padding="same")
                outs.append(r)
        lgn = torch.cat(outs, dim=1)
        return _halfwave(lgn)

    def _v1(self, channels):
        N = channels.shape[0]
        no = self._n_orient
        H, W = channels.shape[-2:]
        mags = []
        for ch in range(channels.shape[1]):
            cmap = channels[:, ch:ch + 1]
            for bank in self._gabor:
                resp = F.conv2d(cmap, bank, padding="same")
                resp = resp.view(N, no, 2, H, W)
                mag = torch.sqrt(resp[:, :, 0] ** 2 + resp[:, :, 1] ** 2 + 1e-6)
                mags.append(mag)
        v1 = torch.cat(mags, dim=1)
        v1 = _divnorm(v1, self._dn_k[1], self._dn_sigma[1])
        return v1

    def _deep_stage(self, x, W, nm, dn_k, dn_sigma):
        k = W.shape[-1]
        y = F.conv2d(x, W, padding=k // 2)
        y = _halfwave(y)
        P, b = self._gate[nm]
        y = _cross_channel_gate(y, P, b)
        y = _soft_competition(y, self._soft_beta)
        y = _divnorm(y, dn_k, dn_sigma)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)
        return y

    def _readout(self, x, nm, device):
        H, W = x.shape[-2:]
        fov, mid, per = self._get_windows(H, W, x.device, x.dtype)
        feats = []
        feats.append(torch.sqrt(_spp(x, self._spp_levels[nm]).abs() + 1e-6))
        ff = _windowed_pool(x, fov, self._fov_grid[nm])
        feats.append(torch.sqrt(ff.abs() + 1e-6))
        fm = _windowed_pool(x, mid, self._mid_grid[nm])
        feats.append(torch.sqrt(fm.abs() + 1e-6))
        fp = _windowed_pool(x, per, self._per_grid[nm])
        feats.append(torch.sqrt(fp.abs() + 1e-6))
        if nm in self._gram_proj:
            feats.append(self._gram_windowed(x, nm, per, device))
            feats.append(self._gram_windowed(x, nm, fov, device))
        if nm in ("v4", "it"):
            feats.append(self._offset_cooccur(x, nm, device))
        # CAPACITY SCALING: wide random-ReLU feature expansion at deep stages.
        if self._rf_enabled and nm in self._rf_proj:
            rf = _random_relu_expand(x, self._rf_proj[nm], self._rf_levels[nm])
            feats.append(torch.sqrt(rf.abs() + 1e-6))
        # CAPACITY SCALING v2: RBF random-Fourier expansion (different kernel).
        if self._rff_enabled and nm in self._rff_W:
            rff = _rff_expand(x, self._rff_W[nm], self._rff_b[nm],
                              self._rff_levels[nm])
            feats.append(rff)
        return torch.cat(feats, dim=1)

    def __call__(self, channels):
        device, dtype = channels.device, channels.dtype
        if self._dev != device:
            self._build(device, dtype)

        lgn = self._lgn(channels)
        lgn = _divnorm(lgn, self._dn_k[0], self._dn_sigma[0])
        lgn_p = F.avg_pool2d(lgn, kernel_size=2, stride=2)

        v1 = self._v1(channels)
        h = F.avg_pool2d(v1, kernel_size=2, stride=2)

        stage_maps = {"lgn": lgn_p, "v1": h}
        names = ["v2", "v4", "it"]
        for W, nm, dk, ds in zip(self._W, names,
                                 self._dn_k[2:], self._dn_sigma[2:]):
            h = self._deep_stage(h, W, nm, dk, ds)
            stage_maps[nm] = h

        feats = []
        for nm in ["lgn", "v1", "v2", "v4", "it"]:
            feats.append(self._readout(stage_maps[nm], nm, device))

        # FOVEAL MAGNIFICATION branch (single magnified-center cascade; the other
        # redundant crop branches are dropped to keep the forward fast + D small
        # so the f64 ridge readout fits under heavy shared-GPU contention).
        H0, W0 = channels.shape[-2:]
        cy0, cy1 = H0 // 3, 2 * H0 // 3
        cx0, cx1 = W0 // 3, 2 * W0 // 3
        crop = channels[:, :, cy0:cy1, cx0:cx1]
        crop = F.interpolate(crop, size=(H0, W0), mode="bilinear",
                             align_corners=False)
        fov_v1 = self._v1(crop)
        fh = F.avg_pool2d(fov_v1, kernel_size=2, stride=2)
        feats.append(torch.sqrt(_spp(fh, [2, 4]).abs() + 1e-6))
        for W, nm, dk, ds in zip(self._W, names, self._dn_k[2:], self._dn_sigma[2:]):
            fh = self._deep_stage(fh, W, nm, dk, ds)
        feats.append(torch.sqrt(_spp(fh, [1, 2]).abs() + 1e-6))
        feats.append(self._gram_windowed(fh, "it",
                     self._get_windows(fh.shape[-2], fh.shape[-1],
                                       device, dtype)[0], device))
        # CAPACITY-SCALING expansion also on the foveal-magnified IT map.
        if self._rf_enabled:
            feats.append(torch.sqrt(_random_relu_expand(
                fh, self._rf_proj["it"], self._rf_levels["it"]).abs() + 1e-6))
        if self._rff_enabled:
            feats.append(_rff_expand(fh, self._rff_W["it"], self._rff_b["it"],
                                     self._rff_levels["it"]))

        # TIGHT FOVEA (restored: full elite01 branch)
        ty0, ty1 = (2 * H0) // 5, (3 * H0) // 5
        tx0, tx1 = (2 * W0) // 5, (3 * W0) // 5
        tcrop = channels[:, :, ty0:ty1, tx0:tx1]
        tcrop = F.interpolate(tcrop, size=(H0, W0), mode="bilinear",
                              align_corners=False)
        th = F.avg_pool2d(self._v1(tcrop), kernel_size=2, stride=2)
        for W, nm, dk, ds in zip(self._W, names, self._dn_k[2:], self._dn_sigma[2:]):
            th = self._deep_stage(th, W, nm, dk, ds)
        feats.append(torch.sqrt(_spp(th, [1, 2]).abs() + 1e-6))

        # PERIPHERAL LOW-SF GLOBAL branch (restored: full elite01 branch)
        blur = F.avg_pool2d(channels, kernel_size=4, stride=4)
        blur = F.interpolate(blur, size=(H0, W0), mode="bilinear",
                             align_corners=False)
        ph = F.avg_pool2d(self._v1(blur), kernel_size=2, stride=2)
        for W, nm, dk, ds in zip(self._W, names, self._dn_k[2:], self._dn_sigma[2:]):
            ph = self._deep_stage(ph, W, nm, dk, ds)
        Hp, Wp = ph.shape[-2:]
        _, _, per = self._get_windows(Hp, Wp, device, dtype)
        feats.append(torch.sqrt(_windowed_pool(ph, per, 1).abs() + 1e-6))
        feats.append(self._gram_windowed(ph, "it", per, device))
        return torch.cat(feats, dim=1)


class Model:
    name = "Full elite01 + RBF random-Fourier capacity expansion @128px"
    idea = (
        "Take the proven gating+competition+co-occurrence cascade (0.3201 ventral) "
        "and DRAMATICALLY scale its untrained feature capacity by adding a WIDE "
        "random-ReLU (arc-cosine kernel) expansion at the deep V2/V4/IT stages: "
        "project each location's channel vector through a much wider fixed-random "
        "matrix, ReLU, and spatial-pyramid pool. Tests whether a far richer "
        "untrained nonlinear basis generalizes better on held-out ventral or "
        "merely overfits the readout (watch the train-test gap)."
    )
    summary = (
        "Color-opponent LGN->V1 Gabor energy->random-orthonormal V2/V4/IT cascade "
        "with per-image cross-channel gating + soft competition + divisive norm. "
        "Read out through eccentricity windows + windowed grams + offset "
        "co-occurrence (the proven elite01 features) PLUS a wide random-ReLU "
        "feature expansion at each deep stage (fixed random MxC projection, ReLU, "
        "SPP-pooled) that multiplies the feature dimensionality. All "
        "per-image/analytic/fixed-random; never trained, Y unseen."
    )
    n_params = 0

    def __init__(self):
        self._cascade = CapacityScaledCascade()

    def _to_channels(self, X):
        N = X.shape[0]
        side = int(round(math.sqrt(X.shape[1] / 3.0)))
        img = X.view(N, side, side, 3).permute(0, 3, 1, 2)
        R, G, B = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        lum = (R + G + B) / 3.0
        rg = R - G
        by = B - (R + G) / 2.0
        return torch.cat([lum, rg, by], dim=1)

    def _forward_chunk(self, X):
        channels = self._to_channels(X)
        return self._cascade(channels)

    @torch.no_grad()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        bs = 32
        N = X.shape[0]
        parts = []
        for i in range(0, N, bs):
            out_i = self._forward_chunk(X[i:i + bs])
            out_i = out_i / (out_i.norm(dim=1, keepdim=True) + 1e-6)
            parts.append(out_i.to(torch.float32).cpu())
            del out_i
        out = torch.cat(parts, dim=0)
        return out


MODEL = Model()
