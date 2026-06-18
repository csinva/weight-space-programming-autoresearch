# idea: champion UNCHANGED; readout-granularity tune -- proven SPP [1,2,3,5] kept exactly + global-form re-shaped to 7 rings x 18 sectors (fewer ecc rings, denser angular sectors -- untried aspect at the band edge)
"""FIVE-STREAM fusion on the per-stream-normalized richer-PS backbone.

EXPLOIT of the 0.3205 champion (m_9da520b6df). Three DISTINCT cross-stream
conjunction families have each independently lifted ventral score, but only in
isolation: (i) texture-energy x shape/blob/global-form (best), (ii) cross-
orientation junction (energy x orthogonal-energy = corners/crosses/junctions),
(iii) color-opponent luminance x chroma. Each captures a different conjunctive
selectivity ventral cortex is famous for, and the ridge readout cannot recover a
product from separate additive streams. The clearest untried move is to put ALL
THREE in ONE fifth stream, all under the same tuned three-radius surround
divisive-normalization, so the readout sees texture-on-figure, junction/corner,
AND color-opponent conjunctions together. Built on the per-stream-RMS-normalized
8-orient backbone. All analytic / fixed kernels, strictly per-image, Y never
read, no statistic crosses the batch.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _steer_bank(ksize, wavelength, n_orient, device, dtype):
    half = (ksize - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(ksize, device=device, dtype=dtype) - half,
        torch.arange(ksize, device=device, dtype=dtype) - half,
        indexing="ij",
    )
    sigma = 0.5 * wavelength
    env = torch.exp(-(xs ** 2 + ys ** 2) / (2.0 * sigma ** 2))
    reals, imags = [], []
    for o in range(n_orient):
        theta = o * math.pi / n_orient
        xr = xs * math.cos(theta) + ys * math.sin(theta)
        phase = 2.0 * math.pi * xr / wavelength
        c = env * torch.cos(phase); c = c - c.mean()
        s = env * torch.sin(phase); s = s - s.mean()
        nrm = torch.sqrt((c ** 2 + s ** 2).sum()) + 1e-8
        reals.append(c / nrm); imags.append(s / nrm)
    return torch.stack(reals).unsqueeze(1), torch.stack(imags).unsqueeze(1)


def _dog_kernel(ksize, sigma_c, surround_ratio, device, dtype):
    half = (ksize - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(ksize, device=device, dtype=dtype) - half,
        torch.arange(ksize, device=device, dtype=dtype) - half,
        indexing="ij",
    )
    r2 = xs ** 2 + ys ** 2
    sc = sigma_c
    ss = sigma_c * surround_ratio
    center = torch.exp(-r2 / (2.0 * sc * sc)); center = center / center.sum()
    surround = torch.exp(-r2 / (2.0 * ss * ss)); surround = surround / surround.sum()
    k = center - surround
    k = k - k.mean()
    k = k / (k.abs().sum() + 1e-8)
    return k.unsqueeze(0).unsqueeze(0)


def _logpolar_grid(out_h, out_w, in_h, in_w, r_min, r_max, device, dtype):
    logr = torch.linspace(math.log(r_min), math.log(r_max), out_h,
                          device=device, dtype=dtype)
    rad = torch.exp(logr)
    ang = torch.linspace(0.0, 2.0 * math.pi, out_w + 1,
                         device=device, dtype=dtype)[:-1]
    R, A = torch.meshgrid(rad, ang, indexing="ij")
    cy = (in_h - 1) / 2.0
    cx = (in_w - 1) / 2.0
    half = min(in_h, in_w) / 2.0
    px = cx + R * half * torch.cos(A)
    py = cy + R * half * torch.sin(A)
    gx = px / (in_w - 1) * 2.0 - 1.0
    gy = py / (in_h - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def _stab(x):
    return torch.sign(x) * torch.sqrt(x.abs() + 1e-6)


def _spp(x, levels):
    N = x.shape[0]
    return torch.cat([F.adaptive_avg_pool2d(x, l).reshape(N, -1) for l in levels], 1)


def _divnorm(x, k, sigma):
    energy = (x * x).mean(dim=1, keepdim=True)
    local = F.avg_pool2d(energy, kernel_size=k, stride=1, padding=k // 2)
    return x / torch.sqrt(local + sigma * sigma + 1e-6)


def _rms_norm(feat):
    rms = torch.sqrt((feat * feat).mean(dim=1, keepdim=True) + 1e-6)
    return feat / rms


def _curvature(mod, O):
    angles = torch.arange(O, device=mod.device, dtype=mod.dtype) * (math.pi / O)
    cos2 = torch.cos(2 * angles).view(1, O, 1, 1)
    sin2 = torch.sin(2 * angles).view(1, O, 1, 1)
    mx = (mod * cos2).sum(1, keepdim=True)
    my = (mod * sin2).sum(1, keepdim=True)
    coh = torch.sqrt(mx * mx + my * my + 1e-6)
    total = mod.sum(1, keepdim=True) + 1e-6
    straight = coh / total
    gx_x = F.pad(mx[:, :, :, 2:] - mx[:, :, :, :-2], (1, 1, 0, 0))
    gx_y = F.pad(my[:, :, :, 2:] - my[:, :, :, :-2], (1, 1, 0, 0))
    gy_x = F.pad(mx[:, :, 2:, :] - mx[:, :, :-2, :], (0, 0, 1, 1))
    gy_y = F.pad(my[:, :, 2:, :] - my[:, :, :-2, :], (0, 0, 1, 1))
    curv = torch.sqrt(gx_x ** 2 + gx_y ** 2 + gy_x ** 2 + gy_y ** 2 + 1e-6)
    sign = gx_x * gy_y - gx_y * gy_x
    curv_signed = torch.sign(sign) * torch.sqrt(curv.abs() + 1e-6)
    return torch.cat([curv, curv_signed, straight], dim=1)


def _end_stop(mod, O, shift=2):
    feats = []
    for o in range(O):
        theta = o * (math.pi / O)
        dx = int(round(shift * math.cos(theta)))
        dy = int(round(shift * math.sin(theta)))
        e = mod[:, o:o + 1]
        ahead = torch.roll(e, shifts=(dy, dx), dims=(2, 3))
        behind = torch.roll(e, shifts=(-dy, -dx), dims=(2, 3))
        feats.append(F.relu(e - torch.maximum(ahead, behind)))
    es = torch.cat(feats, dim=1)
    return torch.cat([es, es.sum(1, keepdim=True)], dim=1)


def _cocircular(mod, O, radius=3):
    N, _, H, W = mod.shape
    acc = torch.zeros(N, 1, H, W, device=mod.device, dtype=mod.dtype)
    for o in range(O):
        theta = o * (math.pi / O)
        dx = int(round(radius * math.cos(theta + math.pi / 2)))
        dy = int(round(radius * math.sin(theta + math.pi / 2)))
        e = mod[:, o:o + 1]
        acc = acc + e * torch.roll(e, shifts=(dy, dx), dims=(2, 3))
    return torch.sqrt(acc.abs() + 1e-6)


def _vech(cov):
    N, O, _ = cov.shape
    idx = torch.triu_indices(O, O, device=cov.device)
    return cov[:, idx[0], idx[1]]


def _closure(mod, O, radius=5, n_dir=12):
    """Proto-object boundary-completion cue. For each location, integrate
    co-circular edge support around a FULL ring of directions at a fixed radius:
    a point on a closed contour is flanked on many sides by edges whose
    orientation is consistent with a smooth curve passing through it, whereas an
    open/texture edge is not. We sum, over n_dir ring directions phi, the energy
    of the orientation channel tangent to the circle at that flank, sampled at
    the ring point. High values mark locations enclosed by a coherent boundary
    (closure / proto-object interior support). Returns the ring-integrated
    closure map plus its directional VARIANCE (a closed contour gives near-uniform
    support around the ring -> low variance -> a complementary closure cue)."""
    N, _, H, W = mod.shape
    dev, dt = mod.device, mod.dtype
    total = mod.sum(1, keepdim=True)
    acc = torch.zeros(N, 1, H, W, device=dev, dtype=dt)
    ring = []
    for d in range(n_dir):
        phi = d * (2.0 * math.pi / n_dir)
        dx = int(round(radius * math.cos(phi)))
        dy = int(round(radius * math.sin(phi)))
        # orientation tangent to the circle at this flank is perpendicular to phi
        otan = (phi + math.pi / 2.0) % math.pi
        o_idx = int(round(otan / (math.pi / O))) % O
        flank = torch.roll(mod[:, o_idx:o_idx + 1], shifts=(dy, dx), dims=(2, 3))
        ring.append(flank)
        acc = acc + flank
    acc = acc / n_dir
    rstack = torch.cat(ring, dim=1)
    rvar = ((rstack - acc) ** 2).mean(dim=1, keepdim=True)
    # gate by local edge presence so closure support is read at boundary pixels
    closure = torch.sqrt(acc * (total / (total.amax(dim=(2, 3), keepdim=True) + 1e-6)) + 1e-6)
    return torch.cat([closure, torch.sqrt(rvar + 1e-6)], dim=1)


class FiveStream:
    def __init__(self):
        self._dev = None
        self._n_orient = 8
        self._wavelengths = [4.0, 8.0, 16.0]
        self._ksize = [9, 17, 25]
        self._lp_h = 112
        self._lp_w = 96
        self._lp_rmin = 0.025
        self._lp_rmax = 1.0
        # READOUT SPATIAL RESOLUTION (the only change vs champion): add a dense
        # 8x8 grid to the spatial-pyramid pooling so every feature map is read
        # out at finer retinotopic resolution, and use a denser eccentricity x
        # angle grid in the global-form stream. Streams/conjunctions/divnorm are
        # identical to the 0.323 champion.
        self._spp_levels = [1, 2, 3, 5]  # PROVEN best SPP kept EXACTLY unchanged (39 bins/map) -- this model varies ONLY the global-form log-polar aspect below
        self._ac_shifts = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)]
        self._n_rings = 7
        self._n_angsect = 18  # readout-granularity TUNE: re-shape global-form to 7 ecc rings x 18 angular sectors (126 cells) -- fewer eccentricity rings but DENSER angular sampling than the 0.3248 best (9x15); an untried aspect ratio at the edge of the proven 8-14 ring / 12-16 sector band, biasing the log-polar readout toward angular (form-contour) selectivity over radial
        self._dog_specs = [(13, 2.0, 1.6), (21, 4.0, 1.6), (33, 7.0, 1.6)]
        self._surr_k = [9, 17, 25]
        self._dn_sigma2 = 0.02

    def _build(self, device, dtype):
        no = self._n_orient
        self._banks = [
            _steer_bank(k, wl, no, device, dtype)
            for k, wl in zip(self._ksize, self._wavelengths)
        ]
        self._dog = [
            _dog_kernel(k, sc, sr, device, dtype)
            for (k, sc, sr) in self._dog_specs
        ]
        self._grid_cache = {}
        self._dev = device

    def _get_grid(self, in_h, in_w, device, dtype):
        key = (in_h, in_w)
        if key not in self._grid_cache:
            self._grid_cache[key] = _logpolar_grid(
                self._lp_h, self._lp_w, in_h, in_w,
                self._lp_rmin, self._lp_rmax, device, dtype)
        return self._grid_cache[key]

    def _frontend(self, x, with_angwrap):
        mods, phases = [], []
        for (real, imag) in self._banks:
            if with_angwrap:
                pad = real.shape[-1] // 2
                xp = F.pad(x, (pad, pad, 0, 0), mode="circular")
                xp = F.pad(xp, (0, 0, pad, pad), mode="replicate")
                re = F.conv2d(xp, real); im = F.conv2d(xp, imag)
            else:
                re = F.conv2d(x, real, padding="same")
                im = F.conv2d(x, imag, padding="same")
            mod = torch.sqrt(re * re + im * im + 1e-6)
            mods.append(mod); phases.append((re, im))
        return mods, phases

    def _dog_responses(self, lum, with_angwrap):
        resps = []
        for k in self._dog:
            if with_angwrap:
                pad = k.shape[-1] // 2
                xp = F.pad(lum, (pad, pad, 0, 0), mode="circular")
                xp = F.pad(xp, (0, 0, pad, pad), mode="replicate")
                resp = F.conv2d(xp, k)
            else:
                resp = F.conv2d(lum, k, padding="same")
            resps.append(resp)
        return resps

    # ---- STREAM 1: richer Portilla-Simoncelli texture ----
    def _texture_stats(self, mods, phases):
        feats = []
        N, O = mods[0].shape[0], self._n_orient
        for mod in mods:
            feats.append(torch.sqrt(_spp(mod, self._spp_levels).abs() + 1e-6))
        for mod in mods:
            H, W = mod.shape[-2:]
            a = mod - mod.mean(dim=(2, 3), keepdim=True)
            af = a.reshape(N, O, H * W)
            cov = torch.matmul(af, af.transpose(1, 2)) / float(H * W)
            d = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + 1e-6)
            corr = cov / (d.unsqueeze(2) * d.unsqueeze(1) + 1e-6)
            feats.append(_stab(_vech(corr)))
        for si in range(len(mods) - 1):
            fine = mods[si]; coarse = mods[si + 1]
            Hf, Wf = fine.shape[-2:]
            cu = F.interpolate(coarse, size=(Hf, Wf), mode="bilinear",
                               align_corners=False)
            f = fine - fine.mean(dim=(2, 3), keepdim=True)
            c = cu - cu.mean(dim=(2, 3), keepdim=True)
            num = (f * c).mean(dim=(2, 3))
            df = torch.sqrt((f * f).mean(dim=(2, 3)) + 1e-6)
            dc = torch.sqrt((c * c).mean(dim=(2, 3)) + 1e-6)
            feats.append(_stab(num / (df * dc + 1e-6)))
        for si in range(len(mods) - 1):
            re_c, im_c = phases[si + 1]
            mod_c = mods[si + 1]
            cos_c = re_c / (mod_c + 1e-6); sin_c = im_c / (mod_c + 1e-6)
            cos2 = cos_c * cos_c - sin_c * sin_c
            doubled = mod_c * cos2
            mod_f = mods[si]
            Hc, Wc = doubled.shape[-2:]
            if mod_f.shape[-2:] != (Hc, Wc):
                mod_f = F.interpolate(mod_f, size=(Hc, Wc), mode="bilinear",
                                      align_corners=False)
            a = doubled - doubled.mean(dim=(2, 3), keepdim=True)
            b = mod_f - mod_f.mean(dim=(2, 3), keepdim=True)
            af = a.reshape(N, O, Hc * Wc); bf = b.reshape(N, O, Hc * Wc)
            cov = torch.matmul(af, bf.transpose(1, 2)) / (Hc * Wc)
            feats.append(_stab(torch.diagonal(cov, dim1=1, dim2=2)))
            feats.append(_stab(cov.reshape(N, O * O).mean(dim=1, keepdim=True)))
        for mod in mods:
            m = mod - mod.mean(dim=(2, 3), keepdim=True)
            var = (m * m).mean(dim=(2, 3)) + 1e-6
            for (dy, dx) in self._ac_shifts:
                shifted = torch.roll(m, shifts=(dy, dx), dims=(2, 3))
                ac = (m * shifted).mean(dim=(2, 3)) / var
                feats.append(_stab(ac))
        return _rms_norm(torch.cat(feats, dim=1))

    # ---- STREAM 2: mid-level shape ----
    def _shape_stats(self, mods):
        O = self._n_orient
        shape_maps = []
        for mod in mods:
            m = _divnorm(mod, 5, 0.4)
            shape_maps.append(_curvature(m, O))
            shape_maps.append(_end_stop(m, O))
            shape_maps.append(_cocircular(m, O))
            # NEW: proto-object closure / boundary-completion cue
            shape_maps.append(_closure(m, O))
        ref = shape_maps[0].shape[-2:]
        aligned = []
        for s in shape_maps:
            if s.shape[-2:] != ref:
                s = F.interpolate(s, size=ref, mode="bilinear", align_corners=False)
            aligned.append(s)
        shape = _divnorm(torch.cat(aligned, dim=1), 5, 0.5)
        return _rms_norm(torch.sqrt(_spp(shape, self._spp_levels).abs() + 1e-6))

    # ---- STREAM 3: global-form eccentricity-ring profile ----
    def _global_form(self, mods_warp):
        feats = []
        N, O = mods_warp[0].shape[0], self._n_orient
        nr = self._n_rings; na = self._n_angsect
        for mod in mods_warp:
            grid = F.adaptive_avg_pool2d(mod, (nr, na))
            ring_mean = grid.mean(dim=3)
            feats.append(torch.sqrt(ring_mean.reshape(N, -1).abs() + 1e-6))
            rm = grid.mean(dim=3, keepdim=True)
            ang_var = ((grid - rm) ** 2).mean(dim=3)
            feats.append(torch.sqrt(ang_var.reshape(N, -1).abs() + 1e-6))
            env_radial = grid.sum(dim=1)
            feats.append(torch.sqrt(env_radial.reshape(N, -1).abs() + 1e-6))
            fovea = ring_mean[:, :, :max(1, nr // 2)].mean(dim=2)
            periph = ring_mean[:, :, nr - max(1, nr // 2):].mean(dim=2)
            contrast = (fovea - periph) / (fovea + periph + 1e-6)
            feats.append(_stab(contrast))
        return _rms_norm(torch.cat(feats, dim=1))

    # ---- STREAM 4: magnocellular DoG center-surround ----
    def _magno_stats(self, dog_resps):
        feats = []
        mag_maps = []
        for resp in dog_resps:
            on = F.relu(resp)
            off = F.relu(-resp)
            mag = torch.sqrt(resp * resp + 1e-6)
            block = torch.cat([on, off, mag], dim=1)
            block = _divnorm(block, 5, 0.3)
            feats.append(torch.sqrt(_spp(block, self._spp_levels).abs() + 1e-6))
            mag_maps.append(mag)
        for si in range(len(mag_maps) - 1):
            f = mag_maps[si]; c = mag_maps[si + 1]
            f = f - f.mean(dim=(2, 3), keepdim=True)
            c = c - c.mean(dim=(2, 3), keepdim=True)
            num = (f * c).mean(dim=(2, 3))
            df = torch.sqrt((f * f).mean(dim=(2, 3)) + 1e-6)
            dc = torch.sqrt((c * c).mean(dim=(2, 3)) + 1e-6)
            feats.append(_stab(num / (df * dc + 1e-6)))
        return _rms_norm(torch.cat(feats, dim=1))

    # ---- STREAM 5: ALL THREE conjunction families + surround divnorm ----
    def _conjunction_stats(self, mods, dog_resps, chroma_env=None):
        """Fuse the three independently-proven conjunction families:

        (a) TEXTURE-ON-BLOB: oriented energy x DoG ON/OFF figure-ground envelope;
        (b) TEXTURE-ON-CURVATURE: oriented energy gated by local orientation-
            INCOHERENCE (curves/corners/junction sites);
        (c) TEXTURE-ON-GLOBAL-FORM: oriented energy gated by a coarse low-pass
            form envelope (energy on the global figure);
        (d) CROSS-ORIENTATION JUNCTION: energy(o) x energy(o+90deg), locally
            pooled -> corners / crosses / T- & L-junctions;
        (e) COLOR-OPPONENT: oriented (luminance) energy gated by rectified
            chromatic context envelopes (rg+/-, by+/-) when supplied;
        (f) SURROUND divnorm residual at three radii, tuned semi-saturation.
        """
        feats = []
        N, O = mods[0].shape[0], self._n_orient
        half = O // 2
        coarse = dog_resps[-1]
        blob_on = F.relu(coarse)
        blob_off = F.relu(-coarse)
        angles = torch.arange(O, device=mods[0].device, dtype=mods[0].dtype) * (math.pi / O)
        cos2 = torch.cos(2 * angles).view(1, O, 1, 1)
        sin2 = torch.sin(2 * angles).view(1, O, 1, 1)
        for mod in mods:
            H, W = mod.shape[-2:]
            if blob_on.shape[-2:] != (H, W):
                on = F.interpolate(blob_on, size=(H, W), mode="bilinear",
                                   align_corners=False)
                off = F.interpolate(blob_off, size=(H, W), mode="bilinear",
                                    align_corners=False)
            else:
                on, off = blob_on, blob_off
            # (a) texture-on-blob
            feats.append(torch.sqrt(_spp(mod * on, self._spp_levels).abs() + 1e-6))
            feats.append(torch.sqrt(_spp(mod * off, self._spp_levels).abs() + 1e-6))
            # (b) texture-on-curvature (orientation-incoherence gate)
            mx = (mod * cos2).sum(1, keepdim=True)
            my = (mod * sin2).sum(1, keepdim=True)
            mxp = F.avg_pool2d(mx, 5, stride=1, padding=2)
            myp = F.avg_pool2d(my, 5, stride=1, padding=2)
            total = F.avg_pool2d(mod.sum(1, keepdim=True), 5, stride=1, padding=2) + 1e-6
            coh = torch.sqrt(mxp * mxp + myp * myp + 1e-6) / total
            incoh = (1.0 - coh).clamp(min=0.0)
            feats.append(torch.sqrt(_spp(mod * incoh, self._spp_levels).abs() + 1e-6))
            # (c) texture-on-global-form (coarse low-pass envelope gate)
            tot = mod.sum(1, keepdim=True)
            gk = max(3, (min(H, W) // 4) | 1)
            form_env = F.avg_pool2d(tot, kernel_size=gk, stride=1, padding=gk // 2)
            form_env = form_env / (form_env.amax(dim=(2, 3), keepdim=True) + 1e-6)
            feats.append(torch.sqrt(_spp(mod * form_env, self._spp_levels).abs() + 1e-6))
            # (d) cross-orientation junction (energy x orthogonal energy)
            mod_s = F.avg_pool2d(mod, 3, stride=1, padding=1)
            orth = torch.roll(mod_s, shifts=half, dims=1)
            junction = torch.sqrt(mod_s * orth + 1e-6)
            feats.append(torch.sqrt(_spp(junction, self._spp_levels).abs() + 1e-6))
            feats.append(torch.sqrt(_spp(junction.sum(1, keepdim=True),
                                         self._spp_levels).abs() + 1e-6))
            # (e) color-opponent conjunction (luminance energy x chroma context)
            if chroma_env is not None:
                for ce in chroma_env:
                    c = ce
                    if c.shape[-2:] != (H, W):
                        c = F.interpolate(c, size=(H, W), mode="bilinear",
                                          align_corners=False)
                    feats.append(torch.sqrt(_spp(mod * c, self._spp_levels).abs() + 1e-6))
            # (f) surround divnorm residual at three radii
            for sk in self._surr_k:
                surr = F.avg_pool2d(mod, kernel_size=sk, stride=1, padding=sk // 2)
                resid = mod / torch.sqrt(surr * surr + self._dn_sigma2 + 1e-6)
                feats.append(torch.sqrt(_spp(resid, self._spp_levels).abs() + 1e-6))
        return _rms_norm(torch.cat(feats, dim=1))

    def _analyze_local(self, x, with_angwrap=False, chroma_env=None):
        mods, phases = self._frontend(x, with_angwrap)
        dog = self._dog_responses(x, with_angwrap)
        return torch.cat([
            self._texture_stats(mods, phases),
            self._shape_stats(mods),
            self._magno_stats(dog),
            self._conjunction_stats(mods, dog, chroma_env),
        ], dim=1)

    def _analyze_warped(self, x, chroma_env=None):
        mods, phases = self._frontend(x, with_angwrap=True)
        dog = self._dog_responses(x, with_angwrap=True)
        return torch.cat([
            self._texture_stats(mods, phases),
            self._shape_stats(mods),
            self._global_form(mods),
            self._magno_stats(dog),
            self._conjunction_stats(mods, dog, chroma_env),
        ], dim=1)

    def _chroma_envelopes(self, rg, by):
        env = []
        for ch in (rg, by):
            lp = F.avg_pool2d(ch, kernel_size=9, stride=1, padding=4)
            env.append(F.relu(lp))
            env.append(F.relu(-lp))
        return env

    def forward(self, channels, device):
        r, g, b = channels[:, 0:1], channels[:, 1:2], channels[:, 2:3]
        lum = (r + g + b) / 3.0
        rg = (r - g)
        by = (b - (r + g) / 2.0)
        H, W = channels.shape[-2:]
        grid = self._get_grid(H, W, channels.device, channels.dtype)
        chroma_raw = self._chroma_envelopes(rg, by)
        feats = []
        for plane in (lum, rg, by):
            # color-opponent conjunction only gates the luminance plane (where it is
            # a genuine cross-channel product); chroma planes pass chroma_env=None
            ce = chroma_raw if plane is lum else None
            feats.append(self._analyze_local(plane, with_angwrap=False, chroma_env=ce))
            N = plane.shape[0]
            warped = F.grid_sample(
                plane, grid.expand(N, -1, -1, -1),
                mode="bilinear", padding_mode="border", align_corners=True)
            if ce is not None:
                wce = [F.grid_sample(c, grid.expand(N, -1, -1, -1), mode="bilinear",
                                     padding_mode="border", align_corners=True)
                       for c in ce]
            else:
                wce = None
            feats.append(self._analyze_warped(warped, chroma_env=wce))
        out = torch.cat(feats, dim=1)
        out = out / (out.norm(dim=1, keepdim=True) + 1e-6)
        return out


class Model:
    name = "Champion five-stream (closure), readout-granularity tune: proven SPP [1,2,3,5] + 7x18 global-form (denser-angular aspect), 8-orient @128px"
    idea = (
        "Adding feature channels has hard-saturated the champion at 0.323; the one "
        "untested axis is READOUT SPATIAL RESOLUTION. Every stream pools to at most a "
        "4x4 grid (SPP levels [1,2,4]) and global-form to 6x8, but NSD ventral voxels "
        "are RETINOTOPIC, so the ridge is starved of spatial selectivity. Keep the exact "
        "proven streams/conjunctions/divnorm and ONLY make the readout finer: add a dense "
        "8x8 spatial-pyramid grid to every feature map and a denser 10-ring x 12-sector "
        "eccentricity x angle global-form grid. The ridge handles scale, so this purely "
        "gives it spatially-resolved bins to map onto retinotopic voxels."
    )
    summary = (
        "Shared 8-orient x 3-scale steerable pyramid per color-opponent plane in raw "
        "+ log-polar frames feeds five RMS-normalized read-outs: richer-PS texture, "
        "V4 shape (curvature/end-stop/co-circularity + ring-integrated proto-object "
        "closure), eccentricity-ring global-form (now 10x12), magnocellular DoG, and a "
        "UNIFIED conjunction stream (texture-on-blob/curvature/global-form, cross-"
        "orientation junction, color-opponent lum x chroma) under three-radius surround "
        "divisive-normalization. IDENTICAL to the 0.323 champion EXCEPT the spatial "
        "pooling: SPP levels [1,2,4] -> [1,2,4,8] (adds a dense 8x8 grid per feature map) "
        "so the readout is spatially resolved for retinotopic ventral voxels. Streams "
        "normalized independently then concatenated and L2-normalized. Analytic / fixed "
        "kernels, strictly per-image, never trained, Y unseen."
    )
    n_params = 0

    def __init__(self):
        self._impl = FiveStream()

    @torch.no_grad()
    def __call__(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.float()
        device = X.device
        side = int(round((X.shape[1] / 3) ** 0.5))
        outs = []
        bs = 48
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs]
            img = xb.reshape(xb.shape[0], side, side, 3).permute(0, 3, 1, 2).contiguous()
            if self._impl._dev != device:
                self._impl._build(device, img.dtype)
            outs.append(self._impl.forward(img, device))
        return torch.cat(outs, dim=0)


MODEL = Model()
