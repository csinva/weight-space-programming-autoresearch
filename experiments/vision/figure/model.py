# idea: exact coarse relational variance calibration (0.50 vs 0.55 on 114-dim geom_rel) in Stage 2I
import math
import numpy as np
import torch
import torch.nn.functional as F

D_OUT = 9996


def _make_gabor_bank(device, dtype):
    """Create analytic 7x7 quadrature Gabor + DoG filter bank on the fly."""
    ksize = 7
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx * xx + yy * yy

    filters = []
    for sigma, lambd in [(1.1, 2.5), (1.7, 4.2)]:
        for ori_idx in range(8):
            theta = ori_idx * (math.pi / 8.0)
            x_theta = xx * math.cos(theta) + yy * math.sin(theta)
            y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
            env_ori = torch.exp(-0.5 * (x_theta * x_theta + 0.49 * y_theta * y_theta) / (sigma * sigma))
            phase = 2.0 * math.pi * x_theta / lambd
            even = env_ori * torch.cos(phase)
            odd = env_ori * torch.sin(phase)
            even = (even - even.mean()) / (even.norm() + 1e-8)
            odd = (odd - odd.mean()) / (odd.norm() + 1e-8)
            filters.append(even)
            filters.append(odd)

    for s_c, s_s in [(0.8, 1.8), (1.3, 2.8)]:
        g_c = torch.exp(-0.5 * r2 / (s_c * s_c))
        g_c = g_c / g_c.sum()
        g_s = torch.exp(-0.5 * r2 / (s_s * s_s))
        g_s = g_s / g_s.sum()
        dog = g_c - g_s
        dog = (dog - dog.mean()) / (dog.norm() + 1e-8)
        filters.append(dog)

    return torch.stack(filters, dim=0).unsqueeze(1)


def _make_stage2_bank(device, dtype):
    """Create 5x5 second-order modulation bank (4 orientations quadrature + 1 isotropic DoG)."""
    ksize = 5
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx * xx + yy * yy
    sigma, lambd = 1.15, 2.8
    filters = []
    for ori_idx in range(4):
        theta = ori_idx * (math.pi / 4.0)
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        env = torch.exp(-0.5 * (x_theta * x_theta + 0.5 * y_theta * y_theta) / (sigma * sigma))
        phase = 2.0 * math.pi * x_theta / lambd
        even = env * torch.cos(phase)
        odd = env * torch.sin(phase)
        even = (even - even.mean()) / (even.norm() + 1e-8)
        odd = (odd - odd.mean()) / (odd.norm() + 1e-8)
        filters.append(even)
        filters.append(odd)

    g_c = torch.exp(-0.5 * r2 / (0.75 * 0.75))
    g_c = g_c / g_c.sum()
    g_s = torch.exp(-0.5 * r2 / (1.8 * 1.8))
    g_s = g_s / g_s.sum()
    dog = g_c - g_s
    dog = (dog - dog.mean()) / (dog.norm() + 1e-8)
    filters.append(dog)
    return torch.stack(filters, dim=0).unsqueeze(1)


def _upper_tri_gram(feat_map, tri_i, tri_j):
    """Compute centered spatial covariance strict upper triangle per image: feat_map (B, C, H, W)."""
    b, c, h, w = feat_map.shape
    flat = feat_map.reshape(b, c, h * w)
    flat = flat - flat.mean(dim=2, keepdim=True)
    flat = flat / (flat.std(dim=2, keepdim=True) + 1e-5)
    gram = torch.bmm(flat, flat.transpose(1, 2)) / float(h * w)
    return gram[:, tri_i, tri_j]


def _cross_gram(fine_map, coarse_map):
    """Compute full asymmetric cross-correlation matrix between fine_map (B, C1, H, W) and coarse_map (B, C2, H, W)."""
    b, c1, h, w = fine_map.shape
    c2 = coarse_map.shape[1]
    f1 = fine_map.reshape(b, c1, h * w)
    f2 = coarse_map.reshape(b, c2, h * w)
    f1 = (f1 - f1.mean(dim=2, keepdim=True)) / (f1.std(dim=2, keepdim=True) + 1e-5)
    f2 = (f2 - f2.mean(dim=2, keepdim=True)) / (f2.std(dim=2, keepdim=True) + 1e-5)
    return (torch.bmm(f1, f2.transpose(1, 2)) / float(h * w)).flatten(1)


def _pool_denom(g):
    """Local 3x3 + global mean pool for a channel group."""
    local_p = F.avg_pool2d(
        F.pad(g.mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="reflect"),
        kernel_size=3, stride=1
    )
    return local_p + g.mean(dim=(1, 2, 3), keepdim=True) * 0.5


def _nematic_curvature_map(ori_energy):
    """Compute 13-channel orientation harmonic, balanced nematic curvature & surround pop-out map."""
    device, dtype = ori_energy.device, ori_energy.dtype
    angles = torch.arange(8, device=device, dtype=dtype) * (math.pi / 8.0)
    cos2 = torch.cos(2.0 * angles).view(1, 8, 1, 1) * 0.25
    sin2 = torch.sin(2.0 * angles).view(1, 8, 1, 1) * 0.25
    cos4 = torch.cos(4.0 * angles).view(1, 8, 1, 1) * 0.25
    sin4 = torch.sin(4.0 * angles).view(1, 8, 1, 1) * 0.25

    c2 = (ori_energy * cos2).sum(dim=1, keepdim=True)
    s2 = (ori_energy * sin2).sum(dim=1, keepdim=True)
    m2 = torch.sqrt(c2 * c2 + s2 * s2 + 1e-6)

    c4 = (ori_energy * cos4).sum(dim=1, keepdim=True)
    s4 = (ori_energy * sin4).sum(dim=1, keepdim=True)
    m4 = torch.sqrt(c4 * c4 + s4 * s4 + 1e-6)

    kx_kernel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3) * (2.2 / 8.0)
    ky_kernel = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3) * (2.2 / 8.0)

    c2_pad = F.pad(c2, (1, 1, 1, 1), mode="reflect")
    s2_pad = F.pad(s2, (1, 1, 1, 1), mode="reflect")
    dx_c2 = F.conv2d(c2_pad, kx_kernel)
    dy_c2 = F.conv2d(c2_pad, ky_kernel)
    dx_s2 = F.conv2d(s2_pad, kx_kernel)
    dy_s2 = F.conv2d(s2_pad, ky_kernel)

    curv_x = dx_c2 + dy_s2
    curv_y = dx_s2 - dy_c2
    curv_mag = torch.sqrt(curv_x * curv_x + curv_y * curv_y + 1e-6)

    c2_surr = F.avg_pool2d(F.pad(c2, (3, 3, 3, 3), mode="reflect"), kernel_size=7, stride=1)
    s2_surr = F.avg_pool2d(F.pad(s2, (3, 3, 3, 3), mode="reflect"), kernel_size=7, stride=1)
    m2_surr = torch.sqrt(c2_surr * c2_surr + s2_surr * s2_surr + 1e-6)
    cos_2dtheta = (c2 * c2_surr + s2 * s2_surr) / (m2 * m2_surr + 1e-5)
    e_iso = m2 * F.relu(cos_2dtheta)
    e_ortho = m2 * F.relu(-cos_2dtheta) * 1.5

    return torch.cat([
        c2, s2, m2,
        c4, s4, m4,
        F.relu(curv_x), F.relu(-curv_x),
        F.relu(curv_y), F.relu(-curv_y),
        curv_mag,
        e_iso, e_ortho,
    ], dim=1)


def _phase_congruency_map(even_l, odd_l, energy_l, lum_ref, norm_map):
    """Compute 9-channel phase congruency, bar/crease polarity, spectral slope & odd gradient sign map."""
    device, dtype = even_l.device, even_l.dtype
    denom = lum_ref + 1e-4

    b_ridge = torch.log1p(F.relu(even_l).mean(dim=1, keepdim=True) / denom)
    b_crease = torch.log1p(F.relu(-even_l).mean(dim=1, keepdim=True) / denom)
    s_line_edge = torch.tanh((even_l.abs() - odd_l.abs()).mean(dim=1, keepdim=True) / denom)

    ef, em = even_l[:, 0:8], even_l[:, 8:16]
    of, om = odd_l[:, 0:8], odd_l[:, 8:16]
    enf, enm = energy_l[:, 0:8], energy_l[:, 8:16]
    cos_dphi = (ef * em + of * om) / (enf * enm + 1e-5)
    phase_coh = torch.log1p((F.relu(cos_dphi) * torch.sqrt(enf * enm + 1e-6)).mean(dim=1, keepdim=True) / denom)

    d_freq = torch.log1p(enf.mean(dim=1, keepdim=True) / denom) - torch.log1p(enm.mean(dim=1, keepdim=True) / denom)

    angles = torch.arange(8, device=device, dtype=dtype) * (math.pi / 8.0)
    cos1 = torch.cos(angles).view(1, 8, 1, 1)
    sin1 = torch.sin(angles).view(1, 8, 1, 1)
    odd_norm = 0.5 * (of + om) / denom
    gx = (odd_norm * cos1).mean(dim=1, keepdim=True)
    gy = (odd_norm * sin1).mean(dim=1, keepdim=True)

    return torch.cat([
        b_ridge, b_crease, s_line_edge, phase_coh, d_freq,
        torch.log1p(F.relu(gx)), torch.log1p(F.relu(-gx)),
        torch.log1p(F.relu(gy)), torch.log1p(F.relu(-gy)),
    ], dim=1)


def _octave_summary12(smap, cmap, pmap, gmap):
    """Extract 12 non-negative summary channels at a given octave including 3D convexity and 2D cornerness."""
    ori8 = 0.5 * (smap[:, 0:8] + smap[:, 8:16])
    ori4 = 0.5 * (ori8[:, 0:8:2] + ori8[:, 1:8:2])
    dog1 = 0.25 * smap[:, 16:20].sum(dim=1, keepdim=True)
    convex = 0.5 * (gmap[:, 3:4] + gmap[:, 4:5])
    rg1 = 0.25 * smap[:, 20:24].sum(dim=1, keepdim=True)
    by1 = 0.25 * smap[:, 24:28].sum(dim=1, keepdim=True)
    m2_kmag = cmap[:, [2, 10]]
    mu_min_pc = torch.cat([gmap[:, 5:6], pmap[:, 3:4]], dim=1)
    return torch.cat([ori4, dog1, convex, rg1, by1, m2_kmag, mu_min_pc], dim=1)


def _compact_pool(z, grid):
    """Multi-scale hierarchical spatial pooling on a 4x4 grid."""
    pooled = F.adaptive_avg_pool2d(z, grid)
    coarse2 = F.avg_pool2d(pooled, 2).repeat_interleave(2, 2).repeat_interleave(2, 3)
    coarse4 = F.avg_pool2d(pooled, 4).repeat_interleave(4, 2).repeat_interleave(4, 3)
    return (pooled + (1.25 ** 0.5 - 1.0) * coarse2 + (1.3125 ** 0.5 - 1.25 ** 0.5) * coarse4).flatten(1)


def _path_maps(small):
    """Compute color-opponent distance from 5 canonical spatial anchors (4 corners + center) on 32x32 grid."""
    small = (small - 0.9 * small.mean(1, keepdim=True)) * ((3.0 / 2.01) ** 0.5)
    anchors = F.avg_pool2d(F.pad(small, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    maps = []
    for oy, ox in ((0, 0), (0, 31), (31, 0), (31, 31), (16, 16)):
        distance = (small - anchors[:, :, oy : oy + 1, ox : ox + 1]).square().mean(1).sqrt()
        maps.append(distance)
    return torch.stack(maps, 1)


def _additive_path_maps(small):
    """Compute L1 boundary path-variation from 5 canonical spatial anchors on 32x32 grid."""
    horizontal = 100.0 * (small[:, :, :, 1:] - small[:, :, :, :-1]).square().mean(1).sqrt()
    vertical = 100.0 * (small[:, :, 1:, :] - small[:, :, :-1, :]).square().mean(1).sqrt()
    hp = F.pad(horizontal.cumsum(2), (1, 0, 0, 0))
    vp = F.pad(vertical.cumsum(1), (0, 0, 1, 0))
    maps = []
    for oy, ox in ((0, 0), (0, 31), (31, 0), (31, 31), (16, 16)):
        hfirst = (hp[:, oy, :] - hp[:, oy, ox][:, None]).abs()
        vafter = (vp - vp[:, oy : oy + 1, :]).abs()
        vfirst = (vp[:, :, ox] - vp[:, oy, ox][:, None]).abs()
        hafter = (hp - hp[:, :, ox : ox + 1]).abs()
        maps.append(torch.minimum(hfirst[:, None, :] + vafter, vfirst[:, :, None] + hafter))
    distance = torch.stack(maps, 1)
    return (distance.clamp_min(0.0) / 10.0).log1p()


def _extrema_counts(values):
    """Count bright and dark local-extremum plateaus on a 32x32 grayscale map."""
    values = values.tolist()
    seen = [False] * 1024
    maxima = minima = 0
    for u in range(1024):
        if seen[u]:
            continue
        value = values[u]
        seen[u] = True
        stack = [u]
        is_peak = is_trough = True
        has_boundary = False
        while stack:
            v = stack.pop()
            y, x = divmod(v, 32)
            neighbors = []
            if x:
                neighbors.append(v - 1)
            if x < 31:
                neighbors.append(v + 1)
            if y:
                neighbors.append(v - 32)
            if y < 31:
                neighbors.append(v + 32)
            for w in neighbors:
                other = values[w]
                if other == value:
                    if not seen[w]:
                        seen[w] = True
                        stack.append(w)
                else:
                    has_boundary = True
                    if other > value:
                        is_peak = False
                    if other < value:
                        is_trough = False
        maxima += int(has_boundary and is_peak)
        minima += int(has_boundary and is_trough)
    return max(maxima - 1, 0), max(minima - 1, 0)


def _persistence_features(x):
    """Compute 2-dim square-root density of bright and dark local-extremum plateaus."""
    gray = F.interpolate(x, size=(32, 32), mode="area").mean(1).flatten(1).cpu().numpy()
    counts = np.asarray([_extrema_counts(values) for values in gray], dtype=np.float64)
    return 1.2 * torch.as_tensor(np.sqrt(counts / 1024.0), device=x.device, dtype=x.dtype)


def _eccentricity_and_symmetry(oct_map):
    """Compute 9-ch foveal-vs-peripheral contrast (9 dims) & upper/lower bilateral mirror correlation (18 dims)."""
    b, _, h, w = oct_map.shape
    device, dtype = oct_map.device, oct_map.dtype
    # 9 horizontally-symmetric summary channels
    h0 = 0.25 * oct_map[:, 0:4].sum(dim=1, keepdim=True)
    hv_diff = oct_map[:, 0:1] - oct_map[:, 2:3]
    sym9 = torch.cat([h0, hv_diff, oct_map[:, [4, 5, 6, 7, 8, 9, 11]]], dim=1)  # (B, 9, H, W)

    yc = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xc = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yc, xc, indexing="ij")
    fov_mask = ((xx * xx + yy * yy) < 0.30).to(dtype).view(1, 1, h, w)
    per_mask = 1.0 - fov_mask

    fov_mean = (sym9 * fov_mask).sum(dim=(2, 3)) / (fov_mask.sum() + 1e-6)
    per_mean = (sym9 * per_mask).sum(dim=(2, 3)) / (per_mask.sum() + 1e-6)
    ecc_diff = fov_mean - per_mean  # (B, 9)

    # Bilateral left-right reflection correlation in upper and lower hemifields
    half_h, half_w = h // 2, w // 2
    sym_corrs = []
    for yh in (slice(0, half_h), slice(half_h, 2 * half_h)):
        left = sym9[:, :, yh, :half_w].reshape(b, 9, -1)
        right = torch.flip(sym9[:, :, yh, w - half_w :], dims=[3]).reshape(b, 9, -1)
        left = (left - left.mean(dim=2, keepdim=True)) / (left.std(dim=2, keepdim=True) + 1e-5)
        right = (right - right.mean(dim=2, keepdim=True)) / (right.std(dim=2, keepdim=True) + 1e-5)
        sym_corrs.append((left * right).mean(dim=2))  # (B, 9)

    return torch.cat([ecc_diff, sym_corrs[0], sym_corrs[1]], dim=1)  # (B, 27)


def _geom_sym_scale(gmap, n_ch=12):
    """Compute foveal-vs-peripheral contrast & bilateral mirror correlation on 12-ch reflection-invariant surface geometry."""
    b, _, h, w = gmap.shape
    device, dtype = gmap.device, gmap.dtype
    g6 = torch.cat([
        gmap[:, 3:4] + gmap[:, 4:5],
        gmap[:, 0:1] + gmap[:, 1:2],
        gmap[:, 2:3],
        gmap[:, 5:6],
        gmap[:, 8:9],
        gmap[:, 9:10],
    ], dim=1)
    gc6 = torch.cat([
        gmap[:, 4:5] - gmap[:, 3:4],
        gmap[:, 0:1] - gmap[:, 1:2],
        gmap[:, 6:7],
        gmap[:, 7:8],
        gmap[:, 10:11],
        gmap[:, 11:12],
    ], dim=1)
    g_in = torch.cat([g6[:, :4], g6[:, 4:5] + g6[:, 5:6], gc6], dim=1) if n_ch == 11 else torch.cat([g6, gc6], dim=1)
    yc = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xc = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yc, xc, indexing="ij")
    fov = ((xx * xx + yy * yy) < 0.30).to(dtype).view(1, 1, h, w)
    per = 1.0 - fov
    ecc = (g_in * fov).sum(dim=(2, 3)) / (fov.sum() + 1e-6) - (g_in * per).sum(dim=(2, 3)) / (per.sum() + 1e-6)
    hw = w // 2
    left = g_in[:, :, :, :hw].reshape(b, n_ch, -1)
    right = torch.flip(g_in[:, :, :, w - hw :], dims=[3]).reshape(b, n_ch, -1)
    left = (left - left.mean(dim=2, keepdim=True)) / (left.std(dim=2, keepdim=True) + 1e-5)
    right = (right - right.mean(dim=2, keepdim=True)) / (right.std(dim=2, keepdim=True) + 1e-5)
    sym = (left * right).mean(dim=2)
    return torch.cat([ecc, sym], dim=1)


def _cort_warp(feat_map):
    """Warp feature map from retinal (x, y) to uniform cortical sheet (u, v) via x = 0.58*u + 0.42*u^3."""
    b, _, h, w = feat_map.shape
    device, dtype = feat_map.device, feat_map.dtype
    u = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
    v = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
    xu = 0.58 * u + 0.42 * (u * u * u)
    yv = 0.58 * v + 0.42 * (v * v * v)
    grid_y, grid_x = torch.meshgrid(yv, xu, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _stage1_filter(lum, rg, by, filters, chrom_filters):
    """Compute 36-channel pathway-normalized Stage 1 feature map and luminance quadrature components."""
    lum_conv = F.conv2d(F.pad(lum, (3, 3, 3, 3), mode="reflect"), filters)
    even_l, odd_l = lum_conv[:, 0:32:2], lum_conv[:, 1:32:2]
    energy_l = torch.sqrt(even_l * even_l + odd_l * odd_l + 1e-6)
    dog_l = lum_conv[:, 32:34]
    dog_l_rect = torch.cat([F.relu(dog_l), F.relu(-dog_l)], dim=1)

    rg_conv = F.conv2d(F.pad(rg, (3, 3, 3, 3), mode="reflect"), chrom_filters)
    by_conv = F.conv2d(F.pad(by, (3, 3, 3, 3), mode="reflect"), chrom_filters)
    energy_rg = (torch.sqrt(rg_conv[:, 0:8:2] ** 2 + rg_conv[:, 1:8:2] ** 2 + 1e-6) - 1e-3).clamp_min(0.0)
    energy_by = (torch.sqrt(by_conv[:, 0:8:2] ** 2 + by_conv[:, 1:8:2] ** 2 + 1e-6) - 1e-3).clamp_min(0.0)
    dog_rg_rect = torch.cat([F.relu(rg_conv[:, 8:10]), F.relu(-rg_conv[:, 8:10])], dim=1)
    dog_by_rect = torch.cat([F.relu(by_conv[:, 8:10]), F.relu(-by_conv[:, 8:10])], dim=1)

    lum_ref = _pool_denom(energy_l)
    hi_denom = 0.85 * _pool_denom(energy_l[:, 0:8]) + 0.15 * lum_ref + 1e-4
    lo_denom = 0.85 * _pool_denom(energy_l[:, 8:16]) + 0.15 * lum_ref + 1e-4
    dog_l_denom = 0.85 * _pool_denom(dog_l_rect) + 0.15 * lum_ref + 1e-4
    rg_gab_denom = 0.85 * _pool_denom(energy_rg) + 0.15 * lum_ref + 1e-4
    by_gab_denom = 0.85 * _pool_denom(energy_by) + 0.15 * lum_ref + 1e-4
    dog_rg_denom = 0.85 * _pool_denom(dog_rg_rect) + 0.15 * lum_ref + 1e-4
    dog_by_denom = 0.85 * _pool_denom(dog_by_rect) + 0.15 * lum_ref + 1e-4

    norm_map = torch.log1p(torch.cat([
        energy_l[:, 0:8] / hi_denom,
        energy_l[:, 8:16] / lo_denom,
        dog_l_rect / dog_l_denom,
        energy_rg / rg_gab_denom,
        energy_by / by_gab_denom,
        dog_rg_rect / dog_rg_denom,
        dog_by_rect / dog_by_denom,
    ], dim=1))
    return norm_map, even_l, odd_l, energy_l, lum_ref


def _std_pool2d(feat_map, k, peak=False):
    m = F.adaptive_avg_pool2d(feat_map, (k, k))
    sq = F.adaptive_avg_pool2d(feat_map * feat_map, (k, k))
    std = torch.sqrt((sq - m * m).clamp_min(1e-8))
    if peak:
        sub = F.avg_pool2d(F.pad(feat_map, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        std = 0.82 * std + 0.117 * F.relu(F.adaptive_max_pool2d(sub, (k, k)) - m)
    return std.flatten(1)


def _chrom_i2d(ch):
    p = F.pad(ch, (1, 1, 1, 1), mode="reflect")
    cx = 0.5 * (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2])
    cy = 0.5 * (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1])
    jxx = F.avg_pool2d(cx * cx, kernel_size=3, stride=1, padding=1)
    jyy = F.avg_pool2d(cy * cy, kernel_size=3, stride=1, padding=1)
    jxy = F.avg_pool2d(cx * cy, kernel_size=3, stride=1, padding=1)
    tr = jxx + jyy
    det_disc = torch.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy + 1e-10)
    mu_min = torch.sqrt((0.5 * (tr - det_disc)).clamp_min(0.0) + 1e-8)
    norm = F.avg_pool2d(torch.sqrt(tr + 1e-8), kernel_size=5, stride=1, padding=2) + 0.03
    return mu_min / norm


def _surface_geom_map(lum, rg, by):
    """Compute 14-channel Hessian Shape Index, Curvedness, Structure Tensor Intrinsic-2D & Isophote Curvature map."""
    L = F.avg_pool2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    rg_s = F.avg_pool2d(F.pad(rg, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    by_s = F.avg_pool2d(F.pad(by, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    Lp = F.pad(L, (1, 1, 1, 1), mode="reflect")
    Lx = 0.5 * (Lp[:, :, 1:-1, 2:] - Lp[:, :, 1:-1, :-2])
    Ly = 0.5 * (Lp[:, :, 2:, 1:-1] - Lp[:, :, :-2, 1:-1])
    Lxx = Lp[:, :, 1:-1, 2:] - 2.0 * L + Lp[:, :, 1:-1, :-2]
    Lyy = Lp[:, :, 2:, 1:-1] - 2.0 * L + Lp[:, :, :-2, 1:-1]
    Lxy = 0.25 * (Lp[:, :, 2:, 2:] - Lp[:, :, 2:, :-2] - Lp[:, :, :-2, 2:] + Lp[:, :, :-2, :-2])

    H_tr = Lxx + Lyy
    D1 = Lxx - Lyy
    D2 = 2.0 * Lxy
    H_dev = torch.sqrt(D1 * D1 + D2 * D2 + 1e-8)
    curvedness = 0.5 * torch.sqrt(H_tr * H_tr + H_dev * H_dev + 1e-8)
    c_comp = torch.log1p(6.0 * curvedness)
    c_denom = F.avg_pool2d(c_comp, kernel_size=5, stride=1, padding=2) + 0.1
    c_norm = c_comp / c_denom

    shape_idx = (2.0 / math.pi) * torch.atan2(-H_tr, H_dev)
    shape_bins = [
        c_norm * torch.exp(-((shape_idx - center) ** 2) / (2.0 * 0.35 * 0.35))
        for center in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ]

    jxx = F.avg_pool2d(Lx * Lx, kernel_size=3, stride=1, padding=1)
    jyy = F.avg_pool2d(Ly * Ly, kernel_size=3, stride=1, padding=1)
    jxy = F.avg_pool2d(Lx * Ly, kernel_size=3, stride=1, padding=1)
    tr = jxx + jyy
    disc = torch.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy + 1e-10)
    mu_max = torch.sqrt(0.5 * (tr + disc) + 1e-8)
    mu_min = torch.sqrt((0.5 * (tr - disc)).clamp_min(0.0) + 1e-8)
    st_denom = F.avg_pool2d(mu_max, kernel_size=5, stride=1, padding=2) + 0.03
    mu_min_n = mu_min / st_denom
    mu_max_n = mu_max / st_denom
    coherence = disc / (tr + 1e-5)

    grad_sq = Lx * Lx + Ly * Ly + 1e-4
    kappa_w = -(Ly * Ly * Lxx - 2.0 * Lx * Ly * Lxy + Lx * Lx * Lyy) / grad_sq
    k_pos = torch.log1p(6.0 * F.relu(kappa_w)) / c_denom
    k_neg = torch.log1p(6.0 * F.relu(-kappa_w)) / c_denom
    d1_n = torch.log1p(4.0 * D1.abs()) / c_denom
    d2_n = torch.log1p(4.0 * D2.abs()) / c_denom

    return torch.cat(
        shape_bins + [mu_min_n, mu_max_n, coherence, k_pos, k_neg, d1_n, d2_n, _chrom_i2d(rg_s), _chrom_i2d(by_s)],
        dim=1,
    )


class Model:
    """5-octave 12-channel surface differential geometry cascade with calibrated coarse relational variance."""

    name = "calibrated-coarse-relational-surface-cascade"
    idea = "exact coarse relational variance calibration (0.50 vs 0.55 on 114-dim geom_rel) in Stage 2I"
    description = (
        "Extends 2pxr1 with a single-factor change in Stage 2I reducing the 114-dim coarse surface-topology relational block "
        "scale from 0.55 to 0.50 (0.50 * geom_rel / sqrt(||geom_rel||^2 + 8.0)) so its total variance (0.01034) matches the "
        "63-dim zr79v/e9n7s coarse relational baseline (0.01030) while retaining all 114 [w3, wc3, c23, cc23, 0.5*c34_diag] features."
    )
    insight = (
        "In 58q88, reducing the combined 114-dim coarse relational variance from 0.0213 to 0.01251 improved both early (+0.000114) "
        "and ventral (+0.000081) encoding, but left a 21.5% variance overshoot relative to zr79v's 0.0103. Equilibrating the scale "
        "to 0.50 tests whether eliminating the remaining coarse variance overshoot further improves early and pooled encoding."
    )
    n_params = 0

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X).float()
        if X.ndim == 1:
            X = X.unsqueeze(0)
        n = X.shape[0]
        flat_dim = X.shape[1]
        if flat_dim < 3 or int(round(math.sqrt(flat_dim / 3.0))) ** 2 * 3 != flat_dim:
            return X[:, : min(flat_dim, D_OUT)]

        side = int(round(math.sqrt(flat_dim / 3.0)))
        chunk_size = 128
        out_chunks = []
        filters = _make_gabor_bank(X.device, X.dtype)
        chrom_filters = filters[[16, 17, 20, 21, 24, 25, 28, 29, 32, 33]]
        s2_filters = _make_stage2_bank(X.device, X.dtype)
        tri24_i, tri24_j = torch.triu_indices(24, 24, offset=1, device=X.device)
        keep_cg = ~(((tri24_i >= 8) & (tri24_j < 12)) | ((tri24_i == 20) & (tri24_j == 21)) | ((tri24_i == 22) & (tri24_j == 23)))
        cg_i, cg_j = tri24_i[keep_cg], tri24_j[keep_cg]
        in_iso = (tri24_i == 13) | (tri24_i == 14) | (tri24_i == 18) | (tri24_i == 19)
        jn_iso = (tri24_j == 13) | (tri24_j == 14) | (tri24_j == 18) | (tri24_j == 19)
        in_g4 = (tri24_i == 7) | (tri24_i == 8) | (tri24_i == 11) | (tri24_i == 12)
        jn_g4 = (tri24_j == 7) | (tri24_j == 8) | (tri24_j == 11) | (tri24_j == 12)
        drop_csc = (in_iso & jn_iso) | (in_g4 & jn_g4) | ((tri24_i == 5) & (tri24_j == 6)) | ((tri24_i == 7) & (tri24_j == 9)) | (tri24_i == 15) | (tri24_j == 15)
        keep_csc = (tri24_j < 20) & ~((tri24_i == 0) & (tri24_j == 19)) & ~drop_csc
        csc_i, csc_j = tri24_i[keep_csc], tri24_j[keep_csc]
        tri12_i, tri12_j = torch.triu_indices(12, 12, offset=1, device=X.device)
        tri6_i, tri6_j = torch.triu_indices(6, 6, offset=1, device=X.device)

        for start in range(0, n, chunk_size):
            x = X[start : start + chunk_size].reshape(-1, side, side, 3).permute(0, 3, 1, 2)
            x = x.clamp(0.0, 1.0)
            x_ret = (x + 0.04).pow(0.45)
            x_cort = _cort_warp(x_ret)

            r, g, b = x_ret[:, 0:1], x_ret[:, 1:2], x_ret[:, 2:3]
            lum = (r + g + b) * (1.0 / math.sqrt(3.0))
            rg = (r - g) * (1.0 / math.sqrt(2.0))
            by = (r + g - 2.0 * b) * (1.0 / math.sqrt(6.0))

            rc, gc, bc = x_cort[:, 0:1], x_cort[:, 1:2], x_cort[:, 2:3]
            lum_c = (rc + gc + bc) * (1.0 / math.sqrt(3.0))
            rg_c = (rc - gc) * (1.0 / math.sqrt(2.0))
            by_c = (rc + gc - 2.0 * bc) * (1.0 / math.sqrt(6.0))

            small = F.interpolate(x, size=(32, 32), mode="area")
            patterns = _compact_pool(_path_maps(small), 4).reshape(len(x), 5, 16)
            strengths = _compact_pool(_additive_path_maps(small), 4).reshape(len(x), 5, 16).norm(dim=2, keepdim=True)
            path_features = 0.02 * (patterns * strengths / patterns.norm(dim=2, keepdim=True).clamp_min(1e-8)).flatten(1)
            feats = [path_features, _persistence_features(x)]
            stage1_maps = []
            curv_maps = []
            phase_maps = []
            geom_maps = []

            for scale_idx in range(3):
                if scale_idx > 0:
                    lum = F.avg_pool2d(lum, kernel_size=2, stride=2)
                    rg = F.avg_pool2d(rg, kernel_size=2, stride=2)
                    by = F.avg_pool2d(by, kernel_size=2, stride=2)
                    lum_c = F.avg_pool2d(lum_c, kernel_size=2, stride=2)
                    rg_c = F.avg_pool2d(rg_c, kernel_size=2, stride=2)
                    by_c = F.avg_pool2d(by_c, kernel_size=2, stride=2)

                norm_map, even_l, odd_l, energy_l, lum_ref = _stage1_filter(lum, rg, by, filters, chrom_filters)
                cort_map, _, _, _, _ = _stage1_filter(lum_c, rg_c, by_c, filters, chrom_filters)

                stage1_maps.append(norm_map)
                curv_maps.append(_nematic_curvature_map(0.5 * (norm_map[:, 0:8] + norm_map[:, 8:16])))
                phase_maps.append(_phase_congruency_map(even_l, odd_l, energy_l, lum_ref, norm_map))
                geom_maps.append(_surface_geom_map(lum, rg, by))

                n30 = torch.cat([
                    norm_map[:, 0:16],
                    0.5 * (norm_map[:, 16:20:2] + norm_map[:, 17:20:2]),
                    norm_map[:, 20:28],
                    0.5 * (norm_map[:, 28:36:2] + norm_map[:, 29:36:2]),
                ], dim=1)
                c30 = torch.cat([
                    cort_map[:, 0:16],
                    0.5 * (cort_map[:, 16:20:2] + cort_map[:, 17:20:2]),
                    cort_map[:, 20:28],
                    0.5 * (cort_map[:, 28:36:2] + cort_map[:, 29:36:2]),
                ], dim=1)

                if scale_idx == 0:
                    g9 = F.adaptive_avg_pool2d(c30, (9, 9)).flatten(1)
                    g5 = _std_pool2d(n30, 5, True)
                    feats.append(g9 / (g9.norm(dim=1, keepdim=True) + 1e-6))
                    feats.append(g5 / (g5.norm(dim=1, keepdim=True) + 1e-6))
                elif scale_idx == 1:
                    g6 = F.adaptive_avg_pool2d(c30, (6, 6)).flatten(1)
                    g3 = _std_pool2d(n30, 3, True)
                    feats.append(g6 / (g6.norm(dim=1, keepdim=True) + 1e-6))
                    feats.append(g3 / (g3.norm(dim=1, keepdim=True) + 1e-6))
                else:
                    g4 = F.adaptive_avg_pool2d(c30, (4, 4)).flatten(1)
                    g2 = _std_pool2d(n30, 2, True)
                    g_mean = n30.mean(dim=(2, 3))
                    g_std = n30.std(dim=(2, 3))
                    feats.append(g4 / (g4.norm(dim=1, keepdim=True) + 1e-6))
                    feats.append(g2 / (g2.norm(dim=1, keepdim=True) + 1e-6))
                    g_stat = torch.cat([g_mean, g_std], dim=1)
                    feats.append(g_stat / (g_stat.norm(dim=1, keepdim=True) + 1e-6))

            color_grid = torch.cat([
                F.adaptive_avg_pool2d(lum, (6, 6)).flatten(1),
                F.adaptive_avg_pool2d(rg, (6, 6)).flatten(1),
                F.adaptive_avg_pool2d(by, (6, 6)).flatten(1),
            ], dim=1)
            feats.append(color_grid / (color_grid.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2A: Second-order oriented modulation (computed on undistorted retinal maps) ---
            m1 = stage1_maps[1]
            s1_energy = 0.5 * (m1[:, 0:8] + m1[:, 8:16])
            s1_Selected = torch.cat([s1_energy, m1[:, 20:24], m1[:, 24:28]], dim=1)
            s1_ds = F.avg_pool2d(s1_Selected, kernel_size=2, stride=2)
            s1_ds = s1_ds - s1_ds.mean(dim=(2, 3), keepdim=True)

            s2_weight = s2_filters.repeat(16, 1, 1, 1)
            s2_conv = F.conv2d(F.pad(s1_ds, (2, 2, 2, 2), mode="reflect"), s2_weight, groups=16)
            s2_conv = s2_conv.reshape(-1, 16, 9, s1_ds.shape[2], s1_ds.shape[3])
            s2_mod = torch.sqrt(s2_conv[:, :, 0:8:2] ** 2 + s2_conv[:, :, 1:8:2] ** 2 + 1e-6)
            s2_lap = torch.abs(s2_conv[:, :, 8:9])
            s2_map = torch.log1p(torch.cat([s2_mod, s2_lap], dim=2).flatten(1, 2))

            s2_g3 = F.adaptive_avg_pool2d(_cort_warp(s2_map), (3, 3)).flatten(1)
            s2_stat = torch.cat([s2_map.mean(dim=(2, 3)), s2_map.std(dim=(2, 3))], dim=1)
            feats.append(0.75 * s2_g3 / (s2_g3.norm(dim=1, keepdim=True) + 1e-6))
            feats.append(0.75 * s2_stat / (s2_stat.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2B: Within-scale cross-channel covariance (strictly off-diagonal on retinal maps) ---
            cov_list = []
            for smap in (stage1_maps[1], stage1_maps[2]):
                c24 = torch.cat([
                    0.5 * (smap[:, 0:8] + smap[:, 8:16]),
                    smap[:, 16:20],
                    smap[:, 20:28],
                    0.5 * (smap[:, 28:32] + smap[:, 32:36]),
                ], dim=1)
                cov_list.append(_upper_tri_gram(c24, cg_i, cg_j))
            cov_global = torch.cat(cov_list, dim=1)  # 528
            feats.append(0.75 * cov_global / (torch.sqrt(cov_global.pow(2).sum(dim=1, keepdim=True) + 48.0) + 1e-6))

            s1_quad12 = torch.cat([
                s1_energy,
                0.5 * (m1[:, 20:22] + m1[:, 22:24]),
                0.5 * (m1[:, 24:26] + m1[:, 26:28]),
            ], dim=1)
            s1_quad12_w = _cort_warp(s1_quad12)
            q_covs = []
            h1, w1 = s1_quad12_w.shape[2] // 2, s1_quad12_w.shape[3] // 2
            for qi in range(2):
                for qj in range(2):
                    patch = s1_quad12_w[:, :, qi * h1 : (qi + 1) * h1, qj * w1 : (qj + 1) * w1]
                    q_covs.append(_upper_tri_gram(patch, tri12_i, tri12_j))
            cov_quad = torch.cat(q_covs, dim=1)  # 264
            feats.append(0.75 * cov_quad / (torch.sqrt(cov_quad.pow(2).sum(dim=1, keepdim=True) + 48.0) + 1e-6))

            # --- Stage 2C: Multi-scale Nematic Curvature, Harmonics & Contextual Pop-out ---
            c_s0 = F.adaptive_avg_pool2d(_cort_warp(curv_maps[0]), (4, 4)).flatten(1)
            c_s1 = F.adaptive_avg_pool2d(_cort_warp(curv_maps[1]), (4, 4)).flatten(1)
            c_s2 = torch.cat([
                F.adaptive_avg_pool2d(_cort_warp(curv_maps[2]), (2, 2)).flatten(1),
                curv_maps[2].mean(dim=(2, 3)),
                curv_maps[2].std(dim=(2, 3)),
            ], dim=1)
            feats.append(0.75 * c_s0 / (c_s0.norm(dim=1, keepdim=True) + 1e-6))
            feats.append(0.75 * c_s1 / (c_s1.norm(dim=1, keepdim=True) + 1e-6))
            feats.append(0.75 * c_s2 / (c_s2.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2E: Multi-scale Phase Congruency, Bar/Crease Polarity & Spectral Slope ---
            p_s0 = F.adaptive_avg_pool2d(_cort_warp(phase_maps[0]), (4, 4)).flatten(1)
            p_s1 = F.adaptive_avg_pool2d(_cort_warp(phase_maps[1]), (4, 4)).flatten(1)
            p_s2 = torch.cat([
                F.adaptive_avg_pool2d(_cort_warp(phase_maps[2]), (2, 2)).flatten(1),
                phase_maps[2].mean(dim=(2, 3)),
                phase_maps[2].std(dim=(2, 3)),
            ], dim=1)
            feats.append(0.75 * p_s0 / (p_s0.norm(dim=1, keepdim=True) + 1e-6))
            feats.append(0.75 * p_s1 / (p_s1.norm(dim=1, keepdim=True) + 1e-6))
            feats.append(0.75 * p_s2 / (p_s2.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2D: Joint Surface-Topology, Curvature, Phase, Shading & Color Covariance (off-diagonal) ---
            csc_list = []
            for cmap, pmap, smap, gmap in (
                (curv_maps[1], phase_maps[1], stage1_maps[1], geom_maps[1]),
                (curv_maps[2], phase_maps[2], stage1_maps[2], geom_maps[2]),
            ):
                c7 = cmap[:, [0, 1, 2, 5, 10, 11, 12]]
                g6_csc = torch.cat([
                    gmap[:, 3:4] + gmap[:, 4:5],
                    gmap[:, 0:1] + gmap[:, 1:2],
                    gmap[:, 2:3],
                    gmap[:, 5:6],
                    gmap[:, 8:9],
                    gmap[:, 9:10],
                ], dim=1)
                p5 = pmap[:, 0:5]
                s6 = torch.cat([
                    0.5 * (smap[:, 16:20:2] + smap[:, 17:20:2]),
                    0.5 * (smap[:, 20:22] + smap[:, 22:24]),
                    0.5 * (smap[:, 24:26] + smap[:, 26:28]),
                ], dim=1)
                csc24 = torch.cat([c7, g6_csc, p5, s6], dim=1)
                csc_list.append(_upper_tri_gram(csc24, csc_i, csc_j))
            cov_csc = torch.cat(csc_list, dim=1)  # 522
            feats.append(0.75 * cov_csc / (torch.sqrt(cov_csc.pow(2).sum(dim=1, keepdim=True) + 48.0) + 1e-6))

            # --- Stage 2F: Adjacent-Octave Persistence, Diff & Parent-Child Covariance with Surface Topology ---
            oct0 = _octave_summary12(stage1_maps[0], curv_maps[0], phase_maps[0], geom_maps[0])
            oct1 = _octave_summary12(stage1_maps[1], curv_maps[1], phase_maps[1], geom_maps[1])
            oct2 = _octave_summary12(stage1_maps[2], curv_maps[2], phase_maps[2], geom_maps[2])

            oct0_ds = F.avg_pool2d(oct0, kernel_size=2, stride=2)
            oct1_ds = F.avg_pool2d(oct1, kernel_size=2, stride=2)

            persist_01 = torch.sqrt(F.relu(oct0_ds) * F.relu(oct1) + 1e-6)
            diff_01 = oct0_ds - oct1
            pd_01 = torch.cat([persist_01, diff_01], dim=1)

            persist_12 = torch.sqrt(F.relu(oct1_ds) * F.relu(oct2) + 1e-6)
            diff_12 = oct1_ds - oct2
            pd_12 = torch.cat([persist_12, diff_12], dim=1)

            pd_feat = torch.cat([
                F.adaptive_avg_pool2d(_cort_warp(pd_01), (2, 2)).flatten(1),
                F.adaptive_avg_pool2d(_cort_warp(pd_12), (2, 2)).flatten(1),
            ], dim=1)  # 192
            feats.append(0.75 * pd_feat / (pd_feat.norm(dim=1, keepdim=True) + 1e-6))

            env1 = torch.cat([oct1[:, 4:8], curv_maps[1][:, [2, 5, 10, 12]]], dim=1)
            env2 = torch.cat([oct2[:, 4:8], curv_maps[2][:, [2, 5, 10, 12]]], dim=1)
            cross_cov = torch.cat([
                _cross_gram(oct0_ds, env1),
                _cross_gram(oct1_ds, env2),
            ], dim=1)  # 192
            feats.append(0.75 * cross_cov / (cross_cov.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2G: Eccentricity & Bilateral Symmetry (54 dims) ---
            sym_feat = torch.cat([
                _eccentricity_and_symmetry(oct1),  # 27
                _eccentricity_and_symmetry(oct2),  # 27
            ], dim=1)  # 54

            feats.append(0.25 * sym_feat / (sym_feat.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2H: 5-Octave Surface Differential Geometry (128, 64, 32, 16, 8) ---
            lum16, rg16, by16 = F.avg_pool2d(lum, 2, 2), F.avg_pool2d(rg, 2, 2), F.avg_pool2d(by, 2, 2)
            lum8, rg8, by8 = F.avg_pool2d(lum16, 2, 2), F.avg_pool2d(rg16, 2, 2), F.avg_pool2d(by16, 2, 2)
            gm3 = _surface_geom_map(lum16, rg16, by16)
            gm4 = _surface_geom_map(lum8, rg8, by8)

            geom_s0_m = F.adaptive_avg_pool2d(_cort_warp(geom_maps[0]), (4, 4)).flatten(1)  # 224
            geom_s0_s = _std_pool2d(geom_maps[0], 3)  # 126
            geom_s1_m = F.adaptive_avg_pool2d(_cort_warp(geom_maps[1]), (4, 4)).flatten(1)  # 224
            geom_s1_s = _std_pool2d(geom_maps[1], 3)  # 126
            geom_s2_m = F.adaptive_avg_pool2d(_cort_warp(geom_maps[2]), (3, 3)).flatten(1)  # 126
            geom_s2_s = torch.cat([_std_pool2d(geom_maps[2], 2), geom_maps[2].mean(dim=(2, 3)), geom_maps[2].std(dim=(2, 3))], dim=1)  # 84
            geom_s3 = torch.cat([F.adaptive_avg_pool2d(gm3, (2, 2)).flatten(1), gm3.mean(dim=(2, 3)), gm3.std(dim=(2, 3))], dim=1)  # 84
            geom_s4 = torch.cat([gm4.mean(dim=(2, 3)), gm4.std(dim=(2, 3))], dim=1)  # 28

            for w_g, g_blk in (
                (1.50, geom_s0_m),
                (1.50, geom_s0_s),
                (1.15, geom_s1_m),
                (1.15, geom_s1_s),
                (0.85, geom_s2_m),
                (0.85, geom_s2_s),
                (0.55, geom_s3),
                (0.35, geom_s4),
            ):
                feats.append(w_g * g_blk / (g_blk.norm(dim=1, keepdim=True) + 1e-6))

            # --- Stage 2I: Multi-Octave Standardized Surface-Topology Relational Correlations & Symmetry ---
            g6_0 = torch.cat([
                geom_maps[0][:, 3:4] + geom_maps[0][:, 4:5],
                geom_maps[0][:, 0:1] + geom_maps[0][:, 1:2],
                geom_maps[0][:, 2:3],
                geom_maps[0][:, 5:6],
                geom_maps[0][:, 8:9],
                geom_maps[0][:, 9:10],
            ], dim=1)
            g6_1 = torch.cat([
                geom_maps[1][:, 3:4] + geom_maps[1][:, 4:5],
                geom_maps[1][:, 0:1] + geom_maps[1][:, 1:2],
                geom_maps[1][:, 2:3],
                geom_maps[1][:, 5:6],
                geom_maps[1][:, 8:9],
                geom_maps[1][:, 9:10],
            ], dim=1)
            g6_2 = torch.cat([
                geom_maps[2][:, 3:4] + geom_maps[2][:, 4:5],
                geom_maps[2][:, 0:1] + geom_maps[2][:, 1:2],
                geom_maps[2][:, 2:3],
                geom_maps[2][:, 5:6],
                geom_maps[2][:, 8:9],
                geom_maps[2][:, 9:10],
            ], dim=1)
            g6_3 = torch.cat([
                gm3[:, 3:4] + gm3[:, 4:5],
                gm3[:, 0:1] + gm3[:, 1:2],
                gm3[:, 2:3],
                gm3[:, 5:6],
                gm3[:, 8:9],
                gm3[:, 9:10],
            ], dim=1)
            c01 = _cross_gram(F.avg_pool2d(g6_0, kernel_size=2, stride=2), g6_1)  # 36
            c12 = _cross_gram(F.avg_pool2d(g6_1, kernel_size=2, stride=2), g6_2)  # 36
            geom_rel_fine = torch.cat([1.8 * c01, 1.0 * c12], dim=1)  # 72
            feats.append(0.75 * geom_rel_fine / (torch.sqrt(geom_rel_fine.pow(2).sum(dim=1, keepdim=True) + 3.0) + 1e-6))

            gc6_0 = torch.cat([
                geom_maps[0][:, 4:5] - geom_maps[0][:, 3:4],
                geom_maps[0][:, 0:1] - geom_maps[0][:, 1:2],
                geom_maps[0][:, 6:7],
                geom_maps[0][:, 7:8],
                geom_maps[0][:, 10:11],
                geom_maps[0][:, 11:12],
            ], dim=1)
            gc6_1 = torch.cat([
                geom_maps[1][:, 4:5] - geom_maps[1][:, 3:4],
                geom_maps[1][:, 0:1] - geom_maps[1][:, 1:2],
                geom_maps[1][:, 6:7],
                geom_maps[1][:, 7:8],
                geom_maps[1][:, 10:11],
                geom_maps[1][:, 11:12],
            ], dim=1)
            gc6_2 = torch.cat([
                geom_maps[2][:, 4:5] - geom_maps[2][:, 3:4],
                geom_maps[2][:, 0:1] - geom_maps[2][:, 1:2],
                geom_maps[2][:, 6:7],
                geom_maps[2][:, 7:8],
                geom_maps[2][:, 10:11],
                geom_maps[2][:, 11:12],
            ], dim=1)
            gc6_3 = torch.cat([
                gm3[:, 4:5] - gm3[:, 3:4],
                gm3[:, 0:1] - gm3[:, 1:2],
                gm3[:, 6:7],
                gm3[:, 7:8],
                gm3[:, 10:11],
                gm3[:, 11:12],
            ], dim=1)
            cc01 = _cross_gram(F.avg_pool2d(gc6_0, kernel_size=2, stride=2), gc6_1)  # 36
            cc12 = _cross_gram(F.avg_pool2d(gc6_1, kernel_size=2, stride=2), gc6_2)  # 36
            geom_comp_fine = torch.cat([1.8 * cc01, 1.0 * cc12], dim=1)  # 72
            feats.append(0.75 * geom_comp_fine / (torch.sqrt(geom_comp_fine.pow(2).sum(dim=1, keepdim=True) + 3.0) + 1e-6))

            w3 = _upper_tri_gram(g6_3, tri6_i, tri6_j)  # 15
            c23 = _cross_gram(F.avg_pool2d(g6_2, kernel_size=2, stride=2), g6_3)  # 36
            wc3 = _upper_tri_gram(gc6_3, tri6_i, tri6_j)  # 15
            cc23 = _cross_gram(F.avg_pool2d(gc6_2, kernel_size=2, stride=2), gc6_3)  # 36
            f3 = F.avg_pool2d(gm3[:, :12], kernel_size=2, stride=2).flatten(2)
            f4 = gm4[:, :12].flatten(2)
            f3n = (f3 - f3.mean(dim=2, keepdim=True)) / (f3.std(dim=2, keepdim=True) + 1e-5)
            f4n = (f4 - f4.mean(dim=2, keepdim=True)) / (f4.std(dim=2, keepdim=True) + 1e-5)
            c34_diag = (f3n * f4n).mean(dim=2)  # 12
            geom_rel = torch.cat([w3, wc3, c23, cc23, 0.5 * c34_diag], dim=1)  # 114
            feats.append(0.50 * geom_rel / (torch.sqrt(geom_rel.pow(2).sum(dim=1, keepdim=True) + 8.0) + 1e-6))

            gsym = torch.cat([
                1.3 * _geom_sym_scale(geom_maps[1], 12),  # 24
                1.0 * _geom_sym_scale(geom_maps[2], 12),  # 24
                0.7 * _geom_sym_scale(gm3, 11),           # 22
            ], dim=1)  # 70
            feats.append(0.30 * gsym / (torch.sqrt(gsym.pow(2).sum(dim=1, keepdim=True) + 1.2) + 1e-6))

            out_chunks.append(torch.cat(feats, dim=1))

        return torch.cat(out_chunks, dim=0)


MODEL = Model()
