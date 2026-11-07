"""TFS-IF v3: strong envelope substrate + analytic-signal TFS additions.

Paradigm: temporal fine structure / FM (phase, instantaneous frequency).

Architecture:
  BASE: multi-window log + cube-root cochleagrams (windows 1024, 2048, 4096) +
        multi-tau dynamics. Gives the ridge a clean envelope substrate.
  TFS:  analytic signal on the 4096 window via the Hilbert transform ->
        instantaneous frequency (IF), IF deviation from band fc, instantaneous
        bandwidth (std of IF), FM slope (up/down), envelope-IF coupling,
        inter-band phase locking (PLV, gap 1 and 2), and ENVELOPE-MODULATED
        TFS products (genuinely nonlinear in the bands).

Closed-form. Linear ridge is the only fit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data import SR
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Filterbank helpers
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_power(win, n_filters=32, f_min=80.0, f_max=6000.0, device="cuda"):
    """Real-valued gammatone-magnitude power bank for a given window length."""
    freqs = torch.fft.rfftfreq(win, d=1.0 / SR).to(device)
    fcs_np = erb_space(f_min, f_max, n_filters)
    fcs = torch.from_numpy(fcs_np.astype(np.float32)).to(device)
    bw = torch.from_numpy((1.019 * erb(fcs_np)).astype(np.float32)).to(device)
    H = (1.0 + ((freqs[:, None] - fcs[None, :]) / bw[None, :]) ** 2) ** (-4)
    H = H / H.max(dim=0, keepdim=True).values.clamp(min=1e-12)
    return H, fcs, bw


def leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    alpha = 1.0 / (1.0 + tau_frames)
    K = max(int(6 * tau_frames) + 1, 4)
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = alpha * (1.0 - alpha) ** t
    xp = x.transpose(0, 1).unsqueeze(0)
    xp_p = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.flip(0).view(1, 1, -1).expand(x.shape[1], 1, K).contiguous()
    y = torch.nn.functional.conv1d(xp_p, kk, groups=x.shape[1])
    return y.squeeze(0).transpose(0, 1).contiguous()


def acausal_leaky_lowpass(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    return torch.flip(leaky_lowpass(torch.flip(x, dims=[0]), tau_frames), dims=[0])


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0

    N_BANDS = 32
    N_BANDS_ENV = 48
    F_MIN = 80.0
    F_MAX = 6000.0
    WIN_LIST = (1024, 2048, 4096)
    PHASE_LOCK_HZ = 1500.0
    INNER = 2048
    TAUS = (15.0, 60.0)
    F0_CANDS = (75.0, 95.0, 115.0, 140.0, 170.0, 200.0, 235.0, 275.0, 320.0, 380.0, 450.0, 540.0)
    N_HARMONICS = 6

    def __init__(self, device: str = "cuda"):
        self.device = device
        # --- envelope filterbanks per window (richer 48-band) ---
        self.fbs = []
        for w in self.WIN_LIST:
            H, fcs, bw = gammatone_power(w, n_filters=self.N_BANDS_ENV,
                                          f_min=self.F_MIN, f_max=self.F_MAX,
                                          device=device)
            self.fbs.append(H)
        # --- analytic-signal filterbank for 4096 window ---
        H4, fcs4, bw4 = gammatone_power(self.win, n_filters=self.N_BANDS,
                                          f_min=self.F_MIN, f_max=self.F_MAX,
                                          device=device)
        n_freqs = self.win // 2 + 1
        factor = torch.full((n_freqs,), 2.0, device=device)
        factor[0] = 1.0
        factor[-1] = 1.0
        self.H_analytic = H4 * factor[:, None]  # (n_freqs, n_bands)
        self.fcs = fcs4
        self.bw_norm = bw4
        self.log_fcs = torch.log(fcs4)
        self.n_freqs = n_freqs
        self.hann_w = {w: torch.hann_window(w, periodic=True, device=device) for w in self.WIN_LIST}
        # Pre-mask phase-locking range
        self.pl_mask = (fcs4 <= self.PHASE_LOCK_HZ).float()
        # Sample-index time vector for FM-slope regression (zero-mean)
        ti = torch.arange(self.INNER - 1, device=device, dtype=torch.float32)
        ti = ti - ti.mean()
        self.ti = ti
        self.ti_sq_sum = float((ti ** 2).sum().item())
        # F0 candidates and per-band harmonic-coincidence target table
        f0_cands = torch.tensor(self.F0_CANDS, dtype=torch.float32, device=device)
        self.f0_cands = f0_cands
        # For each (f0, band fc), compute the closest harmonic ratio and the deviation
        # in cents from that harmonic. We use this every call after IF is known.
        # Tolerance per band scales with the band's bandwidth (in Hz).
        self.harm_tol_hz = self.bw_norm.clone()  # (B,)

    @torch.no_grad()
    def _envelope_block(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-window log + cube-root cochleagrams + multi-tau, on (T, 4096) audio."""
        # x: (T, 4096) real
        blocks = []
        for w, fb in zip(self.WIN_LIST, self.fbs):
            if w == self.win:
                xw = x
            else:
                start = (self.win - w) // 2
                xw = x[:, start:start + w]
            hann = self.hann_w[w]
            Xf = torch.fft.rfft(xw * hann, n=w, dim=1)
            P = (Xf.real ** 2 + Xf.imag ** 2)  # (T, n_freqs)
            E = P @ fb  # (T, n_bands)
            logE = torch.log(E.clamp(min=1e-10))
            cubeE = torch.pow(E.clamp(min=1e-10), 1.0 / 3.0)
            blocks.append(logE)
            blocks.append(cubeE)
        envelope = torch.cat(blocks, dim=1)  # (T, n_windows * 2 * n_bands = 192)
        # Multi-tau deltas
        feats = [envelope]
        for tau in self.TAUS:
            feats.append(envelope - leaky_lowpass(envelope, tau_frames=tau))
        # Acausal (future) for long-window log only
        long_logE = blocks[-2]  # the 4096 log
        feats.append(acausal_leaky_lowpass(long_logE, tau_frames=60.0) - long_logE)
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def _analytic_signal_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Tb, 4096). Returns (Tb, n_bands, 4096) complex64."""
        hann = self.hann_w[self.win]
        Xf = torch.fft.rfft(x * hann, n=self.win, dim=1)  # (Tb, n_freqs)
        Yp = Xf.unsqueeze(1) * self.H_analytic.transpose(0, 1).unsqueeze(0)  # (Tb, n_bands, n_freqs)
        Z_full = torch.zeros((Yp.shape[0], self.N_BANDS, self.win),
                              dtype=torch.complex64, device=self.device)
        Z_full[..., :self.n_freqs] = Yp
        z = torch.fft.ifft(Z_full, n=self.win, dim=2)
        return z

    @torch.no_grad()
    def _tfs_block(self, x: torch.Tensor) -> torch.Tensor:
        """Compute TFS features per frame. x: (T, win). Returns (T, F_tfs)."""
        T = x.shape[0]
        Tb = 128
        out_list = []
        i0 = (self.win - self.INNER) // 2
        pl = self.pl_mask.unsqueeze(0)
        for s in range(0, T, Tb):
            e = min(s + Tb, T)
            z = self._analytic_signal_batch(x[s:e])  # (Tb, B, win) complex
            zc = z[..., i0:i0 + self.INNER]  # (Tb, B, INNER)
            amp = zc.abs()  # (Tb, B, INNER)
            E = amp.mean(dim=2)  # (Tb, B)

            # IF via phase-difference
            zr = zc[..., 1:] * zc[..., :-1].conj()
            ifreq = torch.angle(zr) * (SR / (2.0 * np.pi))  # (Tb, B, INNER-1) Hz
            w = amp[..., 1:]
            wsum = w.sum(dim=2).clamp(min=1e-10)
            if_mean = (w * ifreq).sum(dim=2) / wsum
            df = ifreq - if_mean.unsqueeze(2)
            if_var = (w * df ** 2).sum(dim=2) / wsum
            if_std = torch.sqrt(if_var.clamp(min=0.0) + 1e-6)

            # IF deviation from filter center, normalized
            if_dev = if_mean - self.fcs.unsqueeze(0)
            if_dev_norm = if_dev / self.bw_norm.unsqueeze(0)
            log_if = torch.log(if_mean.clamp(min=20.0))

            # FM slope: weighted regression of ifreq on ti
            num = (w * df * self.ti).sum(dim=2)
            denom = (w * self.ti ** 2).sum(dim=2).clamp(min=1e-6)
            fm_slope = num / denom  # Hz/sample
            fm_hz_s = fm_slope * SR
            fm_feat = torch.sign(fm_hz_s) * torch.log1p(fm_hz_s.abs())
            fm_up = fm_feat.clamp(min=0.0)
            fm_dn = (-fm_feat).clamp(min=0.0)

            # Envelope-IF coupling (within-window correlation)
            env_mid = amp[..., :-1]
            env_dm = env_mid - env_mid.mean(dim=2, keepdim=True)
            if_dm = df
            cov_env_if = (env_dm * if_dm).mean(dim=2)
            env_std = env_mid.std(dim=2).clamp(min=1e-3)
            corr_env_if = (cov_env_if / (env_std * if_std + 1e-3)).clamp(-1.0, 1.0)

            # Inter-band PLV (gap 1 and gap 2)
            unit = zc / zc.abs().clamp(min=1e-8)
            plv1 = (unit[:, :-1, :] * unit[:, 1:, :].conj()).mean(dim=2).abs()
            plv2 = (unit[:, :-2, :] * unit[:, 2:, :].conj()).mean(dim=2).abs()
            plv1 = torch.cat([plv1, plv1[:, -1:]], dim=1)
            plv2 = torch.cat([plv2, plv2[:, -2:]], dim=1)

            # Apply phase-locking mask
            log_if_pl = log_if * pl
            if_dev_norm_pl = if_dev_norm * pl
            if_std_pl = if_std * pl
            fm_up_pl = fm_up * pl
            fm_dn_pl = fm_dn * pl
            corr_env_if_pl = corr_env_if * pl
            plv1_pl = plv1 * pl
            plv2_pl = plv2 * pl

            # Envelope-modulated TFS products (genuinely nonlinear in the bands)
            logE = torch.log(E.clamp(min=1e-10))
            logE_rect = logE - logE.amin(dim=1, keepdim=True)
            prod_E_ifdev = logE_rect * if_dev_norm_pl
            prod_E_fm = logE_rect * (fm_up_pl - fm_dn_pl)
            prod_E_plv = logE_rect * plv1_pl

            # --- harmonic-coincidence from IF across bands (closed-form pitch) ---
            # For each F0 candidate f0, for each band, the nearest harmonic ratio is
            # round(IF / f0). Coincidence score = exp(-((IF - round(IF/f0)*f0) / tol)^2)
            # weighted by envelope, summed across phase-locked bands.
            # if_mean: (Tb, B); f0_cands: (F0,)
            f0_cands = self.f0_cands.view(1, 1, -1)  # (1,1,F0)
            ifm = if_mean.unsqueeze(2)  # (Tb, B, 1)
            ratios = ifm / f0_cands  # (Tb, B, F0)
            nearest_k = torch.round(ratios).clamp(min=1.0, max=20.0)
            harm_predict = nearest_k * f0_cands  # (Tb, B, F0)
            dev_hz = (ifm - harm_predict).abs()  # (Tb, B, F0)
            tol = self.harm_tol_hz.view(1, -1, 1)  # (1,B,1)
            coincidence = torch.exp(-(dev_hz / tol) ** 2)  # (Tb, B, F0)
            # Envelope-weighted, restrict to phase-locking bands
            env_w = (E * self.pl_mask.unsqueeze(0)).unsqueeze(2)  # (Tb, B, 1)
            harm_pitch = (coincidence * env_w).sum(dim=1) / (env_w.sum(dim=1) + 1e-10)
            # (Tb, F0)
            best_f0_strength, best_f0_idx = harm_pitch.max(dim=1, keepdim=True)
            log_best_f0 = torch.log(self.f0_cands[best_f0_idx.squeeze(1)].unsqueeze(1).clamp(min=20.0))
            # Reduce F0 candidates to 12-dim feature (already (Tb,12))

            # --- cross-band envelope products: log E_i * log E_j for adjacent pairs ---
            # Compact: 6 broad bands by averaging consecutive 5-6 bands.
            logE_compact = torch.zeros(E.shape[0], 6, device=self.device)
            chunks = torch.tensor_split(torch.arange(self.N_BANDS, device=self.device), 6)
            for ci, idx in enumerate(chunks):
                logE_compact[:, ci] = torch.log(E[:, idx].mean(dim=1).clamp(min=1e-10))
            # 5 adjacent + 4 skip-1 + 3 skip-2 = 12 pairs
            xb_pairs = []
            for a, b in [(0,1),(1,2),(2,3),(3,4),(4,5),(0,2),(1,3),(2,4),(3,5),(0,3),(1,4),(2,5)]:
                xb_pairs.append((logE_compact[:, a] * logE_compact[:, b]).unsqueeze(1))
            xband_prod = torch.cat(xb_pairs, dim=1)  # (Tb, 12)

            block = torch.cat([
                log_if_pl,        # 32
                if_dev_norm_pl,   # 32
                if_std_pl,        # 32
                fm_up_pl,         # 32
                fm_dn_pl,         # 32
                corr_env_if_pl,   # 32
                plv1_pl,          # 32
                plv2_pl,          # 32
                prod_E_ifdev,     # 32
                prod_E_fm,        # 32
                prod_E_plv,       # 32
                harm_pitch,       # 12 harmonic-coincidence pitch salience
                best_f0_strength, # 1
                log_best_f0,      # 1
                xband_prod,       # 12 cross-band envelope products
            ], dim=1)  # 11 * 32 + 26 = 378
            out_list.append(block.cpu())
            del z, zc, amp, zr, ifreq, w, env_mid, unit, plv1, plv2
        tfs = torch.cat(out_list, dim=0).to(self.device)
        # Multi-tau on TFS
        feats = [tfs]
        for tau in self.TAUS:
            feats.append(tfs - leaky_lowpass(tfs, tau_frames=tau))
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        env = self._envelope_block(x)   # base substrate
        tfs = self._tfs_block(x)        # TFS additions
        feat = torch.cat([env, tfs], dim=1).float().cpu().numpy()
        return feat


model_shorthand_name = "TFS-IF-v4"
model_description = (
    "v4 = v3 + HARMONIC-COINCIDENCE pitch (across-band sum of envelope-weighted "
    "Gaussian distance between IF and nearest f0-multiple per F0 candidate, 12 "
    "F0s) and CROSS-BAND log envelope PRODUCTS (12 pairs across 6 broadband "
    "compactions). Also widened envelope cochleagram to 48 bands. v3=0.0591."
)


def build_embedder(device: str = "cuda") -> WaveEmbedder:
    return WaveEmbedder(device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available -- falling back to cpu")
        args.device = "cpu"

    t0 = time.time()
    cfg = EncodingConfig(num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = int(getattr(embedder, "n_params", 0))

    upsert_overall_results(
        [make_result_row(r, model_shorthand_name, n_params, model_description)], RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"subject:        {r['subject']}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['train_corr']:.4f}, "
          f"frac_elec>0.1={r['frac_test_elec_above_0.1']:.4f})")
    print(f"n_params:       {n_params:.2e}")
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
