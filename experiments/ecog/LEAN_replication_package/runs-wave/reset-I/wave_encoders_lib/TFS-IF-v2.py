"""TFS-IF v1: Temporal Fine Structure / FM-slope analytic-signal front-end.

Paradigm: instead of the envelope-only cochleagram (which a ridge can already
reconstruct from log-power bands), extract NONLINEAR quantities derived from
the per-band analytic signal:

  - instantaneous frequency (IF, in Hz) from phase derivative
  - instantaneous bandwidth (std of IF inside the window)
  - FM sweep slope (linear regression of IF over the window)
  - FM up/down rectified detectors
  - IF skewness
  - envelope-IF coupling (cov within window)
  - inter-band phase coherence (PLV)
  - envelope-to-TFS ratio

The analytic signal is built via the Hilbert transform realized as
  z = ifft( H_band * 2*rfft(x), n=win )
on each 4096-sample Hann-windowed frame; the central half of each window is
used to avoid window edges.

Closed-form, no learned parameters. Linear ridge is the only fit.
"""
from __future__ import annotations

import argparse
import hashlib
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_I"
os.makedirs(CACHE_DIR, exist_ok=True)


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


# ---------------------------------------------------------------------------
# Embedder: TFS / Instantaneous-Frequency front-end
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0

    # Cochlear filterbank
    N_BANDS = 32
    F_MIN = 80.0
    F_MAX = 6000.0
    # We restrict TFS to phase-locking range (< ~1.5 kHz). Above that we keep
    # envelope only (the analytic-signal IF is computed everywhere but the
    # interpretable phase-locking quantities are concentrated below).
    PHASE_LOCK_HZ = 1500.0
    # Center-of-window analysis: use middle 2048 samples to avoid Hann edges.
    INNER = 2048
    TAUS = (15.0, 60.0)

    def __init__(self, device: str = "cuda"):
        self.device = device

        win = self.win
        n_freqs = win // 2 + 1
        freqs = torch.fft.rfftfreq(win, d=1.0 / SR).to(device)  # (n_freqs,)
        fcs = torch.from_numpy(erb_space(self.F_MIN, self.F_MAX, self.N_BANDS).astype(np.float32)).to(device)
        # Narrower-than-ERB bandwidth so adjacent bands don't bleed IF estimates
        bw = 1.019 * torch.from_numpy(erb(fcs.cpu().numpy()).astype(np.float32)).to(device)
        # Gammatone-magnitude shape, normalized to unit peak
        H = (1.0 + ((freqs[:, None] - fcs[None, :]) / bw[None, :]) ** 2) ** (-4)  # (n_freqs, n_bands)
        H = H / H.max(dim=0, keepdim=True).values.clamp(min=1e-12)
        # Analytic factor: positive-freq bins doubled, DC/Nyquist kept
        factor = torch.full((n_freqs,), 2.0, device=device)
        factor[0] = 1.0
        factor[-1] = 1.0
        H = H * factor[:, None]  # (n_freqs, n_bands)
        self.H = H  # (n_freqs, n_bands)
        self.fcs = fcs  # (n_bands,) center freqs (Hz)
        self.log_fcs = torch.log(fcs)
        self.n_freqs = n_freqs
        self.hann = torch.hann_window(win, periodic=True, device=device)
        # Mask for phase-locking range (1 below 1.5 kHz, 0 above)
        self.pl_mask = (fcs <= self.PHASE_LOCK_HZ).float()  # (n_bands,)
        # Sample index for FM-slope regression in inner window
        ti = torch.arange(self.INNER, device=device, dtype=torch.float32)
        ti = ti - ti.mean()
        self.ti = ti
        self.ti_var = float((ti ** 2).sum().item())
        # Channel index for centroid-style features
        self.ch_idx = torch.arange(self.N_BANDS, device=device, dtype=torch.float32)
        # Precomputed bandwidth (Hz) per band for IF-deviation normalization
        self.bw_norm = bw.clone()

    @torch.no_grad()
    def _analytic_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, win) real. Returns z: (T, n_bands, win) complex64 analytic signals."""
        # Hann window x to suppress edge artifacts before Hilbert
        Xf = torch.fft.rfft(x * self.hann, n=self.win, dim=1)  # (T, n_freqs)
        # Per band: rfft * H -> embed in full complex spectrum length win, zeros for
        # negative freqs. Then ifft of length win = analytic signal.
        Yp = Xf.unsqueeze(1) * self.H.transpose(0, 1).unsqueeze(0)  # (T, n_bands, n_freqs)
        Z_full = torch.zeros((Yp.shape[0], self.N_BANDS, self.win),
                              dtype=torch.complex64, device=self.device)
        Z_full[..., :self.n_freqs] = Yp
        z = torch.fft.ifft(Z_full, n=self.win, dim=2)  # (T, n_bands, win) complex
        return z

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        T = x.shape[0]

        # Process in mini-batches of frames to bound memory: (Tb, n_bands, win) complex
        Tb = 64  # 64 frames * 32 bands * 4096 complex64 = 64MB per batch
        feats_per_band = []  # list per batch of (Tb, F)
        for s in range(0, T, Tb):
            e = min(s + Tb, T)
            z = self._analytic_chunks(x[s:e])  # (Tb, n_bands, win) complex

            # Use central window only
            i0 = (self.win - self.INNER) // 2
            zc = z[..., i0:i0 + self.INNER]  # (Tb, n_bands, INNER)

            # --- envelope, log envelope ---
            amp = zc.abs()  # (Tb, n_bands, INNER)
            E = amp.mean(dim=2)  # (Tb, n_bands)
            logE = torch.log(E.clamp(min=1e-10))

            # --- instantaneous frequency via phase-difference ---
            # IF[t] = arg(z[t+1] * conj(z[t])) * SR / (2 pi)  (Hz)
            zr = zc[..., 1:] * zc[..., :-1].conj()  # (Tb, n_bands, INNER-1)
            ifreq = torch.angle(zr) * (SR / (2.0 * np.pi))  # Hz

            # Weight IF stats by envelope (more reliable where amplitude is large)
            w = amp[..., 1:]  # (Tb, n_bands, INNER-1)
            w_sum = w.sum(dim=2).clamp(min=1e-10)  # (Tb, n_bands)
            if_mean = (w * ifreq).sum(dim=2) / w_sum  # (Tb, n_bands)
            df = ifreq - if_mean.unsqueeze(2)
            if_var = (w * df ** 2).sum(dim=2) / w_sum
            if_std = torch.sqrt(if_var.clamp(min=0.0) + 1e-6)
            # IF skew (third central moment / std^3)
            if_skew = (w * df ** 3).sum(dim=2) / (w_sum * (if_std ** 3 + 1e-6))
            if_skew = if_skew.clamp(-10, 10)

            # FM slope via weighted linear regression of IF vs sample-time across inner window.
            ti_seg = self.ti[1:]  # (INNER-1,) zero-mean already (approx)
            # Use unweighted regression for stability
            num = (df * ti_seg).sum(dim=2)
            denom = (ti_seg ** 2).sum() + 1e-6
            fm_slope = num / denom  # (Tb, n_bands)  Hz/sample

            # Convert FM slope to Hz/s and log-scale-friendly signed sqrt
            fm_slope_hz_s = fm_slope * SR
            sgn = torch.sign(fm_slope_hz_s)
            fm_slope_feat = sgn * torch.log1p(fm_slope_hz_s.abs())
            fm_up = fm_slope_feat.clamp(min=0.0)
            fm_dn = (-fm_slope_feat).clamp(min=0.0)

            # --- IF deviation from center (fc-relative pitch shift inside each band) ---
            if_dev = if_mean - self.fcs.unsqueeze(0)  # (Tb, n_bands) Hz
            log_if = torch.log(if_mean.clamp(min=20.0))  # (Tb, n_bands)
            # Normalize deviation by bandwidth so it's comparable across channels
            if_dev_norm = if_dev / self.bw_norm.unsqueeze(0)

            # Mask TFS quantities by phase-locking band — they only carry meaning <~1.5 kHz
            pl = self.pl_mask.unsqueeze(0)
            log_if_pl = log_if * pl
            if_dev_norm_pl = if_dev_norm * pl
            if_std_pl = if_std * pl
            if_skew_pl = if_skew * pl
            fm_up_pl = fm_up * pl
            fm_dn_pl = fm_dn * pl
            # bandwidth in log octaves
            log_if_std = torch.log(if_std.clamp(min=0.1)) * pl

            # --- envelope-IF coupling (within-window correlation, sign matters) ---
            # Cov(env, if) weighted by env normalization
            env_mid = amp[..., :-1]  # align with ifreq length
            env_dm = env_mid - env_mid.mean(dim=2, keepdim=True)
            if_dm = ifreq - if_mean.unsqueeze(2)
            cov_env_if = (env_dm * if_dm).mean(dim=2)  # (Tb, n_bands)
            env_std_w = env_mid.std(dim=2)
            corr_env_if = cov_env_if / (env_std_w * if_std + 1e-3)
            corr_env_if = corr_env_if.clamp(-1.0, 1.0) * pl

            # --- inter-band phase coherence (PLV with next band) ---
            # PLV[t] = | mean_t e^{i (phi_b - phi_{b+1})} |
            #       = | mean(z_b * conj(z_{b+1})) / (mean|z_b| mean|z_{b+1}|) | -> use unit phasors
            unit = zc / (amp.unsqueeze(-1).squeeze(-1) if amp.shape == zc.shape else zc.abs()).clamp(min=1e-8)
            # easier: unit = zc / |zc|
            unit = zc / zc.abs().clamp(min=1e-8)
            pair = unit[:, :-1, :] * unit[:, 1:, :].conj()  # (Tb, n_bands-1, INNER)
            plv = pair.mean(dim=2).abs()  # (Tb, n_bands-1)
            # Pad to n_bands
            plv = torch.cat([plv, plv[:, -1:]], dim=1)  # (Tb, n_bands)
            # Restrict PLV to phase-locking pairs
            plv_pl = plv * pl

            # --- envelope-to-TFS ratio: envelope std vs total amp ---
            # Slow envelope: low-passed |z| in time; TFS energy: high-pass residual.
            # Approximate via differences of |z| inside the window.
            # env_slope = std of |z| inside the window; high values => strong amp modulation.
            env_var_in_win = env_mid.var(dim=2)
            ratio_env_tfs = torch.log(env_var_in_win.clamp(min=1e-12)) - 2 * torch.log(E.clamp(min=1e-6))

            # --- sub-frame envelope stats: split inner window into K sub-blocks ---
            # This captures fast (~100 Hz) amplitude modulation WITHIN the frame.
            K = 8
            assert self.INNER % K == 0
            sub_len = self.INNER // K
            amp_sub = amp.reshape(amp.shape[0], self.N_BANDS, K, sub_len).mean(dim=3)  # (Tb, B, K)
            log_amp_sub = torch.log(amp_sub.clamp(min=1e-10))  # log energies per sub-block
            # First-difference across sub-blocks: fast onset/offset within frame
            log_amp_diff = log_amp_sub[..., 1:] - log_amp_sub[..., :-1]   # (Tb, B, K-1)
            # Reduce to: max amp sub, min amp sub, max-min range, max pos diff, max neg diff
            sub_max, _ = log_amp_sub.max(dim=2)
            sub_min, _ = log_amp_sub.min(dim=2)
            sub_range = sub_max - sub_min
            pos_diff_max = log_amp_diff.clamp(min=0.0).amax(dim=2)
            neg_diff_max = (-log_amp_diff).clamp(min=0.0).amax(dim=2)
            sub_std = log_amp_sub.std(dim=2)

            # --- nonlinear PRODUCT features: envelope × TFS quantities ---
            # logE rectified to be non-negative (shifted), then multiplied with TFS scalars.
            logE_rect = (logE - logE.amin(dim=1, keepdim=True))  # 0 at quietest band
            prod_E_ifdev = logE_rect * if_dev_norm_pl
            prod_E_fm_up = logE_rect * fm_up_pl
            prod_E_fm_dn = logE_rect * fm_dn_pl
            prod_E_ifstd = logE_rect * if_std_pl
            prod_E_corrEI = logE_rect * corr_env_if
            prod_E_plv = logE_rect * plv_pl

            # --- extended PLV: skip gaps 2, 4 — formant/harmonic coupling ---
            plv_g2 = (unit[:, :-2, :] * unit[:, 2:, :].conj()).mean(dim=2).abs()
            plv_g4 = (unit[:, :-4, :] * unit[:, 4:, :].conj()).mean(dim=2).abs()
            # Pad both to n_bands
            plv_g2 = torch.cat([plv_g2, plv_g2[:, -2:]], dim=1)
            plv_g4 = torch.cat([plv_g4, plv_g4[:, -4:]], dim=1)
            plv_g2 = plv_g2 * pl
            plv_g4 = plv_g4 * pl

            # --- assemble per-batch feature block: (Tb, F_band) ---
            block = torch.cat([
                logE,             # 32  envelope (base)
                log_if_pl,        # 32  log instantaneous freq (TFS)
                if_dev_norm_pl,   # 32  normalized IF deviation from fc
                if_std_pl,        # 32  instantaneous bandwidth
                fm_up_pl,         # 32  FM upward sweeps
                fm_dn_pl,         # 32  FM downward sweeps
                corr_env_if,      # 32  envelope-IF coupling
                plv_pl,           # 32  inter-band phase coherence (gap 1)
                sub_range,        # 32  sub-frame range
                pos_diff_max,     # 32  fastest within-frame rise
                neg_diff_max,     # 32  fastest within-frame fall
                prod_E_ifdev,     # 32  envelope-modulated IF deviation
                prod_E_fm_up,     # 32  envelope-modulated FM up
                prod_E_fm_dn,     # 32  envelope-modulated FM down
                prod_E_plv,       # 32  envelope-modulated PLV
                plv_g2,           # 32  PLV at band gap 2
            ], dim=1)  # 16 * 32 = 512
            feats_per_band.append(block.cpu())

            del z, zc, amp, zr, ifreq, w, unit, pair

        feat_band = torch.cat(feats_per_band, dim=0).to(self.device)  # (T, 352)

        # Multi-tau dynamics on the band features
        all_feats = [feat_band]
        for tau in self.TAUS:
            sm = leaky_lowpass(feat_band, tau_frames=tau)
            all_feats.append(feat_band - sm)

        # Indices into the 16-block feat_band: logE(0), log_if_pl(1), if_dev_norm_pl(2),
        # if_std_pl(3), fm_up_pl(4), fm_dn_pl(5), corr_env_if(6), plv_pl(7), sub_range(8),
        # pos_diff_max(9), neg_diff_max(10), prod_E_ifdev(11), prod_E_fm_up(12),
        # prod_E_fm_dn(13), prod_E_plv(14), plv_g2(15)
        fm_up_block = feat_band[:, 4 * 32:5 * 32]
        fm_dn_block = feat_band[:, 5 * 32:6 * 32]
        fm_up_sum = fm_up_block.sum(dim=1, keepdim=True)
        fm_dn_sum = fm_dn_block.sum(dim=1, keepdim=True)
        fm_balance = fm_up_sum - fm_dn_sum
        all_feats.append(torch.cat([fm_up_sum, fm_dn_sum, fm_balance], dim=1))

        # PLV summary: mean across bands (low/mid only)
        plv_block = feat_band[:, 7 * 32:8 * 32]
        plv_lo = plv_block[:, :8].mean(dim=1, keepdim=True)
        plv_mid = plv_block[:, 8:16].mean(dim=1, keepdim=True)
        plv_all = plv_block.mean(dim=1, keepdim=True)
        all_feats.append(torch.cat([plv_lo, plv_mid, plv_all], dim=1))


        feat = torch.cat(all_feats, dim=1).float().cpu().numpy()
        return feat


model_shorthand_name = "TFS-IF-v2"
model_description = (
    "v2 trimmed: 16 blocks x 32 bands (logE, log_IF, IF_dev, IF_std, FM_up, "
    "FM_dn, corr_env-IF, PLV gap1, sub_range, pos/neg_diff_max within frame, "
    "envelope*{IF_dev,FM_up,FM_dn,PLV} products, PLV gap2). Multi-tau (15,60). "
    "Drops the noisy IF_skew, log_if_std, ratio_env_tfs, sub_max/min/std, "
    "prod_E_ifstd/corrEI, plv_g4 from initial attempt. v1=0.0556."
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
