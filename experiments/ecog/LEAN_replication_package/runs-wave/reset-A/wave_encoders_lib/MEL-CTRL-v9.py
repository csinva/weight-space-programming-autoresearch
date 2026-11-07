"""
DIAGNOSTIC CONTROL (v9-MEL): pure multi-window mel-spec pipeline, NO biophysical
periphery (no DRNL, no IHC, no AN). Establishes how much of the score comes from
LEAN-style features alone. Compared against the biophys v1-v8 results to see
whether the periphery is helping or hurting in this codebase.

If this matches or beats biophys v7 (0.0650), the periphery is currently dead
weight and v10 needs to redesign the periphery before adding it back.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data import SR, HOP, WINMAX
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (pts / 21.4) - 1.0) * 1000.0 / 4.37


def causal_ema(x: torch.Tensor, tau_frames: float) -> torch.Tensor:
    T, C = x.shape
    alpha = 1.0 / (1.0 + tau_frames)
    K = min(T, max(int(6 * tau_frames) + 1, 4))
    t = torch.arange(K, device=x.device, dtype=x.dtype)
    kernel = (alpha * (1.0 - alpha) ** t).flip(0)
    xp = x.T.unsqueeze(0)
    xp = torch.nn.functional.pad(xp, (K - 1, 0), mode='replicate')
    kk = kernel.view(1, 1, K).expand(C, 1, K).contiguous()
    return torch.nn.functional.conv1d(xp, kk, groups=C).squeeze(0).T


def stmf_gabor_2d(log_e: torch.Tensor, omega_t: float, omega_f: float,
                  fs: float = 30.0) -> torch.Tensor:
    period_t = fs / max(abs(omega_t), 0.5)
    sigma_t = period_t / (2 * np.pi)
    Lt = max(1, int(np.ceil(3 * sigma_t)))
    sigma_f = 1.0 / max(abs(omega_f), 0.1) / (2 * np.pi)
    Lf = max(1, int(np.ceil(3 * sigma_f)))
    t = np.arange(-Lt, Lt + 1); f = np.arange(-Lf, Lf + 1)
    g = np.exp(-((t[:, None] ** 2) / (2 * sigma_t ** 2)
                  + (f[None, :] ** 2) / (2 * sigma_f ** 2)))
    phase = 2 * np.pi * omega_t * t[:, None] / fs + 2 * np.pi * omega_f * f[None, :]
    cos_k = (g * np.cos(phase)).astype(np.float32); sin_k = (g * np.sin(phase)).astype(np.float32)
    cos_k -= cos_k.mean(); sin_k -= sin_k.mean()
    cos_t = torch.from_numpy(cos_k).to(log_e.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
    sin_t = torch.from_numpy(sin_k).to(log_e.device).view(1, 1, 2 * Lt + 1, 2 * Lf + 1)
    xp = log_e.unsqueeze(0).unsqueeze(0)
    c = torch.nn.functional.conv2d(xp, cos_t, padding=(Lt, Lf))
    s = torch.nn.functional.conv2d(xp, sin_t, padding=(Lt, Lf))
    T, F = log_e.shape
    return torch.sqrt(c * c + s * s + 1e-12).squeeze(0).squeeze(0)[:T, :F]


def channel_dog(x: torch.Tensor, sig_c: float, sig_s: float) -> torch.Tensor:
    k = int(np.ceil(3 * sig_s))
    coords = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g_c = torch.exp(-(coords ** 2) / (2 * sig_c ** 2)); g_c /= g_c.sum()
    g_s = torch.exp(-(coords ** 2) / (2 * sig_s ** 2)); g_s /= g_s.sum()
    dog = (g_c - g_s).view(1, 1, -1)
    return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)


# ---------------------------------------------------------------------------

class MelControlV9:
    win = WINMAX
    n_params = 0

    N_BM = 32
    F_MIN = 80.0
    F_MAX = 8000.0

    LONG_TAUS = (9.0, 30.0, 90.0)
    F0_CANDS = tuple(np.geomspace(70.0, 500.0, 14).tolist())

    STMF_RATES = (
        (2.0, 0.25), (2.0, -0.25),
        (4.0, 0.25), (4.0, -0.25),
        (4.0, 0.5),  (4.0, -0.5),
        (8.0, 0.25), (8.0, -0.25),
        (8.0, 0.5),  (8.0, -0.5),
        (16.0, 0.25), (16.0, -0.25),
        (0.0, 0.25), (0.0, 0.5),
    )

    def __init__(self, device: str = "cuda"):
        self.device = device
        fcs = erb_space(self.F_MIN, self.F_MAX, self.N_BM).astype(np.float32)
        self.fcs = torch.from_numpy(fcs).to(device)
        bw_g = 1.019 * 24.7 * (4.37 * self.fcs / 1000.0 + 1.0)
        self.bw_bm = bw_g
        # Pre-build mel filter banks for each window size
        self._H = {}
        for w in (1024, 2048, 4096):
            freqs = torch.fft.rfftfreq(w, d=1.0 / SR).to(device)
            df = freqs[None, :] - self.fcs[:, None]
            bw_w = self.bw_bm * (1.0 + 800.0 / w)
            H = (1.0 + (df / bw_w[:, None]) ** 2) ** (-2.0)
            self._H[w] = (H, freqs)
        # Harmonic indices on the 4096-window filterbank
        self.harm_idx = []
        for f0 in self.F0_CANDS:
            h_idx = []
            for k in range(1, 7):
                fh = f0 * k
                if fh > self.F_MAX:
                    break
                h_idx.append(int(torch.argmin((self.fcs - fh).abs()).item()))
            self.harm_idx.append(h_idx)

    def _mel(self, windows_t: torch.Tensor, w: int):
        start = (windows_t.shape[1] - w) // 2
        xw = windows_t[:, start:start + w]
        hann = torch.hann_window(w, periodic=True, device=xw.device)
        P = torch.fft.rfft(xw * hann, n=w).abs() ** 2          # (T, w//2+1)
        H, freqs = self._H[w]
        e = P @ H.T                                             # (T, C)
        return e, P, freqs

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        windows_t = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []
        T = windows_t.shape[0]
        nF = self.N_BM
        log_eps = 1e-7

        mels = {}
        for w in (1024, 2048, 4096):
            e, P, freqs = self._mel(windows_t, w)
            log_e = torch.log(e + log_eps)
            cube_e = torch.pow(e + log_eps, 1.0 / 3.0)
            mels[w] = (log_e, cube_e, P, freqs)

        # ---- Cochleagrams (log + cube) for each window
        for w in (1024, 2048, 4096):
            log_e, cube_e, _, _ = mels[w]
            feats.extend([log_e, cube_e])

        log_e_long = mels[4096][0]
        cube_e_long = mels[4096][1]
        P_long = mels[4096][2]
        freqs_long = mels[4096][3]

        # ---- Long-tau dynamics on long window
        for tau in self.LONG_TAUS:
            feats.append(log_e_long - causal_ema(log_e_long, tau))

        # ---- MOC-style global slow loudness contrast
        loud = log_e_long.mean(dim=1, keepdim=True)
        for tau in self.LONG_TAUS:
            feats.append(loud - causal_ema(loud, tau))

        # ---- 2D STMF on long log_e
        for (wt, wf) in self.STMF_RATES:
            feats.append(stmf_gabor_2d(log_e_long, wt, wf))

        # ---- Lateral inh DoG
        feats.append(channel_dog(log_e_long, 1.0, 3.0))
        feats.append(channel_dog(log_e_long, 1.0, 6.0))

        # ---- Pitch via harmonic template (cross-channel matched filter)
        baseline = log_e_long.mean(dim=1, keepdim=True)
        templ_scores = []
        for f0, h_idx in zip(self.F0_CANDS, self.harm_idx):
            if not h_idx:
                templ_scores.append(torch.zeros(T, 1, device=windows_t.device))
                continue
            sal = log_e_long[:, h_idx].mean(dim=1, keepdim=True) - baseline
            templ_scores.append(sal)
        harm = torch.cat(templ_scores, dim=1)
        feats.append(harm)
        best_str, best_idx = harm.max(dim=1, keepdim=True)
        f0_cands_t = torch.tensor(self.F0_CANDS, device=windows_t.device, dtype=torch.float32)
        log_f0 = torch.log(f0_cands_t[best_idx.squeeze(1)].unsqueeze(1) + 1.0)
        feats.append(torch.cat([best_str, log_f0], dim=1))

        # ---- Phonetic matched filters
        very_lo = log_e_long[:, :max(1, nF // 8)].mean(dim=1, keepdim=True)
        lo = log_e_long[:, :nF // 4].mean(dim=1, keepdim=True)
        mid_lo = log_e_long[:, nF // 6:nF // 3].mean(dim=1, keepdim=True)
        mid = log_e_long[:, nF // 4:3 * nF // 4].mean(dim=1, keepdim=True)
        mid_hi = log_e_long[:, nF // 2:3 * nF // 4].mean(dim=1, keepdim=True)
        hi = log_e_long[:, 3 * nF // 4:].mean(dim=1, keepdim=True)
        very_hi = log_e_long[:, max(0, nF - nF // 8):].mean(dim=1, keepdim=True)
        spec_var = log_e_long.var(dim=1, keepdim=True)
        sibilance = very_hi - lo
        fric = mid_hi - lo - 0.5 * best_str
        vowel = mid_lo + 0.5 * best_str - 0.3 * very_hi
        plosive = spec_var - 0.5 * best_str
        nasal = very_lo - hi
        approx = mid + 0.3 * best_str - 0.3 * spec_var
        voicedstop = best_str + lo - 0.3 * very_hi
        phon = torch.cat([sibilance, fric, vowel, plosive, nasal, approx,
                          voicedstop], dim=1)
        feats.append(phon)
        for tau in (9.0, 30.0):
            feats.append(phon - causal_ema(phon, tau))

        # ---- Cross-band products
        log_n = log_e_long - log_e_long.mean(dim=1, keepdim=True)
        prods = []
        pair_ratios = [(0.05, 0.55), (0.1, 0.7), (0.2, 0.85),
                       (0.3, 0.95), (0.5, 0.95), (0.4, 0.6)]
        for a, b in pair_ratios:
            ia, ib = int(a * nF), min(nF - 1, int(b * nF))
            prods.append((log_n[:, ia] * log_n[:, ib]).unsqueeze(1))
        feats.append(torch.cat(prods, dim=1))

        # ---- Spectral flux, moments, flatness, rolloff per window
        for w in (1024, 2048, 4096):
            log_e, cube_e, P, freqs = mels[w]
            d = torch.zeros_like(log_e); d[1:] = log_e[1:] - log_e[:-1]
            feats.append(torch.cat([d.clamp(min=0).sum(dim=1, keepdim=True),
                                     (-d).clamp(min=0).sum(dim=1, keepdim=True),
                                     d.abs().sum(dim=1, keepdim=True)], dim=1))
            P_norm = P / (P.sum(dim=1, keepdim=True) + 1e-12)
            mu = (P_norm * freqs).sum(dim=1, keepdim=True)
            std = torch.sqrt(((P_norm * (freqs - mu) ** 2).sum(dim=1, keepdim=True)) + 1e-6)
            feats.append(torch.cat([torch.log(mu.clamp(min=1.0)),
                                     torch.log(std.clamp(min=1.0))], dim=1))
            log_P = torch.log(P.clamp(min=1e-10))
            flat = log_P.mean(dim=1, keepdim=True) - torch.log(P.mean(dim=1, keepdim=True) + 1e-10)
            feats.append(flat)
            cs = torch.cumsum(P, dim=1)
            total = cs[:, -1:] + 1e-12
            roll_idx = (cs >= 0.85 * total).float().argmax(dim=1)
            feats.append(torch.log(freqs[roll_idx].clamp(min=1.0)).unsqueeze(1))

        # ---- Future/past deltas on long log_e
        for k in (3, 6, 12):
            fut = torch.cat([log_e_long[k:], log_e_long[-1:].expand(k, -1)], dim=0)
            feats.append(fut - log_e_long)
            past = torch.cat([log_e_long[:1].expand(k, -1), log_e_long[:-k]], dim=0)
            feats.append(log_e_long - past)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------

model_shorthand_name = "MEL-CTRL-v9"
model_description = (
    "DIAGNOSTIC CONTROL: pure multi-window mel-spec pipeline (1024/2048/4096 "
    "windows, 32 ERB channels each), NO biophysical periphery. Same downstream "
    "as biophys-v7/v8: log+cube cochs, multi-tau dynamics, MOC slow-loud, 14 "
    "STMF on long log_e, lateral-inh DoG, harmonic template pitch + log F0, "
    "7 phonetic templates with dynamics, cross-band products, per-window flux/"
    "moments/flatness/rolloff, future/past deltas. Compare vs biophys v7=0.0650."
)


def build_embedder(device: str = "cuda") -> MelControlV9:
    return MelControlV9(device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available — cpu")
        args.device = "cpu"

    t0 = time.time()
    cfg = EncodingConfig(num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = int(getattr(embedder, "n_params", 0))
    upsert_overall_results(
        [make_result_row(r, model_shorthand_name, n_params, model_description)],
        RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)
    print()
    print("---")
    print(f"subject:        {r['subject']}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['train_corr']:.4f}, "
          f"frac_elec>0.1={r['frac_test_elec_above_0.1']:.4f})")
    print(f"n_params:       {n_params:.2e}")
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
