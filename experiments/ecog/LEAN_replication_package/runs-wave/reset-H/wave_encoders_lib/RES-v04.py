"""
Echo-State / Fixed-Random Reservoir auditory front-end:
raw WAVEFORM -> multi-window cochleagram base -> high-dim nonlinear reservoirs
that provide LONG-RANGE memory (beyond ridge's 4-tap FIR) -> ECoG ridge.

Iter 3 design: combine a strong multi-window cochleagram base (so ridge has the
direct linear info it needs) with slow-leak reservoirs that contribute nonlinear
fading-memory features the ridge cannot synthesize from band energies alone.
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
CACHE_DIR = "/path/to/scratch/feat_cache_reset_H"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cochleagram (gammatone power bank in freq domain)
# ---------------------------------------------------------------------------

def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_space(f_min, f_max, n):
    erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_pts = np.linspace(erb_min, erb_max, n)
    return (10.0 ** (erb_pts / 21.4) - 1.0) * 1000.0 / 4.37


def gammatone_powerbank(win, n_filters=40, f_min=80.0, f_max=8000.0, order=4):
    freqs = np.fft.rfftfreq(win, d=1.0 / SR)
    fcs = erb_space(f_min, f_max, n_filters)
    b = 1.019 * erb(fcs)
    df = freqs[:, None] - fcs[None, :]
    H2 = (1.0 + (df / b[None, :]) ** 2) ** (-order)
    return torch.from_numpy(H2.astype(np.float32))


def power_spectrum(windows_t, win, device):
    hann = torch.hann_window(win, periodic=True, device=device)
    return torch.fft.rfft(windows_t * hann, n=win).abs() ** 2


# ---------------------------------------------------------------------------
# Reservoir (echo-state) — fixed random recurrent network, NEVER trained.
# ---------------------------------------------------------------------------

def make_reservoir(N: int, dim_in: int, spectral_radius: float,
                    input_scale: float, density: float, seed: int):
    rng = np.random.RandomState(seed)
    W = rng.randn(N, N).astype(np.float32)
    mask = (rng.rand(N, N) < density).astype(np.float32)
    W *= mask
    v = rng.randn(N).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(50):
        v = W @ v
        nrm = np.linalg.norm(v) + 1e-12
        v /= nrm
    rho = float(np.linalg.norm(W @ v))
    if rho < 1e-8:
        rho = 1.0
    W *= (spectral_radius / rho)
    W_in = (rng.randn(N, dim_in).astype(np.float32) * input_scale) / np.sqrt(dim_in)
    return W, W_in


@torch.no_grad()
def run_reservoir(u: torch.Tensor, W: torch.Tensor, W_in: torch.Tensor,
                   leak: float) -> torch.Tensor:
    T = u.shape[0]
    N = W.shape[0]
    x = torch.zeros(N, device=u.device, dtype=u.dtype)
    drive_in = u @ W_in.T
    states = torch.empty(T, N, device=u.device, dtype=u.dtype)
    for t in range(T):
        x = (1.0 - leak) * x + leak * torch.tanh(W @ x + drive_in[t])
        states[t] = x
    return states


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0
    n_bands = 40
    WIN_LIST = (1024, 2048, 4096)

    # L1: 3 slow-to-mid reservoirs driven by the cochleagram-based input.
    # (N, spectral_radius, leak, input_scale, density, seed)
    RESERVOIRS = (
        (400, 0.95, 0.05, 1.5, 0.10, 1001),
        (400, 0.95, 0.15, 1.5, 0.10, 2002),
        (400, 0.92, 0.40, 1.5, 0.10, 3003),
    )
    # L2: one deep reservoir driven by the concatenation of L1 states.
    # Provides nonlinear functions of nonlinear features of input — a depth a
    # single ESN cannot reproduce.
    RES_L2 = (600, 0.90, 0.10, 0.3, 0.10, 5005)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fbs = [gammatone_powerbank(w, n_filters=self.n_bands).to(device)
                    for w in self.WIN_LIST]
        # Reservoir input dim: log_e + cube_e + pos_flux + neg_flux from the
        # long window = 40*4 = 160
        self.reservoirs = []
        dim_in = self.n_bands * 4
        for (N, rho, leak, ins, dens, seed) in self.RESERVOIRS:
            W_np, Win_np = make_reservoir(N, dim_in, rho, ins, dens, seed)
            self.reservoirs.append((
                torch.from_numpy(W_np).to(device),
                torch.from_numpy(Win_np).to(device),
                leak,
            ))
        # L2 reservoir: input is concat of L1 states
        l1_total = sum(r[0] for r in self.RESERVOIRS)
        N2, rho2, leak2, ins2, dens2, seed2 = self.RES_L2
        W2_np, Win2_np = make_reservoir(N2, l1_total, rho2, ins2, dens2, seed2)
        self.W2 = torch.from_numpy(W2_np).to(device)
        self.W_in2 = torch.from_numpy(Win2_np).to(device)
        self.leak2 = leak2

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        feats = []
        log_es = []
        for w, fb in zip(self.WIN_LIST, self.fbs):
            if w == 4096:
                xw = x
            else:
                start = (4096 - w) // 2
                xw = x[:, start:start + w]
            p = power_spectrum(xw, w, self.device)
            e = p @ fb  # (T, n_bands)
            log_e = torch.log(torch.clamp(e, min=1e-10))
            cube_e = torch.pow(e + 1e-10, 1.0 / 3.0)
            log_es.append(log_e)
            feats.append(log_e)
            feats.append(cube_e)

        # Long-window features for driving the reservoir
        log_e_long = log_es[-1]                                          # (T, 40)
        cube_e_long = torch.pow(
            power_spectrum(x, 4096, self.device) @ self.fbs[-1] + 1e-10,
            1.0 / 3.0)                                                   # (T, 40)
        # Half-wave rectified flux (pos and neg) — onset/offset
        d = torch.zeros_like(log_e_long)
        d[1:] = log_e_long[1:] - log_e_long[:-1]
        flux_pos = d.clamp(min=0.0)
        flux_neg = (-d).clamp(min=0.0)
        feats.append(flux_pos)
        feats.append(flux_neg)

        # Build reservoir input: z-scored concat of [log_e, cube_e, flux_pos, flux_neg]
        def _zs(v):
            return (v - v.mean(0)) / (v.std(0) + 1e-6)
        u_in = torch.cat(
            [_zs(log_e_long), _zs(cube_e_long), _zs(flux_pos), _zs(flux_neg)], dim=1)

        # L1: drive each reservoir; state IS the representation.
        l1_states = []
        for (W, W_in, leak) in self.reservoirs:
            states = run_reservoir(u_in, W, W_in, leak)
            feats.append(states)
            l1_states.append(states)

        # L2: deep reservoir driven by concat of L1 states (z-scored).
        def _zs(v):
            return (v - v.mean(0)) / (v.std(0) + 1e-6)
        l1_concat = torch.cat(l1_states, dim=1)
        l1_concat = _zs(l1_concat)
        l2_states = run_reservoir(l1_concat, self.W2, self.W_in2, self.leak2)
        feats.append(l2_states)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "RES-v04"
model_description = (
    "RES v04: deep / hierarchical ESN. v03 static (320) + L1 3xN=400 + L2 N=600 "
    "driven by z-scored L1 concat (leak=0.10, rho=0.90). Out=320+1200+600=2120. "
    "L2 should give compositional nonlinear features. v03 test_corr=0.0637."
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
        print("cuda not available — falling back to cpu")
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
