"""
Echo-State / Fixed-Random Reservoir front-end with MULTI-TIMESCALE units.

Key structural change vs v04: replace the L1/L2 hierarchy with a single large
reservoir whose units each have their own leak rate drawn from a log-uniform
distribution. The state is x_i[t] = (1-α_i) x_i[t-1] + α_i tanh(W_res x[t-1] +
W_in u[t]), giving the readout a *spectrum* of fading-memory time constants
within one network — strictly more expressive than any homogeneous-leak ESN of
the same size, and impossible for a linear ridge to synthesize.
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
# Cochleagram (gammatone power bank)
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
# Multi-timescale reservoir
# ---------------------------------------------------------------------------

def make_reservoir_W(N: int, spectral_radius: float, density: float, seed: int):
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
    return W


def make_input_matrix(N: int, dim_in: int, input_scale: float, seed: int):
    rng = np.random.RandomState(seed + 9999)
    return (rng.randn(N, dim_in).astype(np.float32) * input_scale) / np.sqrt(dim_in)


def make_leak_vector(N: int, leak_low: float, leak_high: float, seed: int):
    """Per-unit leak rates sampled log-uniformly so the reservoir spans many
    fading-memory timescales simultaneously."""
    rng = np.random.RandomState(seed + 7777)
    log_low, log_high = np.log(leak_low), np.log(leak_high)
    leaks = np.exp(rng.uniform(log_low, log_high, size=N)).astype(np.float32)
    return leaks


@torch.no_grad()
def run_reservoir_mt(u: torch.Tensor, W: torch.Tensor, W_in: torch.Tensor,
                      leaks: torch.Tensor) -> torch.Tensor:
    """Per-unit leak: x[t] = (1-α)*x[t-1] + α*tanh(W x[t-1] + W_in u[t]).
    `leaks` shape (N,). Vectorized across N; sequential over T."""
    T = u.shape[0]
    N = W.shape[0]
    x = torch.zeros(N, device=u.device, dtype=u.dtype)
    drive_in = u @ W_in.T  # (T, N)
    states = torch.empty(T, N, device=u.device, dtype=u.dtype)
    one_minus = 1.0 - leaks
    for t in range(T):
        x = one_minus * x + leaks * torch.tanh(W @ x + drive_in[t])
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

    # BIDIRECTIONAL multi-timescale reservoir: one runs forward, one backward.
    RES_N = 1200
    RES_RHO = 0.95
    RES_INPUT_SCALE = 1.5
    RES_DENSITY = 0.08
    RES_LEAK_LOW = 0.02
    RES_LEAK_HIGH = 1.00
    RES_SEED_FW = 12345
    RES_SEED_BW = 67890

    # Lags (in 30 Hz frames) to expose to the reservoir as INPUT each step.
    # The reservoir then mixes cross-time pairs nonlinearly via tanh recurrence.
    INPUT_LAGS = (0, 2, 5, 10)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fbs = [gammatone_powerbank(w, n_filters=self.n_bands).to(device)
                    for w in self.WIN_LIST]
        # Two independent MT reservoirs: one forward, one backward.
        # Reservoir input dim: 8 (warped signals) x 4 lags x 40 bands = 1280.
        dim_in = self.n_bands * 8 * len(self.INPUT_LAGS)
        Wfw = make_reservoir_W(self.RES_N, self.RES_RHO,
                                self.RES_DENSITY, self.RES_SEED_FW)
        Winfw = make_input_matrix(self.RES_N, dim_in,
                                    self.RES_INPUT_SCALE, self.RES_SEED_FW)
        leaksfw = make_leak_vector(self.RES_N, self.RES_LEAK_LOW,
                                     self.RES_LEAK_HIGH, self.RES_SEED_FW)
        Wbw = make_reservoir_W(self.RES_N, self.RES_RHO,
                                self.RES_DENSITY, self.RES_SEED_BW)
        Winbw = make_input_matrix(self.RES_N, dim_in,
                                    self.RES_INPUT_SCALE, self.RES_SEED_BW)
        leaksbw = make_leak_vector(self.RES_N, self.RES_LEAK_LOW,
                                     self.RES_LEAK_HIGH, self.RES_SEED_BW)
        self.W_fw = torch.from_numpy(Wfw).to(device)
        self.W_in_fw = torch.from_numpy(Winfw).to(device)
        self.leaks_fw = torch.from_numpy(leaksfw).to(device)
        self.W_bw = torch.from_numpy(Wbw).to(device)
        self.W_in_bw = torch.from_numpy(Winbw).to(device)
        self.leaks_bw = torch.from_numpy(leaksbw).to(device)

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
            e = p @ fb
            log_e = torch.log(torch.clamp(e, min=1e-10))
            cube_e = torch.pow(e + 1e-10, 1.0 / 3.0)
            log_es.append(log_e)
            feats.append(log_e)
            feats.append(cube_e)

        log_e_long = log_es[-1]
        cube_e_long = torch.pow(
            power_spectrum(x, 4096, self.device) @ self.fbs[-1] + 1e-10,
            1.0 / 3.0)
        d = torch.zeros_like(log_e_long)
        d[1:] = log_e_long[1:] - log_e_long[:-1]
        flux_pos = d.clamp(min=0.0)
        flux_neg = (-d).clamp(min=0.0)
        feats.append(flux_pos)
        feats.append(flux_neg)

        def _zs(v):
            return (v - v.mean(0)) / (v.std(0) + 1e-6)

        def _lag(v, k):
            if k <= 0:
                return v
            return torch.cat([v[:1].expand(k, -1), v[:-k]], dim=0)

        # Build multi-lag reservoir input with NONLINEAR-WARPED variants.
        # For each base signal s in [log_e, cube_e, flux_pos, flux_neg]:
        #   - s itself (z-scored)
        #   - sign(s) * s^2 (signed quadratic warp — closed-form nonlinearity)
        # 8 signals x 4 lags x 40 bands = 1280-dim reservoir input.
        base_signals_raw = [_zs(log_e_long), _zs(cube_e_long),
                             _zs(flux_pos), _zs(flux_neg)]
        base_signals = []
        for s in base_signals_raw:
            base_signals.append(s)
            base_signals.append(torch.sign(s) * (s * s))
        u_pieces = []
        for k in self.INPUT_LAGS:
            for sig in base_signals:
                u_pieces.append(_lag(sig, k))
        u_in = torch.cat(u_pieces, dim=1)

        # Forward MT reservoir
        states_fw = run_reservoir_mt(u_in, self.W_fw, self.W_in_fw, self.leaks_fw)
        feats.append(states_fw)
        # Backward MT reservoir: reverse input, run, reverse output
        u_in_rev = torch.flip(u_in, dims=[0])
        states_bw_rev = run_reservoir_mt(u_in_rev, self.W_bw, self.W_in_bw, self.leaks_bw)
        states_bw = torch.flip(states_bw_rev, dims=[0])
        feats.append(states_bw)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "RES-v08b"
model_description = (
    "RES v08b: bidir MT N=1200 each (v07 size, after N=1500 OOM). Enriched "
    "reservoir INPUT: for each base signal s in [log_e,cube_e,flux+,flux-], "
    "include s AND sign(s)*s^2. 8 signals x 4 lags x 40 bands = 1280-dim input. "
    "Out=320+1200+1200=2720. v07=0.0686 (best)."
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
