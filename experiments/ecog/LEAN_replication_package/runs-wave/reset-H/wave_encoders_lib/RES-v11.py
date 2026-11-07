"""
Echo-State / Fixed-Random Reservoir front-end with MIXED NONLINEARITIES.

Iter 9 design: a bidirectional ensemble of two reservoir families:
 - tanh MT reservoir (bounded, symmetric)        — N=600 fwd + N=600 bwd
 - ReLU MT reservoir (sparse, asymmetric, unbounded) — N=600 fwd + N=600 bwd
Both run on the same 4-lag input. Different activation regimes produce
genuinely different fading-memory feature spaces; a linear ridge cannot
synthesize either from the cochleagram alone.
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
# Reservoir (fixed-random recurrent network, NEVER trained)
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
    rng = np.random.RandomState(seed + 7777)
    log_low, log_high = np.log(leak_low), np.log(leak_high)
    leaks = np.exp(rng.uniform(log_low, log_high, size=N)).astype(np.float32)
    return leaks


@torch.no_grad()
def run_reservoir_mt(u: torch.Tensor, W: torch.Tensor, W_in: torch.Tensor,
                      leaks: torch.Tensor, activation: str = "tanh") -> torch.Tensor:
    T = u.shape[0]
    N = W.shape[0]
    x = torch.zeros(N, device=u.device, dtype=u.dtype)
    drive_in = u @ W_in.T
    states = torch.empty(T, N, device=u.device, dtype=u.dtype)
    one_minus = 1.0 - leaks
    if activation == "tanh":
        for t in range(T):
            x = one_minus * x + leaks * torch.tanh(W @ x + drive_in[t])
            states[t] = x
    elif activation == "relu":
        for t in range(T):
            z = W @ x + drive_in[t]
            phi = torch.tanh(0.3 * z.clamp(min=0.0)) * 3.0
            x = one_minus * x + leaks * phi
            states[t] = x
    elif activation == "sin":
        for t in range(T):
            x = one_minus * x + leaks * torch.sin(W @ x + drive_in[t])
            states[t] = x
    else:
        raise ValueError(activation)
    return states


@torch.no_grad()
def run_reservoir_gated(u: torch.Tensor, W: torch.Tensor, W_in: torch.Tensor,
                         W_g: torch.Tensor, W_gi: torch.Tensor) -> torch.Tensor:
    """Fixed-random gated reservoir. Leak is DATA-DRIVEN per timestep per unit:
        g_t = sigmoid(W_g x[t-1] + W_gi u[t])
        x[t] = (1 - g_t) * x[t-1] + g_t * tanh(W x[t-1] + W_in u[t])
    Acts as a fixed-random GRU-like cell — adaptive fading memory.
    """
    T = u.shape[0]
    N = W.shape[0]
    x = torch.zeros(N, device=u.device, dtype=u.dtype)
    drive_in = u @ W_in.T   # (T, N)
    drive_gi = u @ W_gi.T   # (T, N)
    states = torch.empty(T, N, device=u.device, dtype=u.dtype)
    for t in range(T):
        g = torch.sigmoid(W_g @ x + drive_gi[t])
        upd = torch.tanh(W @ x + drive_in[t])
        x = (1.0 - g) * x + g * upd
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

    # 6 bidirectional reservoirs across 3 families: tanh, ReLU, GATED.
    N_TANH = 600
    N_RELU = 600
    N_GATE = 100
    RHO_TANH = 0.95
    RHO_RELU = 0.85
    RHO_GATE = 0.95
    DENSITY = 0.08
    INPUT_SCALE = 1.5
    LEAK_LOW = 0.02
    LEAK_HIGH = 1.00
    SEED_TANH_FW = 12345
    SEED_TANH_BW = 67890
    SEED_RELU_FW = 13579
    SEED_RELU_BW = 24680
    SEED_GATE_FW = 31415
    SEED_GATE_BW = 27182

    INPUT_LAGS = (0, 2, 5, 10)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fbs = [gammatone_powerbank(w, n_filters=self.n_bands).to(device)
                    for w in self.WIN_LIST]
        dim_in = self.n_bands * 4 * len(self.INPUT_LAGS)

        def _make(N, rho, seed, input_scale=None):
            ins = self.INPUT_SCALE if input_scale is None else input_scale
            W = make_reservoir_W(N, rho, self.DENSITY, seed)
            Win = make_input_matrix(N, dim_in, ins, seed)
            leaks = make_leak_vector(N, self.LEAK_LOW, self.LEAK_HIGH, seed)
            return (torch.from_numpy(W).to(device),
                    torch.from_numpy(Win).to(device),
                    torch.from_numpy(leaks).to(device))

        self.tanh_fw = _make(self.N_TANH, self.RHO_TANH, self.SEED_TANH_FW)
        self.tanh_bw = _make(self.N_TANH, self.RHO_TANH, self.SEED_TANH_BW)
        self.relu_fw = _make(self.N_RELU, self.RHO_RELU, self.SEED_RELU_FW)
        self.relu_bw = _make(self.N_RELU, self.RHO_RELU, self.SEED_RELU_BW)

        # GATED bidir reservoir: needs an extra (W_g, W_gi) per direction.
        def _make_gated(N, rho, seed):
            W = make_reservoir_W(N, rho, self.DENSITY, seed)
            W_g = make_reservoir_W(N, 0.3, self.DENSITY, seed + 1)  # small-rho gate
            Win = make_input_matrix(N, dim_in, self.INPUT_SCALE, seed)
            W_gi = make_input_matrix(N, dim_in, 1.0, seed + 1)
            # Bias the gate toward 0.3 (slow integration by default) by adding to W_gi
            return (torch.from_numpy(W).to(device),
                    torch.from_numpy(Win).to(device),
                    torch.from_numpy(W_g).to(device),
                    torch.from_numpy(W_gi).to(device))

        self.gate_fw = _make_gated(self.N_GATE, self.RHO_GATE, self.SEED_GATE_FW)
        self.gate_bw = _make_gated(self.N_GATE, self.RHO_GATE, self.SEED_GATE_BW)

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

        base_signals = [_zs(log_e_long), _zs(cube_e_long), _zs(flux_pos), _zs(flux_neg)]
        u_pieces = []
        for k in self.INPUT_LAGS:
            for sig in base_signals:
                u_pieces.append(_lag(sig, k))
        u_in = torch.cat(u_pieces, dim=1)
        u_in_rev = torch.flip(u_in, dims=[0])

        # tanh bidir
        s = run_reservoir_mt(u_in, *self.tanh_fw, activation="tanh")
        feats.append(s)
        s_rev = run_reservoir_mt(u_in_rev, *self.tanh_bw, activation="tanh")
        feats.append(torch.flip(s_rev, dims=[0]))
        # ReLU bidir
        s = run_reservoir_mt(u_in, *self.relu_fw, activation="relu")
        feats.append(s)
        s_rev = run_reservoir_mt(u_in_rev, *self.relu_bw, activation="relu")
        feats.append(torch.flip(s_rev, dims=[0]))
        # GATED bidir
        s = run_reservoir_gated(u_in, *self.gate_fw)
        feats.append(s)
        s_rev = run_reservoir_gated(u_in_rev, *self.gate_bw)
        feats.append(torch.flip(s_rev, dims=[0]))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "RES-v11"
model_description = (
    "RES v11: bidir ensemble with GATED reservoir. tanh (N=600 bidir) + ReLU "
    "(N=600 bidir) + GATED (N=100 bidir; data-driven leak gate = sigmoid(W_g x "
    "+ W_gi u), fixed-random GRU-like cell). Same 4-lag 640-dim input. "
    "Out=320+1200+1200+200=2920 raw -> 11680 features. v09=0.0710 (best), "
    "v10b (sin variant)=0.0705."
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
