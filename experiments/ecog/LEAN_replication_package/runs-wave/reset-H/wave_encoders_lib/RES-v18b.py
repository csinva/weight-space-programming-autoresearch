"""
Echo-State / Fixed-Random Reservoir front-end with PER-BAND-GROUP RESERVOIRS.

Iter 18: split the 40-band cochleagram into 4 frequency groups (low, mid-low,
mid-high, high). Each group drives its own tanh+ReLU bidirectional MT reservoir
pair, so reservoir dynamics specialize by frequency content (different temporal
statistics in voicing F0 range vs sibilant high-band noise). The ridge mixes
across groups; the reservoirs do NOT see other groups' bands.
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
    else:
        raise ValueError(activation)
    return states


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    win = 4096
    n_params = 0
    n_bands = 40
    WIN_LIST = (1024, 2048, 4096)

    # 4 frequency band groups, each gets its own tanh+ReLU bidir reservoir pair.
    N_GROUPS = 4
    N_PER_RESERVOIR = 150     # tanh fw, tanh bw, relu fw, relu bw each
    RHO_TANH = 0.99
    RHO_RELU = 0.95
    DENSITY = 0.08
    INPUT_SCALE = 1.5
    LEAK_LOW = 0.02
    LEAK_HIGH = 1.00
    INPUT_LAGS = (0, 2, 5, 10)
    BASE_SEED = 12345

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fbs = [gammatone_powerbank(w, n_filters=self.n_bands).to(device)
                    for w in self.WIN_LIST]
        self.group_slices = [slice(i * 10, (i + 1) * 10) for i in range(self.N_GROUPS)]
        dim_in_per_group = 10 * 4 * len(self.INPUT_LAGS)        # 160
        N = self.N_PER_RESERVOIR                                  # 150
        N_total = N * self.N_GROUPS                               # 600
        dim_in_total = dim_in_per_group * self.N_GROUPS           # 640

        def _make_blockdiag(rho, seed):
            """Block-diagonal W (state-mixing only within group) and W_in
            (each unit only sees its group's input slice). Compute is one
            big matmul; semantics are 4 isolated reservoirs."""
            W = np.zeros((N_total, N_total), dtype=np.float32)
            Win = np.zeros((N_total, dim_in_total), dtype=np.float32)
            for g in range(self.N_GROUPS):
                wseed = seed + g * 1000
                W_g = make_reservoir_W(N, rho, self.DENSITY, wseed)
                Win_g = make_input_matrix(N, dim_in_per_group,
                                            self.INPUT_SCALE, wseed)
                rs, re = g * N, (g + 1) * N
                cs, ce = g * dim_in_per_group, (g + 1) * dim_in_per_group
                W[rs:re, rs:re] = W_g
                Win[rs:re, cs:ce] = Win_g
            leaks = make_leak_vector(N_total, self.LEAK_LOW, self.LEAK_HIGH,
                                      seed + 99)
            return (torch.from_numpy(W).to(device),
                    torch.from_numpy(Win).to(device),
                    torch.from_numpy(leaks).to(device))

        self.tanh_fw = _make_blockdiag(self.RHO_TANH, self.BASE_SEED + 100)
        self.tanh_bw = _make_blockdiag(self.RHO_TANH, self.BASE_SEED + 200)
        self.relu_fw = _make_blockdiag(self.RHO_RELU, self.BASE_SEED + 300)
        self.relu_bw = _make_blockdiag(self.RHO_RELU, self.BASE_SEED + 400)

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

        # Pre-z-score the full 40-band signals
        log_e_zs = _zs(log_e_long)
        cube_e_zs = _zs(cube_e_long)
        flux_p_zs = _zs(flux_pos)
        flux_n_zs = _zs(flux_neg)

        # Build full 640-dim input by concatenating per-group inputs in order.
        # Block-diagonal W_in routes each unit only to its group's slice.
        u_pieces = []
        for g, sl in enumerate(self.group_slices):
            base_signals = [log_e_zs[:, sl], cube_e_zs[:, sl],
                             flux_p_zs[:, sl], flux_n_zs[:, sl]]
            for k in self.INPUT_LAGS:
                for sig in base_signals:
                    u_pieces.append(_lag(sig, k))
        u_in = torch.cat(u_pieces, dim=1)              # (T, 640)
        u_in_rev = torch.flip(u_in, dims=[0])

        # Single big scan per type (block-diag W keeps groups isolated).
        s = run_reservoir_mt(u_in, *self.tanh_fw, activation="tanh")
        feats.append(s)
        s_rev = run_reservoir_mt(u_in_rev, *self.tanh_bw, activation="tanh")
        feats.append(torch.flip(s_rev, dims=[0]))
        s = run_reservoir_mt(u_in, *self.relu_fw, activation="relu")
        feats.append(s)
        s_rev = run_reservoir_mt(u_in_rev, *self.relu_bw, activation="relu")
        feats.append(torch.flip(s_rev, dims=[0]))

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "RES-v18b"
model_description = (
    "RES v18b: PER-BAND-GROUP reservoirs implemented as BLOCK-DIAGONAL W and "
    "W_in (single fast scan = v14 compute, semantically 4 isolated reservoirs "
    "per type). 4 groups x 150 units per type x 4 types (tanh fw/bw, relu fw/bw) "
    "= 2400 reservoir features. Static 320 + 2400 = 10880. v14=0.0715 best."
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
