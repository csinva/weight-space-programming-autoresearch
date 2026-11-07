"""
Echo-State / Fixed-Random Reservoir auditory front-end:
raw WAVEFORM -> cochleagram -> high-dim nonlinear reservoir state -> ECoG ridge.

Paradigm: drive a high-dimensional FIXED random recurrent reservoir with the
cochleagram. The reservoir state x[t] = (1-leak)*x[t-1] + leak*tanh(W_res x[t-1]
+ W_in u[t]) is a rich nonlinear function of the entire stimulus history that a
linear ridge over band energies CANNOT synthesize on its own. Random fixed
weights are not a data-fit step.
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
# Cochleagram (closed-form, gammatone power bank in freq domain)
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
    """Build a sparse fixed-random reservoir. Returns (W_res (N,N), W_in (N,dim_in))."""
    rng = np.random.RandomState(seed)
    W = rng.randn(N, N).astype(np.float32)
    mask = (rng.rand(N, N) < density).astype(np.float32)
    W *= mask
    # rescale to target spectral radius via power iteration on the sparse W
    v = rng.randn(N).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(40):
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
    """Run a single reservoir: u (T, dim_in) -> states (T, N).
    Intrinsically sequential per-frame scan; vectorized across N units."""
    T = u.shape[0]
    N = W.shape[0]
    x = torch.zeros(N, device=u.device, dtype=u.dtype)
    # Precompute W_in @ u for all t -> (T, N)
    drive_in = u @ W_in.T  # (T, N)
    states = torch.empty(T, N, device=u.device, dtype=u.dtype)
    for t in range(T):
        x = (1.0 - leak) * x + leak * torch.tanh(W @ x + drive_in[t])
        states[t] = x
    return states


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class WaveEmbedder:
    """Cochleagram-driven echo-state reservoir front-end.

    Pipeline per chunk:
      windows (T, 2048) -> power spectrum (T, F) -> gammatone bank (T, 40)
        -> cube-root compression -> per-chunk z-score
        -> drive 3 fixed-random reservoirs at different timescales
        -> concat [cochleagram, reservoir_1, reservoir_2, reservoir_3].
    """

    win = 2048
    n_params = 0
    n_bands = 40

    # (N, spectral_radius, leak, input_scale, density, seed)
    RESERVOIRS = (
        (400, 0.95, 0.10, 1.0, 0.10, 1001),   # slow timescale
        (400, 0.95, 0.30, 1.0, 0.10, 2002),   # mid
        (400, 0.90, 0.70, 1.0, 0.10, 3003),   # fast
    )

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fb = gammatone_powerbank(self.win, n_filters=self.n_bands).to(device)
        self.reservoirs = []
        for (N, rho, leak, ins, dens, seed) in self.RESERVOIRS:
            W_np, Win_np = make_reservoir(N, self.n_bands, rho, ins, dens, seed)
            self.reservoirs.append((
                torch.from_numpy(W_np).to(device),
                torch.from_numpy(Win_np).to(device),
                leak,
            ))

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)
        # Power spectrum
        p = power_spectrum(x, self.win, self.device)
        # Cochleagram (cube-root compressed, basilar-membrane-like nonlinearity)
        e = p @ self.fb  # (T, n_bands)
        u_cube = torch.pow(e + 1e-10, 1.0 / 3.0)
        # Standardize the reservoir input per chunk along time
        u_std = (u_cube - u_cube.mean(0)) / (u_cube.std(0) + 1e-6)

        feats = []
        # Keep the cochleagram itself as a baseline (log for ridge dynamic range)
        log_e = torch.log(torch.clamp(e, min=1e-10))
        feats.append(log_e)

        # Drive each reservoir; the state IS the representation.
        for (W, W_in, leak) in self.reservoirs:
            states = run_reservoir(u_std, W, W_in, leak)
            feats.append(states)

        feat = torch.cat(feats, dim=1)
        return feat.float().cpu().numpy()


model_shorthand_name = "RES-v01"
model_description = (
    "RES v01: clean echo-state baseline. 40-ch cube-root gammatone cochleagram "
    "feeds three fixed-random reservoirs (N=400 each; leak=0.1/0.3/0.7; "
    "spectral_radius=0.95/0.95/0.90; density=0.10). Output = log_cochleagram(40) "
    "+ 3*400 = 1240 features."
)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

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
