"""
Interpretable audio embedder for patient1 melspec -> ECoG high-gamma encoding.

The agent edits this file. The goal: hand-write the weights of a small
transformer so that the per-frame hidden state it produces for the audio
mel-spectrogram is a good *feature* for predicting high-gamma cortical activity —
beating the linear ridge-on-melspec baseline (`src/baseline.py`).

Rules of the game (same spirit as the sibling `evolve-neuro/` project):
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting,
    no backprop. All parameters must be set in closed form.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits / filter banks, lookup tables, etc.) inside `write_weights()`.
  * `write_weights()` runs once at construction. It must leave every parameter
    of `SimpleTransformer` populated.
  * Do NOT load pretrained weights and do NOT use external tools to compute the
    embedding — it must come from the transformer forward pass.

Input is continuous: each frame is a 50-bin mel vector at 30 Hz (NOT discrete
tokens), so the model uses a linear input projection (50 -> d_model) instead of a
token-embedding table, and fixed sinusoidal positional encodings (so any chunk
length works). The transformer is run causally over the chunk in contiguous
blocks of `block_size` frames; the relevant audio history for high-gamma is far
shorter than a block, so block boundaries are harmless.

Usage:
    python interpretable_transformer.py
    python interpretable_transformer.py --num-train 60 --num-test 20 --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.data import N_MELS
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

def sinusoidal_pos(L: int, d_model: int, device) -> torch.Tensor:
    """Fixed sinusoidal positional encodings, shape (L, d_model)."""
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(L, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer over a continuous melspec sequence. `forward` returns
    the per-frame hidden states (no readout head); the embedder uses every
    frame's final-LN hidden state as the encoding feature for that 30 Hz frame."""

    def __init__(
        self,
        in_dim: int = N_MELS,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.input_proj = nn.Linear(in_dim, d_model)  # continuous melspec -> model
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim) melspec frames
        B, T, _ = x.shape
        h = self.input_proj(x) + sinusoidal_pos(T, self.d_model, x.device)[None]
        for block in self.blocks:
            h = block(h)
        return self.final_ln(h)


class MelspecEmbedder:
    """Runs `SimpleTransformer` causally over each chunk's melspec (in blocks) and
    returns per-frame hidden states. Exposes the embedder interface
    `__call__(melspec (T, 50)) -> np.ndarray (T, d_model)` used by the pipeline."""

    def __init__(self, model: SimpleTransformer, device: str = "cuda", block_size: int = 256):
        self.model = model.to(device).eval()
        self.device = device
        self.block_size = block_size

    @torch.no_grad()
    def __call__(self, melspec: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(melspec, dtype=np.float32)).to(self.device)  # (T, 50)
        T = x.shape[0]
        outs = []
        for s in range(0, T, self.block_size):
            seg = x[s:s + self.block_size].unsqueeze(0)  # (1, L, 50)
            outs.append(self.model(seg)[0])              # (L, d_model)
        return torch.cat(outs, 0).float().cpu().numpy()  # (T, d_model)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: deterministic random init. With random weights the
    embedding is a fixed random nonlinear feature of the audio, which yields only
    a small test correlation. The agent should replace this with hand-designed,
    human-understandable weights — e.g. an input projection that acts as an
    interpretable mel filter bank / spectral-contrast detector, attention that
    integrates a short window of recent audio (the relevant timescale for
    high-gamma), and MLP lookup tables mapping spectral patterns to cortical
    feature axes — to push the test correlation above the linear ridge baseline.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "RandomInitTransformer"
model_description = "Default 2-layer melspec transformer, random weights (no design) — baseline interpretable model."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = "cuda",
                   d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                   d_ff: int = 256, block_size: int = 256) -> MelspecEmbedder:
    model = SimpleTransformer(
        in_dim=N_MELS, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    write_weights(model)
    model.eval()
    return MelspecEmbedder(model, device=device, block_size=block_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=60)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available — falling back to cpu")
        args.device = "cpu"

    t0 = time.time()
    cfg = EncodingConfig(num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = sum(p.numel() for p in embedder.model.parameters())

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
