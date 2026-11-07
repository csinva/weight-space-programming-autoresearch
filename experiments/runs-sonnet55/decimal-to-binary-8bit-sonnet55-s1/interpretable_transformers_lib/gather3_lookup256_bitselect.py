"""
Interpretable transformer for character-level sequence tasks.
Snapshot: gather3_lookup256_bitselect (decimal-to-binary-8bit, accuracy 1.0)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


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
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=4, n_layers=3, d_ff=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built 2-layer circuit for decimal -> 8-bit binary.

    Residual layout (d_model=64):
      0-10 token one-hot | 11-22 position one-hot
      23-32 hundreds digit, 33-42 tens digit, 43-52 units digit (gathered)
      53-60 bits of N (MSB first) | 61 logit '0' | 62 logit '1' | 63 constant 1
    Layer 1 attn: heads 0,1,2 attend to prompt positions 0,1,2 and copy the digit one-hot.
    Layer 1 MLP: one unit per N in 0..255 (AND of its three digits) writes N's 8 bits.
    Layer 2 MLP: at position 3+j, unit reads (bit j) AND (position 3+j) -> logit.
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        for i in range(11):
            model.token_emb.weight[i, i] = 1.0
        for t in range(model.max_seq_len):
            if t < 12:
                model.pos_emb.weight[t, 11 + t] = 3.0
            model.pos_emb.weight[t, 63] = 1.0
        for b in model.blocks:
            b.ln1.weight.fill_(1.0); b.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        b0 = model.blocks[0]
        S = 8.0
        for h in range(3):
            b0.attn.W_q.weight[16 * h + 15, 63] = S
            b0.attn.W_k.weight[16 * h + 15, 11 + h] = 1.0
            for d in range(10):
                b0.attn.W_v.weight[16 * h + d, d] = 1.0
                b0.attn.W_o.weight[23 + 10 * h + d, 16 * h + d] = 2.0
        for N in range(256):
            a, bb, c = N // 100, (N // 10) % 10, N % 10
            for dim in (23 + a, 33 + bb, 43 + c):
                b0.mlp.fc1.weight[N, dim] = 1.0
            b0.mlp.fc1.bias[N] = -11.0
            for k in range(8):
                if (N >> (7 - k)) & 1:
                    b0.mlp.fc2.weight[53 + k, N] = 0.9
        b1 = model.blocks[1]
        for j in range(8):
            b1.mlp.fc1.weight[j, 53 + j] = 2.0
            b1.mlp.fc1.weight[j, 14 + j] = 1.0
            b1.mlp.fc1.bias[j] = -3.2
            b1.mlp.fc2.weight[62, j] = 4.0
            b1.mlp.fc1.weight[8 + j, 53 + j] = -2.0
            b1.mlp.fc1.weight[8 + j, 14 + j] = 1.0
            b1.mlp.fc1.bias[8 + j] = -1.7
            b1.mlp.fc2.weight[61, 8 + j] = 4.0
        model.head.weight[0, 61] = 1.0
        model.head.weight[1, 62] = 1.0


model_shorthand_name = "gather3_lookup256_bitselect"
model_description = "2 layers: 3 heads gather the digits at fixed positions; MLP1 has one AND-unit per N in 0..255 writing its 8 bits; MLP2 selects bit j by position 3+j (pos amp 3, bit weight 2)."

# (evaluation harness unchanged from interpretable_transformer.py)
