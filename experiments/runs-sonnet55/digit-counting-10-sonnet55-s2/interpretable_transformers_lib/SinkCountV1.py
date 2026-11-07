"""
Snapshot SinkCountV1 -- digit-counting-10.
Task: prompt = 11 digits + '=', answer = 2-digit count of the OTHER digits equal to the last digit.
2-layer, 1-head transformer, weights written by hand (see write_weights).
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
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None
        if self.mlp is not None:
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 64,
                 n_heads: int = 1, n_layers: int = 2, d_ff: tuple = (0, 20)):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff[i]) for i in range(n_layers)])
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
    # Residual layout (d_model=64):
    #   0-9 token digit | 10 '=' token | 11-26 position one-hot
    #   27-36 copy of the LAST digit d | 37 sink-attention mass g | 38-47 logit for digit u
    for p in model.parameters():
        p.data.zero_()
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.weight.data.fill_(1.0)
    D = 10
    TOK, POS, COPY, G, OUT = 0, 11, 27, 37, 38
    with torch.no_grad():
        model.token_emb.weight[torch.arange(11), torch.arange(11)] = 1.0
        model.pos_emb.weight[torch.arange(16), POS + torch.arange(16)] = 1.0

        # Layer 1 attention: every position p<=10 attends to itself, positions >=11 attend
        # to position 10 (the last digit); it copies that digit into the COPY slots.
        a1 = model.blocks[0].attn
        for p in range(16):
            tgt = p if p <= 10 else 10
            a1.W_q.weight[tgt, POS + p] = 4.0
            a1.W_k.weight[p, POS + p] = 1.0
        for j in range(D):
            a1.W_v.weight[j, TOK + j] = 1.0
            a1.W_o.weight[COPY + j, j] = 0.2

        # Layer 2 attention: query = copied last digit d; keys = digit tokens.
        # Keys equal to d get a higher score. The '=' position (pos 11) is a "sink" whose
        # value is 1; so its attention mass g = 1/(const + m*(E-1)) encodes the number m
        # of digits equal to d (including itself). Pos 12's own key is masked out.
        a2 = model.blocks[1].attn
        for j in range(D):
            a2.W_q.weight[j, COPY + j] = SQ
            a2.W_k.weight[j, TOK + j] = 1.0
        a2.W_q.weight[10, POS + 11] = 3.0
        a2.W_q.weight[10, POS + 12] = 3.0
        a2.W_k.weight[10, POS + 12] = -3.0
        a2.W_v.weight[0, POS + 11] = 1.0
        a2.W_o.weight[G, 0] = GGAIN

        # Layer 2 MLP: nearest-neighbour decoding of g -> count m, one 'hat' per u=m-1.
        mlp = model.blocks[1].mlp
        for u in range(D):
            mlp.fc1.weight[2 * u, G] = A
            mlp.fc1.bias[2 * u] = -A * THETA[u]
            mlp.fc1.weight[2 * u + 1, G] = -A
            mlp.fc1.bias[2 * u + 1] = A * THETA[u]
            for k in (2 * u, 2 * u + 1):
                # gate: ~0 at pos 12 (layer-normed active pos dim = V_POS), -45 elsewhere (pos 11)
                mlp.fc1.weight[k, POS + 12] = 10.0
                mlp.fc1.bias[k] -= 10.0 * V_POS
                mlp.fc2.weight[OUT + u, k] = -1.0

        # Head: digit u <- OUT+u ; '=' token -> digit 0 (tens digit of the answer is 0)
        for u in range(D):
            model.head.weight[u, OUT + u] = 1.0
        model.head.weight[0, 10] = 1.0
        model.head.weight[10, POS + 12] = -10.0


SQ = 0.37
V_POS = 4.3243  # measured layer-normed value of the active position one-hot dim
GGAIN = 0.1
A = 300.0
BETA, GAMMA, R = -0.28503, 1.107, 6.158
THETA = [BETA + GAMMA / (u + 1 + R) for u in range(10)]

model_shorthand_name = "SinkCountV1"
model_description = "2-layer 1-head: L1 attn copies last digit; L2 attn sink-mass 1/(c+m(E-1)) encodes match count; MLP nearest-neighbour hats (calibrated) decode count-1 into units digit; tens digit always 0."

# (evaluation harness identical to interpretable_transformer.py)
