"""
Interpretable transformer, task decimal-to-binary-8bit: prompt "abc=" -> 8 bits MSB first.
Snapshot: no_attn_L2_ff16.

Circuit: L1 attention (3 heads, each attends to a fixed position 0/1/2) copies the digits;
L1 MLP has 256 AND-units (one per N in 0..255) that write the bits of N (and complements);
L2 MLP (16 units) selects bit j at position 3+j and writes the '0'/'1' logit.
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attn:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff=256,
        use_attn=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        ffs = d_ff if isinstance(d_ff, (list, tuple)) else [d_ff] * n_layers
        attn_flags = use_attn if isinstance(use_attn, (list, tuple)) else [use_attn] * n_layers
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffs[i], attn_flags[i]) for i in range(n_layers)])
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
    """Hand-written circuit (see model_description).

    Residual layout (d_model=72):
      0-10 token one-hot | 11-22 position one-hot | 23 constant 1
      24-33 / 34-43 / 44-53 : hundreds / tens / units digit copied by attention
      54-61 bit_j of N | 62-69 NOT bit_j | 70,71 logit scratch for '0','1'
    Prompt "abc=" sits at positions 0-3; the bit j (MSB first) is predicted at position 3+j.
    """
    TOK, POS, CONST = 0, 11, 23
    DIG = [24, 34, 44]
    BIT, NBIT, LOGIT = 54, 62, 70
    D = model.d_model
    dh = D // model.n_heads
    stoi = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        # --- embeddings: token one-hot, position one-hot, constant ---
        for t in range(task.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p_ in range(12):
            model.pos_emb.weight[p_, POS + p_] = 1.0
        model.pos_emb.weight[:, CONST] = 1.0

        # --- layer 1 attention: head h looks at position h and copies its digit ---
        att = model.blocks[0].attn
        K = 60.0
        for h in range(3):
            att.W_q.weight[h * dh, CONST] = 1.0
            att.W_k.weight[h * dh, POS + h] = K
            for d in range(10):
                att.W_v.weight[h * dh + 1 + d, TOK + stoi[str(d)]] = 1.0
                att.W_o.weight[DIG[h] + d, h * dh + 1 + d] = 1.0

        # --- layer 1 MLP: one unit per N in 0..255 = AND of its three digits;
        #     unit writes bits of N (and complements) ---
        mlp = model.blocks[0].mlp
        for N in range(256):
            for h, dig in enumerate((N // 100, (N // 10) % 10, N % 10)):
                mlp.fc1.weight[N, DIG[h] + dig] = 1.0
            for j in range(8):
                bit = (N >> (7 - j)) & 1
                mlp.fc2.weight[BIT + j, N] = float(bit)
                mlp.fc2.weight[NBIT + j, N] = float(1 - bit)

        def ids_of(s):
            return torch.tensor([[stoi[c] for c in s]])

        def calibrate(mlp_, h_in, unit_match, unit_miss):
            # threshold = midpoint between a fully matching unit and a one-miss unit
            z = F.layer_norm(h_in, (D,))[0, -1]
            pre = mlp_.fc1.weight @ z
            mlp_.fc1.bias.fill_(-(pre[unit_match] + pre[unit_miss]).item() / 2)
            return ((pre[unit_match] - pre[unit_miss]) / 2).item()

        ids = ids_of("123=")
        h = model.token_emb(ids) + model.pos_emb(torch.arange(4))[None]
        h = h + model.blocks[0].attn(model.blocks[0].ln1(h))
        act1 = calibrate(mlp, h, 123, 124)
        mlp.fc2.weight.div_(act1)  # bit features become exactly 0/1

        # --- layer 2 MLP: unit (j,v) fires iff position==3+j and bit_j==v ---
        mlp2 = model.blocks[1].mlp
        for j in range(8):
            mlp2.fc1.weight[2 * j + 0, NBIT + j] = 1.0
            mlp2.fc1.weight[2 * j + 0, POS + 3 + j] = 1.0
            mlp2.fc1.weight[2 * j + 1, BIT + j] = 1.0
            mlp2.fc1.weight[2 * j + 1, POS + 3 + j] = 1.0
            mlp2.fc2.weight[LOGIT + 0, 2 * j + 0] = 1.0
            mlp2.fc2.weight[LOGIT + 1, 2 * j + 1] = 1.0
        h = model.blocks[0](model.token_emb(ids) + model.pos_emb(torch.arange(4))[None])
        # N=123 -> bit_0 = 0: unit 0 matches, unit 1 misses on the bit
        act2 = calibrate(mlp2, h, 0, 1)
        mlp2.fc2.weight.div_(act2)

        # --- readout ---
        model.head.weight[stoi["0"], LOGIT + 0] = 1.0
        model.head.weight[stoi["1"], LOGIT + 1] = 1.0


model_shorthand_name = "no_attn_L2_ff16"
model_description = "Same as digit_gather_and_units but layer 2 has no attention and d_ff=16 (only the 16 select units)."
