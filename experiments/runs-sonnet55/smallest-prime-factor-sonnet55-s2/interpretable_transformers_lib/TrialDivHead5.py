"""
Interpretable transformer, task smallest-prime-factor (snapshot TrialDivHead5: 2.67e5 params, 1000/1000).
prompt "64507=" -> answer "00251" (spf, zero padded; a prime maps to itself).

Circuit (no training, no LayerNorm):
  L1 attn : heads 0-4 read input digit i (value) into DIG_i; head 5 at query pos 5+j copies digit j into CV.
  L1 mlp  : for each prime p<=313, s_p = sum_i w_i*DIG_i with w_i = u*10^(4-i) mod p (signed, unit u chosen to
            minimise the number of reachable multiples); 3 relu neurons per multiple k*p form a bump at s_p = k*p
            -> D_p = [p divides N].
  L2 mlp  : adds F_p = relu(D_p - sum_{q<p} D_q) into the D_p channel (D+F == 2 only for the smallest dividing
            prime); NONE = relu(1 - sum D)  (no prime <= 313 divides N, i.e. N is prime).
  L3 mlp  : neuron (p, j) = relu(X_p + P_{5+j} - 2) writes digit j of p (nonzero digits only; 0 is the default
            logit via CONST); gated bump neurons copy digit CV when NONE = 1.
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
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

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
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 0, d_head: int = 10):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if n_heads > 0 else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


def _primes(n):
    return [i for i in range(2, n + 1) if all(i % q for q in range(2, int(i ** 0.5) + 1))]


PRIMES = _primes(313)  # every composite <= 99999 has a prime factor <= 313


def _layout():
    names = {}
    n = 0

    def add(key, size):
        nonlocal n
        names[key] = n
        n += size
    add("VAL", 1); add("CONST", 1); add("P", 10)
    add("DIG", 5); add("CV", 1); add("D", len(PRIMES)); names["F"] = names["D"]  # F is added into D in place
    add("NONE", 1); add("OUT", 10)
    return names, n


LAY, D_MODEL = _layout()


def _bump_neurons():
    """(prime idx, weight vector, k) for each multiple of p reachable by s_p."""
    out = []
    for pi, p in enumerate(PRIMES):
        best = None
        for u in range(1, p):  # multiplying all weights by a unit u mod p keeps "p | s"
            w = []
            for i in range(5):
                r = (u * pow(10, 4 - i, p)) % p
                if r > p // 2:
                    r -= p
                w.append(r)
            lo = 9 * sum(x for x in w if x < 0)
            hi = 9 * sum(x for x in w if x > 0)
            cost = hi // p - (-(-lo // p))  # number of multiples of p reachable = number of bumps
            if best is None or cost < best[0]:
                best = (cost, w, lo, hi)
        _, w, lo, hi = best
        for k in range(-(-lo // p), hi // p + 1):
            out.append((pi, w, k))
    return out


BUMPS = _bump_neurons()


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        d = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        n_p = len(PRIMES)
        self.blocks = nn.ModuleList([
            Block(d, 3 * len(BUMPS), n_heads=6, d_head=5),
            Block(d, n_p + 1),
            Block(d, sum(ch != "0" for p in PRIMES for ch in f"{p:05d}") + 30),
        ])
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def write_weights(model: SimpleTransformer, task) -> None:
    L = LAY
    n_p = len(PRIMES)
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        for c in range(10):
            model.token_emb.weight[c, L["VAL"]] = float(c)
        for pos in range(model.max_seq_len):
            if pos < 10:
                model.pos_emb.weight[pos, L["P"] + pos] = 1.0
            model.pos_emb.weight[pos, L["CONST"]] = 1.0

        at = model.blocks[0].attn
        B = 80.0
        dh = at.d_head
        for h in range(5):  # head h reads input digit h
            at.W_q.weight[h * dh, L["CONST"]] = 1.0
            at.W_k.weight[h * dh, L["P"] + h] = B
            at.W_v.weight[h * dh, L["VAL"]] = 1.0
            at.W_o.weight[L["DIG"] + h, h * dh] = 1.0
        for j in range(5):  # head 5: query pos 5+j looks at pos j
            at.W_q.weight[5 * dh + j, L["P"] + 5 + j] = 1.0
            at.W_k.weight[5 * dh + j, L["P"] + j] = B
        at.W_v.weight[5 * dh, L["VAL"]] = 1.0
        at.W_o.weight[L["CV"], 5 * dh] = 1.0

        m = model.blocks[0].mlp
        for n, (pi, w, k) in enumerate(BUMPS):
            p = PRIMES[pi]
            for t, (off, coef) in enumerate(((1, 1.0), (0, -2.0), (-1, 1.0))):
                r = 3 * n + t
                for i in range(5):
                    m.fc1.weight[r, L["DIG"] + i] = float(w[i])
                m.fc1.bias[r] = float(-k * p + off)
                m.fc2.weight[L["D"] + pi, r] = coef

        m = model.blocks[1].mlp
        for pi in range(n_p):
            m.fc1.weight[pi, L["D"] + pi] = 1.0
            for q in range(pi):
                m.fc1.weight[pi, L["D"] + q] = -1.0
            m.fc2.weight[L["F"] + pi, pi] = 1.0
        for q in range(n_p):
            m.fc1.weight[n_p, L["D"] + q] = -1.0
        m.fc1.bias[n_p] = 1.0
        m.fc2.weight[L["NONE"], n_p] = 1.0

        m = model.blocks[2].mlp
        r = 0
        for pi, p in enumerate(PRIMES):
            digs = f"{p:05d}"
            for j in range(5):
                if digs[j] == "0":
                    continue
                m.fc1.weight[r, L["F"] + pi] = 1.0
                m.fc1.weight[r, L["P"] + 5 + j] = 1.0
                m.fc1.bias[r] = -2.0
                m.fc2.weight[L["OUT"] + int(digs[j]), r] = 1.0
                r += 1
        M = 10.0
        for c in range(10):
            for off, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                m.fc1.weight[r, L["CV"]] = 1.0
                m.fc1.weight[r, L["NONE"]] = M
                m.fc1.bias[r] = float(-c + off - M)
                m.fc2.weight[L["OUT"] + c, r] = coef
                r += 1

        for c in range(10):
            model.head.weight[c, L["OUT"] + c] = 1.0
        model.head.weight[0, L["CONST"]] = 0.5


model_shorthand_name = "TrialDivHead5"
model_description = "Snapshot: trial division by 65 primes with bump neurons; d_head 5; B=80."


def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model

# (The evaluation harness in interpretable_transformer.py's __main__ block is unchanged and omitted from this snapshot.)
