"""
Interpretable transformer for character-level sequence tasks.
Task: smallest-prime-factor. Prompt "64507=" -> answer "00251" (zero-padded
smallest prime factor; primes map to themselves, e.g. "86171=" -> "86171").

Rules: no training; weights are written in closed form in write_weights().

Usage:
    uv run interpretable_transformer.py --task smallest-prime-factor
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


# ---------------------------------------------------------------------------
# Residual-stream layout: named feature groups at fixed offsets
# ---------------------------------------------------------------------------
PRIME_MAX = 313 # largest possible smallest-prime-factor of a composite < 100000 (313^2 = 97969)
PRIMES = [p for p in range(2, PRIME_MAX + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
NP = len(PRIMES)  # 65
NA = 3            # N = 10^NB * A + B, A = first NA digits, B = last NB digits
NB = 5 - NA
SA, SB, M = 10 ** NA, 10 ** NB, 10 ** NB


N_RES = sum(PRIMES)   # one hidden neuron per (prime, residue)

TOK = 0               # 11  token one-hot ('=' embeds to zero)
POS = 12              # 11  position one-hot
G = 23                # 50  gathered digit one-hots: slot i = digit at position i
D_ = G + 50           # NP  D_p = 1 iff p divides N
LOG = D_ + NP         # 11  output logits
A_ = LOG + 11         # SA  one-hot of first NA digits (A)
B_ = A_ + SA          # SB  one-hot of last NB digits (B)
D_MODEL = B_ + SB


# ---------------------------------------------------------------------------
# Architecture (no LayerNorm; each MLP reads/writes only a slice of the stream)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_attn = n_heads * d_head
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)
        self.W_o = nn.Linear(d_attn, d_model, bias=False)

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


class SliceMLP(nn.Module):
    """relu-MLP that reads residual dims [r0, r1) and adds its output to dims [w0, w1)."""

    def __init__(self, d_ff, r0, r1, w0, w1):
        super().__init__()
        self.r0, self.r1, self.w0, self.w1 = r0, r1, w0, w1
        self.fc1 = nn.Linear(r1 - r0, d_ff)
        self.fc2 = nn.Linear(d_ff, w1 - w0)

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x[..., self.r0:self.r1])))
        return F.pad(out, (self.w0, x.shape[-1] - self.w1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, mlp, use_attn):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if use_attn else None
        self.mlp = mlp

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=D_MODEL, n_heads=5, d_head=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        mlps = [
            SliceMLP(SA + SB, G, G + 50, A_, B_ + SB),         # L1: digits -> one-hot(A), one-hot(B)
            SliceMLP(N_RES, A_, B_ + SB, D_, D_ + NP),         # L2: (A,B) -> divisibility by each prime
            SliceMLP(50 + 1 + 4 + 2 * NP, POS, D_ + NP, LOG, LOG + 11),  # L3: smallest divisor -> logits
        ]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_head, mlp, use_attn=(i == 0)) for i, mlp in enumerate(mlps)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Hand-written weights
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model, task) -> None:
    """Pipeline, computed identically at every answer position (5..9):
      L1 attn : head i copies digit i (token at position i) into slot G+10i.
      L1 MLP  : AND-neurons -> one-hot of A = first NA digits, B = last 5-NA digits (N = 10^NB*A + B).
      L2 MLP  : neuron (p,r) fires iff 10^NB*A = r and B = -r (mod p), i.e. p | N -> D_p.
      L3 MLP  : prime N -> emit digit j of N; composite -> '0','0',hundreds,tens,units of the
                smallest p with D_p=1 (neuron value 2*D_p - 3*#smaller divisors + 2*pos - 3).
    All AND-gates use weights 2 and bias -(2n-1): exactly 1 when all n inputs are on, <=-1 otherwise.
    """
    for prm in model.parameters():
        prm.zero_()
    for v in range(10):   # only digit tokens carry information; '=' embeds to 0 (its position says enough)
        model.token_emb.weight[v, TOK + v] = 1
    for i in range(11):
        model.pos_emb.weight[i, POS + i] = 1

    # L1 attention: head h attends to position h (score 6*10/sqrt(10) ~ 19 vs 0), copies the digit one-hot
    at = model.blocks[0].attn
    for h in range(5):
        at.W_q.weight[h * 10, POS + 5:POS + 11] = 6.0   # query: "I am an answer position"
        at.W_k.weight[h * 10, POS + h] = 10.0
        for d in range(10):
            at.W_v.weight[h * 10 + d, TOK + d] = 1.0
            at.W_o.weight[G + 10 * h + d, h * 10 + d] = 1.0

    # L1 MLP: one-hot of the digit groups A and B   (fc1 columns relative to G, fc2 rows relative to A_)
    m = model.blocks[0].mlp
    for off, size, first in ((0, SA, 0), (SA, SB, NA)):   # A: digits 0..NA-1, B: digits NA..4
        nd = 5 - first if first else NA
        for v in range(size):
            n = off + v
            for i, ch in enumerate(f"{v:0{nd}d}"):         # AND of the nd digit one-hots
                m.fc1.weight[n, 10 * (first + i) + int(ch)] = 2
            m.fc1.bias[n] = -(2 * nd - 1)
            m.fc2.weight[n, n] = 1

    # L2 MLP: divisibility by each prime   (fc1 columns relative to A_, fc2 rows relative to D_)
    m = model.blocks[1].mlp
    base = 0
    ar, br = torch.arange(SA), torch.arange(SB)
    for k, p in enumerate(PRIMES):   # neuron r of prime p: A*M = r and B = -r (mod p)  =>  N = 0 (mod p)
        m.fc1.weight[base + (M * ar) % p, ar] = 2
        m.fc1.weight[base + (-br) % p, SA + br] = 2
        m.fc1.bias[base:base + p] = -3
        m.fc2.weight[k, base:base + p] = 1
        base += p

    # L3 MLP: output logits   (fc1 columns relative to POS, fc2 rows relative to LOG)
    m = model.blocks[2].mlp
    cG, cD = G - POS, D_ - POS
    n = 0
    for j in range(5):
        for v in range(10):  # prime: copy digit j of N        (-c + pos + digit - 1 > 0)
            m.fc1.weight[n, cD:cD + NP] = -3
            m.fc1.weight[n, 5 + j] = 2
            m.fc1.weight[n, cG + 10 * j + v] = 2
            m.fc1.bias[n] = -3
            m.fc2.weight[v, n] = 1
            n += 1
    # composite: two leading zeros (answer positions 0 and 1; c may exceed 1, so large pos weight)
    m.fc1.weight[n, cD:cD + NP] = 1
    m.fc1.weight[n, 5:7] = 6    # c <= 6 distinct prime factors below 1e5 (2*3*5*7*11*13 = 30030)
    m.fc1.bias[n] = -6
    m.fc2.weight[0, n] = 1
    n += 1
    for h in range(4):  # composite: hundreds digit h = some prime in [100h,100h+100) divides N and none below
        lo = sum(1 for p in PRIMES if p < 100 * h)       # primes below the group
        hi = sum(1 for p in PRIMES if p < 100 * h + 100)
        m.fc1.weight[n, cD + lo:cD + hi] = 2
        m.fc1.weight[n, cD:cD + lo] = -15
        m.fc1.weight[n, 5 + 2] = 15
        m.fc1.bias[n] = -15
        m.fc2.weight[h, n] = 1
        n += 1
    for k, p in enumerate(PRIMES):  # composite: tens/units of smallest dividing prime (D_k - sum_{q<p} D_q)
        for j in range(3, 5):
            m.fc1.weight[n, cD:cD + k] = -3
            m.fc1.weight[n, cD + k] = 2
            m.fc1.weight[n, 5 + j] = 2
            m.fc1.bias[n] = -3
            m.fc2.weight[int(f"{p:03d}"[j - 2]), n] = 1
            n += 1

    for v in range(11):
        model.head.weight[v, LOG + v] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "spf_final_tidy"
model_description = "Final clean model: 3 layers, no LN, slice-MLPs; attn (key 10, query 6) gathers digits; margin AND-gates; A=first 3 digits, B=last 2; uniform (prime,residue) neurons for 65 primes; hundreds via 4 group neurons; zero-neuron w6."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
