"""
Interpretable transformer for smallest-prime-factor: "64507=" -> "00251".
Weights are hand-written, never trained. ReLU only: no periodic activation.

Circuit (see write_weights):
  block 1 attention : 5 heads read the prompt digits into 5 one-hot slots
  block 1 MLP       : ReLU comb: y_p = sum of digit residues, bumps at the multiples of p give
                      F_p = (p divides n); lateral inhibition keeps the smallest firing prime,
                      which writes its 3 digits
  block 2 attention : copy head: answer step k reads prompt digit k
  block 2 MLP       : gates: copy the digit if no factor, else the factor digit of this step
  head              : tangent lines of -(A-d)^2: logit_d = 2dA - d^2 picks the nearest digit
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
# Hand-design plan
# ---------------------------------------------------------------------------

PRIME_LIMIT = 316     # trial-divide by all primes up to this (316 = isqrt(99999))
BETA = 10.0           # attention sharpness
DTYPE = torch.float32

PRIMES = [p for p in range(2, PRIME_LIMIT + 1) if all(p % q for q in range(2, p))]
M = len(PRIMES)


def residue(a: int, p: int) -> int:
    """Signed residue in (-p/2, p/2]."""
    r = a % p
    return r - p if r > p // 2 else r


def teeth_for(p: int) -> list[int]:
    """Multiples of p reachable by the sum of the five digit residues."""
    reach = {0}
    for i in range(5):
        reach = {y + residue(d * 10 ** (4 - i), p) for y in reach for d in range(10)}
    return sorted(y for y in reach if y % p == 0)


TEETH = {p: teeth_for(p) for p in PRIMES}
N_BUMPS = sum(len(t) for t in TEETH.values())

# Named slices of the state vector.
T_ = 0                                  # digit one-hot (10)
V_, ONE_ = 10, 11                       # digit value, constant 1
S_, S2_ = 12, 13                        # position s, s^2
STEP_ = 14                              # 3 flags: this is hundreds/tens/units answer step
H_ = STEP_ + 3                          # the 5 prompt digits, one-hot (50)
R_ = H_ + 50                            # 3 digits of the winning prime
K_ = R_ + 3                             # copied prompt digit
A_ = K_ + 1                             # answer digit value
D_MODEL = A_ + 1
# (n_heads, d_head, d_ff, activation) of each block.
BLOCKS = [(5, 10, N_BUMPS, "comb"), (1, 2, 4, "relu")]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head attention with rectangular q/k/v/o maps: heads get only the width they use."""

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
    def __init__(self, d_model: int, d_ff: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        if activation == "comb":
            self.mid = nn.Linear(d_ff, M, bias=False)       # bumps -> one flag per prime
            self.fc2 = nn.Linear(M, d_model)
        else:
            self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.fc1(x)
        if self.activation == "comb":
            u = self.mid(F.relu(1 - u.abs()))         # triangular bump: 1 at u = 0, 0 for |u| >= 1
            u = 2 * u - u.cumsum(-1)                  # lateral inhibition from all smaller primes
        return self.fc2(F.relu(u))


class Block(nn.Module):
    """Pre-LN removed: the hand-written circuit needs exact numeric signals."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, activation: str):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer with per-block sizes, vocab/seq-len configured from the task."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = D_MODEL):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, *spec) for spec in BLOCKS])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in closed form. No training."""
    torch.manual_seed(0)
    model.to(DTYPE)
    for prm in model.parameters():
        prm.data.zero_()
    b1, b2 = model.blocks

    # Embeddings: the digit as one-hot and as a number; for positions: 1, s, s^2, step flags.
    for d in range(10):
        model.token_emb.weight.data[d, T_ + d] = 1.0
        model.token_emb.weight.data[d, V_] = d
    pos = model.pos_emb.weight.data
    pos[:, ONE_] = 1.0
    for s in range(10):
        pos[s, S_], pos[s, S2_] = s, s * s
    for jd in range(3):
        pos[7 + jd, STEP_ + jd] = 1.0

    # --- Block 1 attention: head i has score -BETA (s-i)^2 up to a constant, so it reads digit i.
    #     The score expands to q = (2i, -1), k = (s, s^2).
    gain = BETA * math.sqrt(10)
    for i in range(5):
        off = i * 10
        b1.attn.W_q.weight.data[off, ONE_] = 2.0 * i * gain
        b1.attn.W_q.weight.data[off + 1, ONE_] = -gain
        b1.attn.W_k.weight.data[off, S_] = 1.0
        b1.attn.W_k.weight.data[off + 1, S2_] = 1.0
        for d in range(10):
            b1.attn.W_v.weight.data[off + d, T_ + d] = 1.0
            b1.attn.W_o.weight.data[H_ + off + d, off + d] = 1.0

    # --- Block 1 MLP: each unit reads y_p = sum of digit residues; a triangular unit fires at
    #     each multiple of p; the winner writes its 3 digits.
    u = 0
    for j, p in enumerate(PRIMES):
        for c in TEETH[p]:
            for i in range(5):
                for d in range(10):
                    b1.mlp.fc1.weight.data[u, H_ + 10 * i + d] = residue(d * 10 ** (4 - i), p)
            b1.mlp.fc1.bias.data[u] = -c
            b1.mlp.mid.weight.data[j, u] = 1.0
            u += 1
        for jd in range(3):
            b1.mlp.fc2.weight.data[R_ + jd, j] = int(f"{p:03d}"[jd])

    # --- Block 2 attention: score -BETA (i - (s-5))^2, up to a constant, so step s-5 reads digit
    #     i = s-5. The score expands to q = (2(s-5), -1), k = (i, i^2).
    b2.attn.W_q.weight.data[0, S_] = 2.0 * BETA * math.sqrt(2)
    b2.attn.W_q.weight.data[0, ONE_] = -10.0 * BETA * math.sqrt(2)
    b2.attn.W_q.weight.data[1, ONE_] = -BETA * math.sqrt(2)
    b2.attn.W_k.weight.data[0, S_] = 1.0
    b2.attn.W_k.weight.data[1, S2_] = 1.0
    b2.attn.W_v.weight.data[0, V_] = 1.0
    b2.attn.W_o.weight.data[K_, 0] = 1.0

    # --- Block 2 MLP: copy the digit unless a factor exists; each factor digit opens at its step.
    b2.mlp.fc1.weight.data[0, K_] = 1.0
    b2.mlp.fc1.weight.data[0, R_:R_ + 3] = -9.0               # any factor digit closes the copy
    for jd in range(3):
        b2.mlp.fc1.weight.data[1 + jd, R_ + jd] = 1.0
        b2.mlp.fc1.weight.data[1 + jd, STEP_ + jd] = 9.0
        b2.mlp.fc1.bias.data[1 + jd] = -9.0
    b2.mlp.fc2.weight.data[A_, :4] = 1.0

    # --- Head: logit_d = 2 d A - d^2 is largest for the digit nearest to A.
    for d in range(10):
        model.head.weight.data[d, A_] = 2.0 * d
        model.head.weight.data[d, ONE_] = -float(d * d)
    model.head.weight.data[10, ONE_] = -1e6                   # never emit '='
    exhaustive_check(model, task)


def exhaustive_check(model, task) -> None:
    """Verification only (not part of the model): test every n the prompt can hold."""
    def spf(n):
        for q in range(2, 317):
            if q * q > n:
                break
            if n % q == 0:
                return q
        return n
    ns = list(range(2, 100000))
    ids = torch.tensor([task.encode(f"{n:05d}={spf(n):05d}") for n in ns])
    ok, wrong = 0, []
    with torch.no_grad():
        for b in range(0, len(ns), 5000):
            x = ids[b:b + 5000]
            hit = (model(x[:, :10])[:, 5:10].argmax(-1) == x[:, 6:11]).all(1)
            ok += int(hit.sum())
            wrong += [ns[b + int(i)] for i in (~hit).nonzero()[:, 0]]
    print(f"EXHAUSTIVE {ok}/{len(ns)} wrong[:10]={wrong[:10]} D={D_MODEL} blocks={BLOCKS}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "tri_beta_10"
model_description = "BETA 10: the leak e^-10 times the largest residue 150 is about 0.007, predicted to be safe where BETA 5 is not."


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
