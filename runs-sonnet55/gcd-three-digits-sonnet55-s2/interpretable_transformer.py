"""
Interpretable transformer for character-level sequence tasks (task: gcd of two 3-digit numbers).

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model.
  * `write_weights()` runs once at model construction.

Usage:
    uv run interpretable_transformer.py --task gcd-three-digits
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
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------
# Feature-based design. The residual stream is a set of named "features", each stored
# one-hot style (value v on dim f). One extra "balance" dim always holds minus the sum of all the
# others, so the residual has exactly zero mean and LayerNorm is a pure rescale by a ~constant sigma
# (set by a big "const" feature of size C_MAG; C_MAG>=150 needed so the balance dim barely changes sigma).

C_MAG = 1000.0
N_ID = 1000  # numbers 0..999
N_CAND = 1000  # candidate gcd values 1..N_CAND-1 (larger gcds need a mess of rare inputs; N_CAND=1000 is exact)


class Layout:
    def __init__(self, vocab, maxlen):
        n = 0
        for name, k in (("tok", vocab), ("pos", maxlen), ("const", 1), ("slot", 60),
                        ("A", N_ID), ("H", N_CAND), ("g", 30), ("out", 10)):
            setattr(self, name, n)
            n += k
        self.B = self.A   # a and b share one count-vector: N[m] = #{a,b equal to m} (only the sum matters)
        self.n_feat = n
        self.bal = n                     # balance dim: always holds minus the sum of all other dims
        self.d_model = n + 1
        self.sigma = C_MAG * math.sqrt(2.0 / self.d_model)


class Attn(nn.Module):
    """Hard positional attention: head h copies the token at a fixed position (dk=1, dv=10 per head)."""

    def __init__(self, d_model, n_heads, dk, dv):
        super().__init__()
        self.n_heads, self.dk, self.dv = n_heads, dk, dv
        self.W_q = nn.Linear(d_model, n_heads * dk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * dk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * dv, bias=False)
        self.W_o = nn.Linear(n_heads * dv, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.dv).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        sc = sc.masked_fill(mask, float("-inf"))
        out = (sc.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class Block(nn.Module):
    def __init__(self, d_model, d_ff, attn=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class SimpleTransformer(nn.Module):
    """4 blocks. L1: attention gathers 6 digits + MLP one-hot encodes a and b.
    L2: common-divisor indicators. L3: greatest one (no prime multiple active). L4: position gate."""

    def __init__(self, vocab_size, max_seq_len=16):
        super().__init__()
        self.lay = Layout(vocab_size, max_seq_len)
        D = self.lay.d_model
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([
            Block(D, 2 * N_ID, Attn(D, 6, 1, 10)),
            Block(D, N_CAND),
            Block(D, N_CAND),
            Block(D, 30),
        ])
        self.final_ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model, task) -> None:
    L = model.lay
    sg = L.sigma
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dig = [task.stoi[str(d)] for d in range(10)]

        def wr(W, f, col, v=1.0):  # unit `col` writes feature f (and -v on the balance dim)
            W[f, col] += v
            W[L.bal, col] -= v

        def rd(W, row, f, coef=1.0, mag=1.0):  # row reads feature f (undoing LN scale)
            W[row, f] += coef * sg / mag

        te, pe = model.token_emb.weight, model.pos_emb.weight
        for t in range(te.shape[0]):
            te[t, L.tok + t] = 1
            te[t, L.const] = C_MAG
            te[t, L.bal] = -(1 + C_MAG)
        for p in range(pe.shape[0]):
            pe[p, L.pos + p] = 1
            pe[p, L.bal] = -1

        # ---- L1 attention: head h looks at fixed position src[h], copies its digit into slot h
        blk = model.blocks[0]
        at = blk.attn
        src = [0, 1, 2, 4, 5, 6]  # "abc,def=" -> a digits, b digits
        for h, ps in enumerate(src):
            rd(at.W_q.weight, h, L.const, coef=10.0, mag=C_MAG)
            rd(at.W_k.weight, h, L.pos + ps, coef=10.0)
            for d in range(10):
                rd(at.W_v.weight, h * 10 + d, L.tok + dig[d])
                wr(at.W_o.weight, L.slot + h * 10 + d, h * 10 + d)
        # ---- L1 MLP: one-hot of a (A) and b (B) from three digit slots (AND of three)
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for which, off in ((0, 0), (1, 3)):
            for m in range(N_ID):
                u = which * N_ID + m
                for j, dg in enumerate((m // 100, (m // 10) % 10, m % 10)):
                    rd(W1, u, L.slot + (off + j) * 10 + dg)
                b1[u] = -2.0
                wr(W2, (L.A if which == 0 else L.B) + m, u)
        # ---- L2: H[d] = [d | a] AND [d | b]
        blk = model.blocks[1]
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for d in range(1, N_CAND):
            for m in range(0, N_ID, d):
                rd(W1, d, L.A + m)  # N[m] counts a and b together
            b1[d] = -1.0
            wr(W2, L.H + d, d)
        # ---- L3: d is the gcd iff H[d] and no multiple k*d (k>=2) has H
        blk = model.blocks[2]
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        primes = list(range(2, N_ID))  # all proper multiples k*d, k>=2; d is the gcd iff none of them is a common divisor
        for d in range(1, N_CAND):
            rd(W1, d, L.H + d)
            for p in primes:
                if p * d < N_CAND:
                    rd(W1, d, L.H + p * d, coef=-1.0)
            for k, dg in enumerate((d // 100, (d // 10) % 10, d % 10)):
                wr(W2, L.g + k * 10 + dg, d)
        # ---- L4: gate digit k of the gcd by output position 7+k
        blk = model.blocks[3]
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for k in range(3):
            for t in range(10):
                u = k * 10 + t
                rd(W1, u, L.g + k * 10 + t)
                rd(W1, u, L.pos + 7 + k)
                b1[u] = -1.5
                wr(W2, L.out + t, u, 2.0)
        # ---- head: out[t] -> token logit
        for t in range(10):
            rd(model.head.weight, dig[t], L.out + t, coef=10.0)
        for blk in model.blocks:
            blk.ln1.weight.fill_(1)
            blk.ln2.weight.fill_(1)
        model.final_ln.weight.fill_(1)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "exact_final"
model_description = "Exact gcd via candidate divisors, single-rail features + balance dim (LayerNorm = fixed rescale). L1: 6 positional heads copy digits, MLP one-hots a and b into one shared count vector N. L2: H[d]=relu(sum_{m: d|m} N[m]-1). L3: d is gcd iff H[d] and no multiple k*d active; writes its 3 digits. L4: gate digit by output position."


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
