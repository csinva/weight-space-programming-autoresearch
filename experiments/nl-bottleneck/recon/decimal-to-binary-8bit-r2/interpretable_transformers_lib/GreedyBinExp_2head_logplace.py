"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 1-layer, 2-head, attention-only transformer for
`decimal-to-binary-8bit`: prompt "DDD=" -> answer "BBBBBBBB" (MSB first).

Algorithm: exact greedy binary expansion. At the step emitting bit k,
r = n - S (S = value of bits already emitted); output 1 iff r >= 2^k.
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
# Residual stream dims:
#   V    scalar token value (digit face value, 0/1 for bits, 0.5 for '=')
#   L    log-place-value feature (ln100, ln10, 0 for digits; (11-p)ln2 for '=' and bits)
#   SGN  +1 on digit positions (p<3), -1 elsewhere (region bias, scaled by 6)
#   EQ   1 at the '=' position
#   QP   (3-t)*ln2 : query-side term so '=' gets weight 2^(k+1)  (k = 10-t)
#   ONE  constant 1
#   A0   head-0 output (n/111)
#   A1   head-1 output ((S+2^k)/256)
D = 8
V, L, SGN, EQ, QP, ONE, A0, A1 = range(8)
DQK = 3


class TwoHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2, d_qk: int = DQK):
        super().__init__()
        self.n_heads = n_heads
        self.W_q = nn.Parameter(torch.zeros(n_heads, d_model, d_qk))
        self.W_k = nn.Parameter(torch.zeros(n_heads, d_model, d_qk))
        self.W_v = nn.Parameter(torch.zeros(n_heads, d_model))   # scalar value per head
        self.W_o = nn.Parameter(torch.zeros(n_heads, d_model))   # write scalar back

    def forward(self, x):
        B, T, _ = x.shape
        q = torch.einsum("btd,hde->bhte", x, self.W_q)
        k = torch.einsum("btd,hde->bhte", x, self.W_k)
        v = torch.einsum("btd,hd->bht", x, self.W_v)
        scores = q @ k.transpose(-2, -1)                    # no 1/sqrt(d) scaling
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v.unsqueeze(-1)).squeeze(-1)          # (B,H,T)
        return torch.einsum("bht,hd->btd", out, self.W_o)


class SimpleTransformer(nn.Module):
    """1 layer, 2 heads, no MLP, no LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.attn = TwoHeadAttention(D)
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        h = h + self.attn(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    ln2 = math.log(2.0)
    BIAS = 6.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # token embedding: scalar value + constant
        te = model.token_emb.weight
        for ch, i in task.stoi.items():
            te[i, V] = 0.5 if ch == "=" else float(int(ch))
            te[i, ONE] = 1.0
        # positional features
        pe = model.pos_emb.weight
        for p in range(model.max_seq_len):
            if p < 3:
                pe[p, L] = [math.log(100.0), math.log(10.0), 0.0][p]
                pe[p, SGN] = 1.0
            else:
                pe[p, L] = (11 - p) * ln2
                pe[p, SGN] = -1.0
            if p == 3:
                pe[p, EQ] = 1.0
            pe[p, QP] = (3 - p) * ln2
        A = model.attn
        # Head 0: digits, softmax weights 100:10:1 -> n/111
        A.W_q[0, ONE, 0] = 1.0;  A.W_k[0, L, 0] = 1.0
        A.W_q[0, ONE, 1] = BIAS; A.W_k[0, SGN, 1] = 1.0
        A.W_v[0, V] = 1.0
        A.W_o[0, A0] = 1.0
        # Head 1: '=' and emitted bits; bit j weight 2^j, '=' weight 2^(k+1) -> (S+2^k)/256
        A.W_q[1, ONE, 0] = 1.0;   A.W_k[1, L, 0] = 1.0
        A.W_q[1, ONE, 1] = -BIAS; A.W_k[1, SGN, 1] = 1.0
        A.W_q[1, QP, 2] = 1.0;    A.W_k[1, EQ, 2] = 1.0   # '=' score 8ln2+(3-t)ln2 = (k+1)ln2
        A.W_v[1, V] = 1.0
        A.W_o[1, A1] = 1.0
        # Unembedding: "1" logit = 1 + 222*A0 - 512*A1 = 1 + 2(r - 2^k); "0" gets its negative
        W = model.head.weight
        i1, i0 = task.stoi["1"], task.stoi["0"]
        W[i1, A0] = 222.0;  W[i1, A1] = -512.0; W[i1, ONE] = 1.0
        W[i0, A0] = -222.0; W[i0, A1] = 512.0;  W[i0, ONE] = -1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "GreedyBinExp_2head_logplace"
model_description = ("1-layer 2-head attn-only: head0 place-value softmax over digits -> n/111, head1 over '=' and "
                     "emitted bits -> (S+2^k)/256; logit1 = 1+222a0-512a1 = 2(r-2^k)+1; +-6 positional region bias.")


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
