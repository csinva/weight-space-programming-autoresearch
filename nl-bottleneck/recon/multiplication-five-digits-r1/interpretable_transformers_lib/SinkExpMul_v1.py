"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 2-layer attention-only transformer for
`multiplication-five-digits`: "AAAAA*BBBBB=" -> 10-digit product.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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

class CausalSelfAttention(nn.Module):
    """Causal attention with an explicit per-head dimension."""

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


class Block(nn.Module):
    """Attention-only block: no LayerNorm, no MLP."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    """2-layer attention-only transformer on a 7-channel float64 residual stream."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 7):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads=1, d_head=2),   # layer 1: b*A via sink-exponential head
            Block(d_model, n_heads=2, d_head=1),   # layer 2: P = A*B and V = emitted value
        ])
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

# Residual channels
D_DIG, D_LOG, D_ONE, D_F3, D_F4, D_BA, D_X = range(7)
M = 1000.0            # "minus infinity" offset for masked keys (exp(-M) == 0 relative)
L0 = -1000.0          # stands in for log(0): digit 0 attends only to the sink
SCALE1 = 1e17         # layer-1 sink strength
SCALE2 = 1e22         # layer-2 head-2 sink strength
LN10 = math.log(10.0)


def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights.

    Positions: A digits 0..4, '*' 5, B digits 6..10, '=' 11, answer 12..21.
    Channels:
      DIG  digit value d (token)          LOG  log d (token; log 0 -> -1000)
      ONE  1 for every token except '*'   F3   position fn (head-2 key score / layer-1 key part)
      F4   position fn (head-1 key score; equals -M - place at answer positions)
      BA   layer-1 output b*A             X    layer-2 output P - V + 0.5
    """
    torch.manual_seed(0)
    model.double()
    stoi = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # ---- token embedding
        te = model.token_emb.weight
        for d in range(10):
            i = stoi[str(d)]
            te[i, D_DIG] = d
            te[i, D_LOG] = math.log(d) if d > 0 else L0
            te[i, D_ONE] = 1.0
        te[stoi["="], D_ONE] = 1.0   # '*' keeps ONE = 0 so the log-b term vanishes at the sink

        # ---- position embedding
        pe = model.pos_emb.weight
        for t in range(model.max_seq_len):
            if t <= 4:                      # multiplicand A
                f3 = -M + (4 - t) * LN10
                f4 = -M
            elif t == 5:                    # '*' : layer-1 sink
                f3 = -M + math.log(SCALE1)
                f4 = -M
            elif t <= 10:                   # multiplier B
                f3 = -M
                f4 = (10 - t) * LN10
            elif t == 11:                   # '=' : head-2 sink
                f3 = math.log(SCALE2)
                f4 = -M - 10.0 ** (20 - t)
            else:                           # answer digits
                f3 = (21 - t) * LN10
                f4 = -M - 10.0 ** (20 - t)
            pe[t, D_F3] = f3
            pe[t, D_F4] = f4
            pe[t, D_X] = 0.5                # +0.5 rounding offset of the remainder

        # ---- layer 1: score = log b * ONE_key + (F3 - F4)_key
        #   A_i: log(b*10^(4-i)); '*': log(1e17); B: -M.  value a_i; out *1e17 -> BA
        a1 = model.blocks[0].attn
        s = math.sqrt(2.0)
        a1.W_q.weight[0, D_LOG] = s
        a1.W_q.weight[1, D_ONE] = s
        a1.W_k.weight[0, D_ONE] = 1.0
        a1.W_k.weight[1, D_F3] = 1.0
        a1.W_k.weight[1, D_F4] = -1.0
        a1.W_v.weight[0, D_DIG] = 1.0
        a1.W_o.weight[D_BA, 0] = SCALE1

        # ---- layer 2
        a2 = model.blocks[1].attn
        # head 1: weights B positions ∝ 10^(10-s); averages b_s*A; *11111 -> P
        a2.W_q.weight[0, D_ONE] = 1.0
        a2.W_k.weight[0, D_F4] = 1.0
        a2.W_v.weight[0, D_BA] = 1.0
        a2.W_o.weight[D_X, 0] = 11111.0
        # head 2: answer positions ∝ 10^(21-s) against 1e22 zero-valued sink at '='; -> -V
        a2.W_q.weight[1, D_ONE] = 1.0
        a2.W_k.weight[1, D_F3] = 1.0
        a2.W_v.weight[1, D_DIG] = 1.0
        a2.W_o.weight[D_X, 1] = -SCALE2

        # ---- tropical readout: logit_k = k*X - place*k(k+1)/2, place = -(F4 + M*ONE)
        W = model.head.weight
        for k in range(10):
            i = stoi[str(k)]
            W[i, D_X] = k
            W[i, D_F4] = k * (k + 1) / 2
            W[i, D_ONE] = M * k * (k + 1) / 2
        W[stoi["*"], D_ONE] = -1e15
        W[stoi["="], D_ONE] = -1e15


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SinkExpMul_v1"
model_description = "2-layer attention-only float64 7-channel model: sink-exponential head computes b*A, layer 2 sums to P=A*B and subtracts emitted value V via a 1e22 sink; tropical argmax readout floor(X/place)."


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
