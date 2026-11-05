"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

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
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, f, use_attn=(i == 0)) for i, f in enumerate(ffs)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built 1-layer circuit for decimal -> 8-bit binary (prompt "197=", answer MSB first).

    Residual layout (d_model=48):
      0-10 token one-hot | 11-22 position one-hot (amp 3)
      23-25 hundreds digit (0-2), 26-35 tens digit, 36-45 units digit (gathered by attention)
      46 logit '1' | 47 constant 1 (query source and default '0' logit)
    Attention: head h always attends to prompt position h and copies that digit's one-hot.
    MLP: one unit per (N, j) such that bit j of N (MSB first) is 1. It is a 4-way AND of
         "hundreds==a", "tens==b", "units==c" and "this is output position 3+j", and it
         writes +4 into the '1' logit. No unit fires -> the constant makes '0' win.
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        for i in range(11):
            model.token_emb.weight[i, i] = 1.0
        for t in range(model.max_seq_len):
            if t < 12:
                model.pos_emb.weight[t, 11 + t] = 3.0
            model.pos_emb.weight[t, 47] = 1.0
        for b in model.blocks:
            b.ln2.weight.fill_(1.0)
            if b.use_attn:
                b.ln1.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        b0 = model.blocks[0]
        S = 2.0
        dh = model.d_model // model.n_heads
        DIG = [23, 26, 36]            # start dim of hundreds (3 values), tens, units
        NV = [3, 10, 10]              # number of values each digit can take
        LOG1 = 46                     # logit '1'
        CONST = 47                   # constant-1 dim (query source, default '0' logit)
        for h in range(3):
            b0.attn.W_q.weight[dh * h + dh - 1, CONST] = S
            b0.attn.W_k.weight[dh * h + dh - 1, 11 + h] = 1.0
            for d in range(NV[h]):
                b0.attn.W_v.weight[dh * h + d, d] = 1.0
                b0.attn.W_o.weight[DIG[h] + d, dh * h + d] = 1.0
        u = 0                          # one unit per (N, j) with bit j of N equal to 1
        for N in range(256):
            a, bb, c = N // 100, (N // 10) % 10, N % 10
            for j in range(8):
                if (N >> (7 - j)) & 1:
                    for dim in (DIG[0] + a, DIG[1] + bb, DIG[2] + c):
                        b0.mlp.fc1.weight[u, dim] = 1.0
                    b0.mlp.fc1.weight[u, 14 + j] = 2.0
                    b0.mlp.fc1.bias[u] = -13.3
                    b0.mlp.fc2.weight[LOG1, u] = 4.0
                    u += 1
        model.head.weight[0, CONST] = 6.0
        model.head.weight[1, LOG1] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "one_layer_Wo1_S2"
model_description = "One-layer circuit with unit digit-copy gain (W_o=1, LN rescales it) and gather-head query strength S=2."


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
    max_seq_len = task.seq_len
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, n_layers=1, d_ff=[1024],
                              d_model=48, n_heads=3)
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
