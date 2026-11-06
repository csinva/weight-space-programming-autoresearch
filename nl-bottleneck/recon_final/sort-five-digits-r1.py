"""
Interpretable transformer for character-level sequence tasks.

Hand-written counting-sort transformer for `sort-five-digits`
(1 layer, 1 head, d_model = 1, 53 parameters, float64).

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task sort-five-digits
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
    """Single head, d_model = 1. With W_q = W_k = 0 the scores are all zero,
    so the head computes a uniform causal average of the values."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return self.W_o(attn @ v)


class SimpleTransformer(nn.Module):
    """One layer, one head, d_model = 1 counting-sort transformer.

    x_t     = token_value[tok_t] * pos_sign[t]        (+1 prompt, -1 answer)
    a_t     = causal mean of x  (zero attention scores)
    R_t     = (t+1) * a_t                              (running signed sum)
    logits  = head(R_t)                                (threshold readout)
    """

    def __init__(self, vocab_size: int, max_seq_len: int = 16, d_model: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)   # digit weights 6^(8-d)
        self.pos_sign = nn.Embedding(max_seq_len, d_model)   # +1 prompt / -1 answer
        self.attn = CausalSelfAttention(d_model, 1)
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.token_emb(ids) * self.pos_sign(pos)[None, :, :]
        a = self.attn(x)
        scale = (pos + 1).to(a.dtype)[None, :, None]
        R = a * scale
        return self.head(R)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    model.double()  # float64: subtracting emitted digits cancels large terms
    stoi = task.stoi
    with torch.no_grad():
        # Token values: digit d -> 6^(8-d); digit 9 and '=' -> 0.
        w = torch.zeros(model.vocab_size, dtype=torch.float64)
        for d in range(9):
            w[stoi[str(d)]] = 6.0 ** (8 - d)
        model.token_emb.weight[:, 0] = w

        # Sign: prompt digits (positions 0..4) +1; '=' (value 0 anyway) and
        # fed-back answer digits (positions >= 6) -1.
        n_prompt_digits = task.prompt_len - 1
        sign = torch.full((model.max_seq_len,), -1.0, dtype=torch.float64)
        sign[:n_prompt_digits] = 1.0
        model.pos_sign.weight[:, 0] = sign

        # Attention: zero scores -> uniform causal average; identity value/output.
        model.attn.W_q.weight.fill_(0.0)
        model.attn.W_k.weight.fill_(0.0)
        model.attn.W_v.weight.fill_(1.0)
        model.attn.W_o.weight.fill_(1.0)

        # Readout: logit_d = sum_{j=d..8} (R/tau_j - 1), tau_j = 0.9 * 6^(8-j).
        # logit_9 = 0 (empty sum); '=' gets a large negative bias.
        tau = [0.9 * 6.0 ** (8 - j) for j in range(9)]
        W = torch.zeros(model.vocab_size, dtype=torch.float64)
        b = torch.zeros(model.vocab_size, dtype=torch.float64)
        for d in range(10):
            W[stoi[str(d)]] = sum(1.0 / tau[j] for j in range(d, 9))
            b[stoi[str(d)]] = -float(9 - d)
        W[stoi["="]] = 0.0
        b[stoi["="]] = -1e9
        model.head.weight[:, 0] = W
        model.head.bias[:] = b


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CountingSortBase6_d1"
model_description = ("1 layer, 1 head, d_model=1 counting sort: digit d worth 6^(8-d), sign +1 prompt/-1 answer, "
                     "zero attention scores average, times (t+1) gives remaining-count sum R; "
                     "threshold readout tau_d=0.9*6^(8-d); float64.")


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
