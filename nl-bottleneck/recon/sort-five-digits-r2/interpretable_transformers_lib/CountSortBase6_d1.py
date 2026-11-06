"""
Interpretable transformer for character-level sequence tasks.

Hand-written counting sort for `sort-five-digits` (no training).

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
# One layer, one head, d_model = 1, no LayerNorm, no MLP.
#   token_emb[d] = 6^(8-d) for digits 0..8; 0 for '9' and '='.
#   pos_emb[p]   = sign: +1 for prompt positions (0..5), -1 for fed-back answer positions.
#   x_t = token_emb * pos_emb   (signed value)
#   attention: q = k = 0 -> uniform causal average; multiplied by (t+1) -> sum R.
#   head: logit_d = w_d * R + b_d  ==  sum_{j=d..8} (R/tau_j - 1).
# Params: 11 + 16 + 4 + 11 + 11 = 53.

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        avg = attn @ v                                    # causal mean
        count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[None, :, None]
        return self.W_o(avg * count)                      # sum R


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16, d_model: int = 1, **kw):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn = CausalSelfAttention(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.token_emb(ids) * self.pos_emb(pos)[None, :, :]
        R = self.attn(x)
        return self.head(R)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    model.double()  # float64: subtracting emitted digits cancels large terms
    stoi = task.stoi
    w = {d: 6.0 ** (8 - d) for d in range(9)}
    tau = {d: 0.9 * w[d] for d in range(9)}
    prompt_len = 6  # "DDDDD="
    with torch.no_grad():
        model.token_emb.weight.zero_()
        for d in range(9):
            model.token_emb.weight[stoi[str(d)], 0] = w[d]
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, 0] = 1.0 if p < prompt_len else -1.0
        model.attn.W_q.weight.fill_(0.0)
        model.attn.W_k.weight.fill_(0.0)
        model.attn.W_v.weight.fill_(1.0)
        model.attn.W_o.weight.fill_(1.0)
        # logit_d = sum_{j=d..8} (R/tau_j - 1)
        model.head.weight.zero_()
        model.head.bias.zero_()
        for d in range(9):
            model.head.weight[stoi[str(d)], 0] = sum(1.0 / tau[j] for j in range(d, 9))
            model.head.bias[stoi[str(d)]] = -float(9 - d)
        model.head.weight[stoi["9"], 0] = 0.0
        model.head.bias[stoi["9"]] = 0.0
        model.head.bias[stoi["="]] = -1e9


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "CountSortBase6_d1"
model_description = ("1-layer 1-head d_model=1 counting sort: signed base-6 digit weights 6^(8-d), "
                     "uniform causal attention x(t+1) gives remaining-count sum R; readout thresholds 0.9*w_d; float64.")


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
