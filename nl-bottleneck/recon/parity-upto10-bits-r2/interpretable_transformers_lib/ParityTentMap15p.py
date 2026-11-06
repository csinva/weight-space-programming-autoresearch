"""
Interpretable transformer for character-level sequence tasks.

Hand-written 15-parameter parity model (no training):
  scalar residual, no positions, parameter-free uniform causal attention,
  one tied 2-unit ReLU MLP applied 4 times as a tent map, linear readout.

Usage:
    python interpretable_transformer.py --task parity-upto10-bits
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

class UniformCausalAttention(nn.Module):
    """Parameter-free attention: each position averages itself and all earlier tokens."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        counts = torch.arange(1, T + 1, dtype=x.dtype, device=x.device).view(1, T, 1)
        return x.cumsum(dim=1) / counts


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """1-dim residual, no positions, uniform causal attention, one tied MLP applied n_mlp_apply times."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 1,
                 d_ff: int = 2, n_mlp_apply: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_mlp_apply = n_mlp_apply
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.attn = UniformCausalAttention()
        self.mlp = MLP(d_model, d_ff)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids)
        h = h + self.attn(h)
        for _ in range(self.n_mlp_apply):
            h = h + self.mlp(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Count-then-tent-map parity, all values exact dyadic fractions."""
    s = task.stoi
    with torch.no_grad():
        # Embedding: '0', '_', '=' -> -1/4 ; '1' -> -1/4 + 11/16
        model.token_emb.weight.fill_(-0.25)
        model.token_emb.weight[s['1'], 0] = -0.25 + 11.0 / 16.0
        # At '=' (11 tokens): h = -1/4 + mean = -1/4 + (-1/4 + c/16) = c/16 - 1/2
        # MLP in residual form: x + ReLU(x + 1/2) - 4*ReLU(x) = 1/2 - 2|x|  (tent map)
        model.mlp.fc1.weight.copy_(torch.tensor([[1.0], [1.0]]))
        model.mlp.fc1.bias.copy_(torch.tensor([0.5, 0.0]))
        model.mlp.fc2.weight.copy_(torch.tensor([[1.0, -4.0]]))
        model.mlp.fc2.bias.zero_()
        # Readout: logit('1') = x, logit('0') = -x, others 0
        model.head.weight.zero_()
        model.head.weight[s['1'], 0] = 1.0
        model.head.weight[s['0'], 0] = -1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ParityTentMap15p"
model_description = ("Scalar residual, no positions; uniform causal averaging gives c/16-1/2 at '='; "
                     "tied 2-unit ReLU MLP applied 4x as tent map 1/2-2|x| folds to +-1/2; "
                     "logit('1')=x, logit('0')=-x. 15 params.")


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
