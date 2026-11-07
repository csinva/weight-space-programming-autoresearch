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
    """Parameter-free causal attention: uniform scores (all zero), causal mask,
    identity value/output -> each position averages itself and all earlier tokens."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        scores = torch.zeros(T, T, device=x.device, dtype=x.dtype)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return attn @ x


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """Scalar-residual transformer: no positional info, no LayerNorm, one
    parameter-free averaging attention layer, one 2-unit MLP block applied
    n_mlp_repeats times with tied weights, linear readout. 15 parameters."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 1,
                 d_ff: int = 2, n_mlp_repeats: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_mlp_repeats = n_mlp_repeats
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.attn = CausalSelfAttention()
        self.mlp = MLP(d_model, d_ff)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids)
        h = h + self.attn(h)
        for _ in range(self.n_mlp_repeats):
            h = h + self.mlp(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights.

    emb('0')=emb('_')=emb('=')=-1/4, emb('1')=-1/4+11/16.
    At '=' (position 11 of 11): residual = -1/4 + (-1/4 + c/16) = c/16 - 1/2.
    MLP (tied, x4): x + ReLU(x+1/2) - 4 ReLU(x) = 1/2 - 2|x|  (tent map),
    preserving parity of c while halving the denominator 16->8->4->2->1,
    ending at -1/2 (even) or +1/2 (odd).
    Readout: logit('1')=x, logit('0')=-x, others 0.
    """
    s = task.stoi
    with torch.no_grad():
        E = torch.full((task.vocab_size, 1), -0.25)
        E[s["1"], 0] = -0.25 + 11.0 / 16.0
        model.token_emb.weight.copy_(E)

        model.mlp.fc1.weight.copy_(torch.tensor([[1.0], [1.0]]))
        model.mlp.fc1.bias.copy_(torch.tensor([0.5, 0.0]))
        model.mlp.fc2.weight.copy_(torch.tensor([[1.0, -4.0]]))
        model.mlp.fc2.bias.zero_()

        H = torch.zeros(task.vocab_size, 1)
        H[s["1"], 0] = 1.0
        H[s["0"], 0] = -1.0
        model.head.weight.copy_(H)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ParityTentMapScalar15p"
model_description = ("Scalar residual, no positions: uniform causal averaging gives c/16-1/2 at '=', "
                     "then a tied 2-ReLU tent-map MLP applied 4x folds to +-1/2 by parity; logits +-x. 15 params.")


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
