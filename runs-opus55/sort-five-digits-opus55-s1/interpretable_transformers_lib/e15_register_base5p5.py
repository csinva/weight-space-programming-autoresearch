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

class UniformAttention(nn.Module):
    """Causal attention with all-zero scores: each position averages everything so far."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        T = values.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=values.device), diagonal=1)
        scores = torch.zeros(T, T, dtype=values.dtype, device=values.device)
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ values


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 1,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_sign = nn.Embedding(max_seq_len, 1)
        self.pos_gain = nn.Embedding(max_seq_len, 1)
        self.attn = UniformAttention()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.attn(self.token_emb(ids) * self.pos_sign(pos)) * self.pos_gain(pos)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """One-register counting sort. Digit d is worth w_d = B^(8-d) (digit 9 is worth 0), with
    B = 6, so a multiset of at most 5 digits packs into one number R whose base-6 digits
    are the counts. The largest term dominates, so the smallest remaining digit d is the one
    with w_d <= R <= 5 w_d, and 5 w_(d+1) < 0.9 w_d leaves a 10% gap for a threshold.

      token_emb  w_digit; '=' is worth 0
      pos_sign   +1 on the prompt, -1 on the answer: emitted digits are subtracted
      attention  uniform causal average over positions 0..t
      pos_gain   t + 1, turning the average back into the sum R
      head       tau_j = 0.9 w_j; logit_d = sum_{j=d..8} (R / tau_j - 1), so moving from
                 d + 1 to d gains exactly R / tau_d - 1, positive iff R > tau_d. The argmax
                 is the smallest d with R > tau_d, or 9 if there is none.
    Weights are float64, because subtracting emitted digits cancels large terms.
    """
    torch.manual_seed(0)
    model.double()
    B, frac = 5.5, 0.95
    tau = frac * B ** (8.0 - torch.arange(9.0, dtype=torch.float64))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.token_emb.weight[:9, 0] = tau / frac
        model.pos_sign.weight[:5] = 1.0
        model.pos_sign.weight[5:] = -1.0
        model.pos_gain.weight[:, 0] = torch.arange(1.0, model.max_seq_len + 1, dtype=torch.float64)
        for d in range(9):
            model.head.weight[d, 0] = (1 / tau[d:]).sum()
            model.head.bias[d] = -(9.0 - d)
        model.head.bias[task.stoi["="]] = -100.0
    check_all(model, task)


def check_all(model, task) -> None:
    """Diagnostic only: run generation on all 100000 inputs and report accuracy and margin."""
    n = task.prompt_len - 1
    digits = torch.cartesian_prod(*[torch.arange(10)] * n)
    ids = torch.cat([digits, torch.full((len(digits), 1), task.stoi["="])], dim=1)
    margin = float("inf")
    with torch.no_grad():
        for _ in range(task.answer_len):
            logits = model(ids)[:, -1]
            top = logits.topk(2, dim=-1).values
            margin = min(margin, (top[:, 0] - top[:, 1]).min().item())
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
    acc = (ids[:, -task.answer_len:] == digits.sort(dim=1).values).all(dim=1).float().mean().item()
    print(f"EXHAUSTIVE accuracy={acc:.6f} min_logit_margin={margin:.4f}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e15_register_base5p5"
model_description = "e14 with non-integer base 5.5: the counts no longer decode uniquely, but the leading term still dominates, so the sort should survive with a thinner margin."


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
