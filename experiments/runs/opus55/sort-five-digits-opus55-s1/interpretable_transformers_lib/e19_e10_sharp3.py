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

class PositionAttention(nn.Module):
    """Causal attention whose scores come from one number b per position; tokens are values."""

    def __init__(self):
        super().__init__()
        self.sharp = nn.Parameter(torch.ones(1))
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, b: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        T = values.shape[1]
        scores = -self.sharp * (b @ b.T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=b.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.gain * (attn @ values)


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 9,
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
        self.pos_emb = nn.Embedding(max_seq_len, 1)
        self.tilt = nn.Parameter(torch.ones(d_model))
        self.attn = PositionAttention()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        b = self.pos_emb(torch.arange(T, device=ids.device))
        h = self.attn(b, self.token_emb(ids)) + b * self.tilt
        return h @ self.token_emb.weight.T  # head tied to the token embedding


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Counting sort as a concave score. With cumulative counts C_j = #{i: x_i <= j},
    the k-th smallest digit is the smallest d with C_d > k + 1/2, so it maximises
        F_k(d) = sum_{j<d} (k + 1/2 - C_j),
    whose increments change sign exactly once (C_j is non-decreasing).

    Channels: j = 0..8 = "is the digit <= j?" (digit 9 needs none).
      token_emb  thermometer code of the digit ('=' is the mean of the digit codes)
      pos_emb    one number b_t: +1 on the prompt, -(k + 1/2) on output step k
      attention  scores -sharp * b_t * b_i: output positions attend to the prompt, with
                 hardness set by sharp. Tokens are values only, so nothing skews it.
      gain       5 (mean -> sum), so the attention output is exactly the counts C_j
      tilt       all ones: the rank threshold b_t is subtracted from every channel,
                 and only position (not the token) takes the residual path
      head       tied to token_emb: logit_d = <h, T(d)> = F_k(d) + const, using the
                 kernel <T(x), T(d)> = 9 - max(x, d). The '=' row is the mean of the
                 digit rows, always below the maximum of a non-constant score.
    """
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for digit in range(10):
            model.token_emb.weight[digit, digit:9] = 1.0
        model.token_emb.weight[task.stoi["="]] = model.token_emb.weight[:10].mean(dim=0)
        model.pos_emb.weight[:5] = 1.0
        model.pos_emb.weight[5:, 0] = -(torch.arange(model.max_seq_len - 5) + 0.5)
        model.tilt.fill_(1.0)
        model.attn.sharp.fill_(3.0)
        model.attn.gain.fill_(5.0)
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
model_shorthand_name = "e19_e10_sharp3"
model_description = "e10 with attention sharpness 3 instead of 30: tests how hard the gate onto the prompt positions must be."


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
