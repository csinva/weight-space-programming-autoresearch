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


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


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
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Counting sort as a concave score. With cumulative counts C_j = #{i: x_i <= j},
    the k-th smallest digit is the smallest d with C_d > k + 1/2, so it maximises
        F_k(d) = sum_{j<d} (k + 1/2 - C_j),
    whose increments change sign exactly once (C_j is non-decreasing).

    Channels: j = 0..8 = "is the digit <= j?" (digit 9 needs none). No position channel:
    position lives on the all-ones direction of the same 9 channels.
      token_emb  eps * thermometer code of the digit ('=' is the zero vector)
      pos_emb    prompt: +p on every channel; output step k: -(5p/eps + k + 1/2)
      W_q, W_k   read the channel mean, opposite signs: output positions attend to the
                 prompt. sharp is small enough that the token part of the mean barely
                 skews the attention (p >> eps), and the +/- offset still decides it.
      W_v, W_o   mean of (thermometer + p/eps) * 5 = C_j + 5p/eps; the residual position
                 embedding cancels the offset, leaving h_j = C_j - k - 1/2
      head       F: row d is -1 on channels j < d (a discrete integral);
                 the '=' row is the mean of the digit rows, always below the maximum.
    The residual leak eps * thermometer(previous digit) is harmless for eps < 1/2.
    """
    torch.manual_seed(0)
    eps, p_, sharp = 0.01, 1.0, 0.07
    offset = 5 * p_ / eps
    block = model.blocks[0]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for digit in range(10):
            model.token_emb.weight[digit, digit:9] = eps
            model.head.weight[digit, :digit] = -1.0
        model.head.weight[task.stoi["="]] = model.head.weight[:10].mean(dim=0)
        model.pos_emb.weight[:5] = p_
        k = torch.arange(model.max_seq_len - 5)
        model.pos_emb.weight[5:] = -(offset + k + 0.5)[:, None].expand(-1, 9)

        attn = block.attn
        attn.W_q.weight[0, :] = -sharp / 9
        attn.W_k.weight[0, :] = sharp / 9
        attn.W_v.weight[:] = torch.eye(9) / eps
        attn.W_o.weight[:] = 5 * torch.eye(9)
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
model_shorthand_name = "e07_thermometer_d9"
model_description = "d_model=9, one channel per threshold: position lives on the all-ones direction, so the attention output is directly C_j - k - 1/2 and the head integrates it."


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
