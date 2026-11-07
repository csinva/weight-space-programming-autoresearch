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
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 37,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 18,
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
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Residual layout (d_model=37):
    #   0..9 digit one-hot | 10..15 position one-hot (positions 5..10 only) | 16,17 constants +B,-B-(onehots)
    #   18..27 digit histogram (mean) | 28..36 output dims for digits 0..8 (digit 9 = 1 - sum of them)
    # Constants keep the residual zero-mean, so LayerNorm is ~ a constant rescale undone by its gain.
    # Attention: all positions attend uniformly to the 5 prompt digits -> digit counts.
    # MLP: at output step k (from position), I_d = clamp(C_d - k, 0, 1) = [cumcount(<=d) > k];
    #      answer digit = smallest d with I_d = 1, so logit_d = I_d - I_{d-1}.
    D = model.d_model
    B = 8.0
    TOK, POS, C1, C2, HIST, LOG = 0, 10, 16, 17, 18, 28   # POS dims only for positions 5..10
    for p in model.parameters():
        p.data.zero_()
    dig = [task.stoi[str(d)] for d in range(10)]
    with torch.no_grad():
        for d in range(10):
            model.token_emb.weight[dig[d], TOK + d] = 1.0   # '=' gets no one-hot
        model.token_emb.weight[:, C1] = B
        model.token_emb.weight[:, C2] = -B
        model.token_emb.weight[dig, C2] = -(B + 1)  # digit one-hot cancelled
        for p in range(5, task.seq_len):
            model.pos_emb.weight[p, POS + p - 5] = 1.0
            model.pos_emb.weight[p, C2] = -1.0      # position one-hot cancelled
        # positions 0..4 have no position dims

        g = math.sqrt((2 + B * B + (B + 1.5) ** 2) / D)
        blk = model.blocks[0]
        for ln in (blk.ln1, blk.ln2, model.final_ln):
            ln.weight.fill_(g)
        # query = const, key = "is prompt position 0..4"
        blk.attn.W_q.weight[0, C1] = 8.0
        blk.attn.W_k.weight[0, C1] = 1.0 / B            # key = 1 on prompt positions
        for p in range(5, 11):
            blk.attn.W_k.weight[0, POS + p - 5] = -1.0  # ... and 0 on positions 5..10
        for d in range(10):
            blk.attn.W_v.weight[HIST + d, dig[d]] = 1.0
            blk.attn.W_o.weight[HIST + d, HIST + d] = 1.0   # keep the mean (= count/5)
        blk.attn.W_v.weight[C2, C1] = -1.0 / B
        blk.attn.W_o.weight[C2, C2] = 1.0                   # -1 total: keeps residual zero-mean
        # hidden 2d: relu(C_d - k); hidden 2d+1: relu(C_d - k - 1)
        for d in range(9):
            for h, off in ((2 * d, 0.0), (2 * d + 1, -1.0)):
                blk.mlp.fc1.weight[h, HIST:HIST + d + 1] = 5.0   # count = 5 * mean
                for p in range(5, 10):
                    blk.mlp.fc1.weight[h, POS + p - 5] = -(p - 5.0)
                blk.mlp.fc1.bias[h] = off
                sgn = 1.0 if off == 0.0 else -1.0
                blk.mlp.fc2.weight[LOG + d, h] += sgn        # +I_d to logit d
                if d + 1 < 9:
                    blk.mlp.fc2.weight[LOG + d + 1, h] -= sgn  # -I_d to logit d+1
        for d in range(9):
            model.head.weight[dig[d], LOG + d] = 1.0
            model.head.weight[dig[9], LOG + d] = -1.0   # digit 9 = "else": 1 - sum of the other logits
        model.head.weight[dig[9], C1] = 1.0 / B          # the constant 1 (C1 = B)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "final_d37_ff18_B8"
model_description = "d=37, d_ff=18, B=8, 1 layer/1 head, ~7.4k params: positions 0..4 have no position dims, digit 9 is the 'else' logit (1 - sum of others); 1 layer/1 head: attention averages prompt digits into a histogram; MLP thresholds cumulative counts vs output index k to emit k-th smallest digit."


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
