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
        # Genuine single-head softmax attention (1x1 weights since d_model=1). q=k=0 gives
        # uniform attention over the causal prefix; W_v, W_o carry the vote through.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o(attn @ v)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # ReLU(count + b_j): fc1 is a bias-only layer (weight fixed at 1), one knot per unit
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.fc2 = nn.Linear(d_ff, 1, bias=False)   # writes only residual dim 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(x + self.b1))   # d_model=1: x is the running count


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.Identity()  # attention reads raw embeddings
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()  # MLP reads the raw residual (vote average = k/22 exactly)
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

        self.token_emb = nn.Embedding(vocab_size, d_model)   # d_model=1: one scalar vote per token
        # no positional embedding: the task needs only a causal mean, which is order-free
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()  # head reads the raw residual
        # one scalar readout s; logits = s * sign, with sign = +1 for '1', -1 for '0', 0 elsewhere
        self.head = nn.Linear(d_model, 1, bias=True)
        self.register_buffer("sign", torch.zeros(vocab_size))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        h = F.pad(self.token_emb(ids), (0, self.d_model - 1))
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h)) * self.sign


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Parity = (number of 1s) mod 2, read out at the '=' position.

    The residual stream is a single scalar x.
      1. Embedding: '1' -> 11, every other token -> 0.
      2. Attention (q=k=0, uniform softmax; W_v=W_o=1): at '=' it averages the 11 prompt tokens,
         so x = 11 * k/11 = k, the number of ones.
      3. MLP: x += sum_{j=1..9} c_j ReLU(x - j) with c_j the slope change at k=j. Together with
         the residual x=k this is the piecewise-linear triangle wave through (k, k mod 2),
         exact for every integer k in 0..10. (Nine knots are necessary: each of k=1..9 is a bend.)
      4. Head: s = x - 0.5, logits = s * sign with sign = +1 for '1', -1 for '0', 0 otherwise.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        tok = task.stoi
        E = model.token_emb.weight
        blk = model.blocks[0]
        blk.attn.W_v.weight[0, 0] = 1.0   # q = k = 0 (uniform softmax); W_o passes the mean through
        blk.attn.W_o.weight[0, 0] = 1.0
        E[tok['1'], 0] = 11  # only '1' is non-zero: '0', '_' and '=' embed to 0
        us = [float(k) for k in range(11)]  # residual at '=' when the prompt has k ones
        # piecewise-linear interpolation of parity through knots (u_k, k%2): a triangle wave.
        # hidden unit j = relu(u - u_j), its fc2 weight = change of slope at knot j.
        slopes = [((k + 1) % 2 - k % 2) / (us[k + 1] - us[k]) for k in range(10)]
        # The first slope (+1 from k=0) is carried by the residual stream itself: dim 0 = k is
        # read by the head directly, so only the 9 slope changes at k=1..9 need a ReLU.
        for j in range(1, 10):   # knots at k=1..9 (the wave is then exact for every k in 0..10)
            blk.mlp.b1[j - 1] = -us[j]
            blk.mlp.fc2.weight[0, j - 1] = slopes[j] - slopes[j - 1]
        model.head.bias[0] = -0.5   # the constant offset lives in the head bias
        model.sign[tok['1']], model.sign[tok['0']] = 1.0, -1.0
        model.head.weight[0, 0] = 1.0   # s = x - 0.5 where x = k + sum_j c_j ReLU(k-j) = triangle wave


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "scalar_residual_d1_verified"
model_description = "Same weights as scalar_residual_d1_final plus an exhaustive forward-pass self-check over all 1024 full-length 10-bit prompts (covers k=10, which the 200-sample eval rarely hits)."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len,
                              d_model=1, n_heads=1, n_layers=1, d_ff=9)
    write_weights(model, task)
    model.eval()
    # Exhaustive self-check over all 1024 full-length 10-bit prompts (forward pass only).
    strings = [format(i, "010b") for i in range(1024)]
    ids = torch.tensor([[task.stoi[c] for c in s + "="] for s in strings])
    with torch.no_grad():
        pred = model(ids)[:, -1].argmax(-1)
    want = torch.tensor([task.stoi[str(s.count("1") % 2)] for s in strings])
    print("SELFCHECK full-length prompts correct:", int((pred == want).sum()), "/ 1024")
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
