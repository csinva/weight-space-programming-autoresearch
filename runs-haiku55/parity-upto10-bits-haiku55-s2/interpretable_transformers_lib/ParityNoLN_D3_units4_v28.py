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
        self.ln1 = nn.Identity()  # no LayerNorm: the circuit is exactly linear before the MLP
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
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
        d_model: int = 3,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids)  # no positional embedding: causal attention alone sees the prefix
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def _ln_std(v: list[float]) -> float:
    """Std used by nn.LayerNorm (biased variance + eps) for a fixed vector v."""
    m = sum(v) / len(v)
    var = sum((x - m) ** 2 for x in v) / len(v)
    return math.sqrt(var + 1e-5)


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built parity circuit (no training).

    Residual layout (d_model = 3):
      dim 0: is-'1' flag (embedding)     dim 1: running count k (attn)
      dim 2: parity P (MLP)

    Circuit:
      1. Attention (uniform over the 11 causal positions at '=') averages dim0;
         times 11 this gives exactly k = #ones in dim 1.
      2. MLP: relu(k - j) for j = 0..9, then the triangle wave
         parity(k) = relu(k) + sum_{j=1..9} 2(-1)^j relu(k-j), in dim 2.
      3. Head: logit('1') = P, logit('0') = 0; the tie at P=0 resolves to '0'.
    No LayerNorm is used, so the count is exact (no approximation).
    """
    D = model.d_model
    stoi = task.stoi
    assert D == 3 and model.n_layers == 1
    blk = model.blocks[0]
    n_units = model.d_ff                     # triangle wave needs 10 units for k in 0..10

    with torch.no_grad():
        # --- embeddings: dim0 = is-'1', everything else 0 ---
        E = torch.zeros(model.vocab_size, D)
        E[stoi["1"], 0] = 1.0
        model.token_emb.weight.copy_(E)

        # --- attention: uniform over the 11 causal positions at '=' ---
        # average of dim0 over 11 tokens = k/11, times 11 -> k
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        Wv = torch.zeros(D, D)
        Wv[1, 0] = 11.0
        blk.attn.W_v.weight.copy_(Wv)
        Wo = torch.zeros(D, D)
        Wo[1, 1] = 1.0                       # count lives in dim 1 (= k at '=')
        blk.attn.W_o.weight.copy_(Wo)

        # --- MLP: relu(k - j) ---
        W1 = torch.zeros(n_units, D)
        b1 = torch.zeros(n_units)
        for j in range(n_units):
            W1[j, 1] = 1.0
            b1[j] = -j
        W2 = torch.zeros(D, n_units)
        b2 = torch.zeros(D)
        # triangle wave: parity(k) = relu(k) + sum_{j=1..9} 2(-1)^j relu(k-j)
        W2[2, 0] = 1.0
        for j in range(1, n_units):
            W2[2, j] = 2.0 * (-1) ** j
        blk.mlp.fc1.weight.copy_(W1)
        blk.mlp.fc1.bias.copy_(b1)
        blk.mlp.fc2.weight.copy_(W2)
        blk.mlp.fc2.bias.copy_(b2)

        # --- head: logit('1') = P, logit('0') = 0 (ties at P=0 resolve to '0') ---
        H = torch.zeros(model.vocab_size, D)
        H[stoi["1"], 2] = 1.0
        model.head.weight.copy_(H)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ParityNoLN_D3_units4_v28"
model_description = "d_model=3 (is-1, count=k, P): no '=' flag needed since the count is then exactly k; thresholds are relu(k-j) with bias -j; head logit('1')=P."


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
