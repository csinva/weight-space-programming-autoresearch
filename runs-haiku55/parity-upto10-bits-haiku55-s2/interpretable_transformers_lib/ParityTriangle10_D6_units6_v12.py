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
        d_model: int = 6,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 6,
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
        self.final_ln = nn.LayerNorm(d_model)
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

    Residual layout (d_model = 6):
      dim 0: is-'1' flag (embedding)     dim 1: constant C (all tokens)
      dim 2: is-'=' flag (embedding)     dim 3: running count k-1 (attention)
      dim 4: parity P (MLP)              dim 5: 1 - P (MLP)

    Circuit:
      1. Attention (uniform over the causal prefix, all scores 0) at the '='
         position averages the value  phi = (dim0 - dim2) after LayerNorm,
         which equals +1 for each '1' and -1 for the '=' itself, and 0 for
         '0'/'_'. Scaling by G = 11*std gives dim 3 = (#ones) - 1 = k - 1.
      2. LayerNorm2 is almost linear in k because of the large constant C,
         so the MLP sees a times the affine count, with a chosen so that
         the thresholds land on integers.
      3. MLP: parity(k) = relu(k) + sum_{j=1..9} 2(-1)^j relu(k-j) for
         k in 0..10 (a triangle wave). Output: P in dim 4, 1-P in dim 5.
      4. Final LayerNorm keeps the antisymmetric part (dim4 - dim5), so the
         logits are logit['1'] = +(2P-1)/s and logit['0'] = -(2P-1)/s; the
         logits for '=' and '_' stay at 0.
    """
    D = model.d_model
    C = 139.0
    stoi = task.stoi
    assert D == 6 and model.n_layers == 1
    blk = model.blocks[0]
    n_units = model.d_ff                     # ablation knob: triangle needs 10 units for k in 0..10

    with torch.no_grad():
        # --- embeddings (no positional embedding) ---
        E = torch.zeros(model.vocab_size, D)
        E[stoi["0"], 1] = C
        E[stoi["1"], 0] = 1.0
        E[stoi["1"], 1] = C
        E[stoi["="], 1] = C
        E[stoi["="], 2] = 1.0
        E[stoi["_"], 1] = C
        model.token_emb.weight.copy_(E)

        # --- block 1: LayerNorm 1 + attention (count the ones) ---
        blk.ln1.weight.fill_(1.0)
        blk.attn.W_q.weight.zero_()          # all scores 0 -> uniform attention
        blk.attn.W_k.weight.zero_()
        s_a = _ln_std([1.0, C] + [0.0] * (D - 2))   # std of the '1' token (= '=' token)
        G = 11.0 * s_a
        Wv = torch.zeros(D, D)
        Wv[3, 0] = G                         # +G on the '1' flag
        Wv[3, 2] = -G                        # -G on the '=' flag
        blk.attn.W_v.weight.copy_(Wv)
        Wo = torch.zeros(D, D)
        Wo[3, 3] = 1.0
        blk.attn.W_o.weight.copy_(Wo)

        # --- block 1: LayerNorm 2 + MLP (triangle wave) ---
        blk.ln2.weight.fill_(1.0)
        blk.ln2.bias.zero_()
        s2 = _ln_std([0.0, C, 1.0, 4.0] + [0.0] * (D - 4))  # baseline k = 5
        a = D * s2 / (D - 1)
        sigma = (D + C) / (D - 1)            # linear LN2 maps the count to k - sigma
        W1 = torch.zeros(n_units, D)
        b1 = torch.zeros(n_units)
        W2 = torch.zeros(D, n_units)
        b2 = torch.zeros(D)
        for j in range(n_units):             # unit j computes relu(k - j)
            W1[j, 3] = a
            b1[j] = sigma - j
        # triangle wave: parity(k) = relu(k) + sum_{j=1..9} 2(-1)^j relu(k-j)
        W2[4, 0] = 1.0
        for j in range(1, n_units):
            W2[4, j] = 2.0 * (-1) ** j
        W2[5, :] = -W2[4, :]
        b2[5] = 1.0
        blk.mlp.fc1.weight.copy_(W1)
        blk.mlp.fc1.bias.copy_(b1)
        blk.mlp.fc2.weight.copy_(W2)
        blk.mlp.fc2.bias.copy_(b2)

        # --- final LayerNorm + head ---
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        H = torch.zeros(model.vocab_size, D)
        H[stoi["0"], 5] = 1.0
        H[stoi["0"], 4] = -1.0
        H[stoi["1"], 4] = 1.0
        H[stoi["1"], 5] = -1.0
        model.head.weight.copy_(H)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ParityTriangle10_D6_units6_v12"
model_description = "Ablation: v10 circuit with 6 ReLU units (j=0..5); expected to fail on counts k>=7."


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
