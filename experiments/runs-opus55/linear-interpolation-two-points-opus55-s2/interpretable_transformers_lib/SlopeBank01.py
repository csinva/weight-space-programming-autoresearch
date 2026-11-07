"""
Interpretable transformer for `linear-interpolation-two-points`:
prompt "x1,y1;x2,y2;x=" -> two-digit answer y, e.g. "4,16;5,20;3=" -> "12".

Hand-written circuit (never trained). Every d_model channel has a name:
  READ   Block 1 attention: one head per number; attention weights 10:1 on the
         tens:units characters implement place value.
  TEST   Block 1 MLP: err_k = |(y2-y1) - k(x2-x1)| for each slope hypothesis k.
  FIT    Block 2 MLP: y = sum_k relu(y1 + k(x-x1) - M err_k); only the k that
         fits survives the gate.
  PRINT  Block 3 MLP: out = (y-4.5)/10 at the tens step, y-10*tens at the units
         step. The linear head rounds it: logit_d = 2d*out - d^2, the tangents
         of -(out-d)^2.

Usage:
    uv run interpretable_transformer.py --task linear-interpolation-two-points
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

SLOPES = [1, 2, 3, 4, 5]
READS = ["x1", "y1", "x2", "y2", "x"]
NAMES = (["dig", "one"] + [f"s_{r}" for r in READS] + ["at_tens", "at_units"] + READS
         + [f"err{k}" for k in SLOPES] + ["y", "out"])
C = {n: i for i, n in enumerate(NAMES)}


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, n_heads, bias=False)
        self.W_k = nn.Linear(d_model, n_heads, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.W_q(x).transpose(1, 2)[..., None]              # B H T 1
        k = self.W_k(x).transpose(1, 2)[..., None]
        v = self.W_v(x).transpose(1, 2)[..., None]
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).squeeze(-1).transpose(1, 2))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if n_heads else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    """3-block causal transformer: READ+TEST, FIT, PRINT. No LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        d_model = len(NAMES)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, len(READS), 2 * len(SLOPES)),
            Block(d_model, 0, len(SLOPES)),
            Block(d_model, 0, 2),
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in closed form. No training."""
    for p in model.parameters():
        p.zero_()

    # Embeddings: a token is its digit value; a position is a set of flags.
    for d in range(10):
        model.token_emb.weight[task.stoi[str(d)], C["dig"]] = d
    pos = model.pos_emb.weight
    pos[:, C["one"]] = 1
    L = 40.0  # attention sharpness: selected characters beat the rest by e^L
    place_value = {"x1": {0: L}, "y1": {2: L + math.log(10), 3: L}, "x2": {5: L},
                   "y2": {7: L + math.log(10), 8: L}, "x": {10: L}}
    for r, slots in place_value.items():
        for p, score in slots.items():
            pos[p, C[f"s_{r}"]] = score
    pos[11, C["at_tens"]] = 1
    pos[12, C["at_units"]] = 1

    # READ: head h asks "where is number h?" and returns the digit value there.
    # A two-digit number reads (10a + b) / 11, so W_o rescales by 11.
    read, test = model.blocks[0].attn, model.blocks[0].mlp
    for h, r in enumerate(READS):
        read.W_q.weight[h, C["one"]] = 1
        read.W_k.weight[h, C[f"s_{r}"]] = 1
        read.W_v.weight[h, C["dig"]] = 1
        read.W_o.weight[C[r], h] = 11 if r.startswith("y") else 1

    # TEST: err_k = relu(e_k) + relu(-e_k), with e_k = (y2-y1) - k(x2-x1).
    for i, k in enumerate(SLOPES):
        for j, sign in enumerate([1, -1]):
            row = 2 * i + j
            test.fc1.weight[row, C["y2"]] = sign
            test.fc1.weight[row, C["y1"]] = -sign
            test.fc1.weight[row, C["x2"]] = -sign * k
            test.fc1.weight[row, C["x1"]] = sign * k
            test.fc2.weight[C[f"err{k}"], row] = 1

    # FIT: unit k passes y1 + k(x-x1), unless slope k failed its test.
    fit = model.blocks[1].mlp
    M = 200.0
    for i, k in enumerate(SLOPES):
        fit.fc1.weight[i, C["y1"]] = 1
        fit.fc1.weight[i, C["x"]] = k
        fit.fc1.weight[i, C["x1"]] = -k
        fit.fc1.weight[i, C[f"err{k}"]] = -M
        fit.fc2.weight[C["y"], i] = 1

    # PRINT: unit 0 drives the tens step, unit 1 the units step; each is gated
    # off at the other step. dig is 0 at '=' and the tens digit at the next step.
    prt = model.blocks[2].mlp
    prt.fc1.weight[0, C["y"]] = 1
    prt.fc1.weight[0, C["at_units"]] = -M
    prt.fc1.bias[0] = -4.5
    prt.fc2.weight[C["out"], 0] = 0.1
    prt.fc1.weight[1, C["y"]] = 1
    prt.fc1.weight[1, C["dig"]] = -10
    prt.fc1.weight[1, C["at_tens"]] = -M
    prt.fc2.weight[C["out"], 1] = 1

    # Head: logit_d = 2d*out - d^2 rounds `out`; punctuation is never printed.
    for tok, i in task.stoi.items():
        if tok.isdigit():
            d = int(tok)
            model.head.weight[i, C["out"]] = 2 * d
            model.head.weight[i, C["one"]] = -d * d
        else:
            model.head.weight[i, C["one"]] = -1e3


def diagnostics(model: SimpleTransformer, task) -> None:
    """Sanity checks only (not used by the weights): non-eval-seed data."""
    from collections import Counter
    from fractions import Fraction
    exs = task.generate_examples(20000, seed=1)
    ms, bs, same = Counter(), Counter(), 0
    for e in exs:
        a, b, c = e.prompt[:-1].split(";")
        x1, y1 = map(int, a.split(","))
        x2, y2 = map(int, b.split(","))
        if x1 == x2:
            same += 1
            continue
        m = Fraction(y2 - y1, x2 - x1)
        ms[str(m)] += 1
        bs[str(y1 - m * x1)] += 1
    print("slopes:", sorted(ms.items()), "\nintercepts:", sorted(bs.items()), "\nx1==x2:", same)
    acc, _ = evaluate(model, task, n_samples=5000, seed=7, device="cpu", verbose=False)
    print(f"diagnostic accuracy on 5000 seed-7 examples: {acc:.4f}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SlopeBank01"
model_description = "READ heads with place-value attention; slope-hypothesis tests for k=1..5; gated extrapolation; tangent-line rounding head. No LayerNorm."


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
    diagnostics(model, task)
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
