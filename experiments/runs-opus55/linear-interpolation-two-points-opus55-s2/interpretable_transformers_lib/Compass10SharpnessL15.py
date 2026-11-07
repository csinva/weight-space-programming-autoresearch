"""
Interpretable transformer for `linear-interpolation-two-points`:
prompt "x1,y1;x2,y2;x=" -> two-digit answer y, e.g. "4,16;5,20;3=" -> "12".

Hand-written circuit (never trained). Every d_model channel has a name:
  READ   Block 1 attention: one head per number; attention weights 10:1 on the
         tens:units characters implement place value. At the '=' step each head
         also attends to '=' itself (digit value 0), so numbers read at 1/10
         volume: the same circuit then computes y/10 for the tens digit.
         Prompt positions are muted: they attend to ',' so their registers stay 0.
  FIT    Block 2 attention (no MLP anywhere): one head per slope k. With
         e_k = (y2-y1) - k(x2-x1), q = e_k and key = -beta e_k give a self score
         of -beta e_k^2, so the head stays on itself only if slope k fits, and
         otherwise falls back to the zero-valued ','. Its value at itself is
         y1 + k(x-x1) - 10*dig, with dig the tens digit just printed (0 at '=').
  PRINT  Linear head rounds out: logit_d = 2d*out - d^2, the tangents of
         -(out-d)^2. '=' embeds -0.45 into out, turning round into floor.

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
NAMES = ["dig", "live", "phase", "stay", "a_cos", "a_sin", "y1", "dy", "dx", "dq", "out"]
C = {n: i for i, n in enumerate(NAMES)}


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, 2 * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, 2).transpose(1, 2)        # B H T 2
        k = self.W_k(x).view(B, T, H, 2).transpose(1, 2)
        v = self.W_v(x).transpose(1, 2)[..., None]              # B H T 1
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).squeeze(-1).transpose(1, 2))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    """2-block causal attention-only transformer: READ, FIT. No MLP, no LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        d_model = len(NAMES)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, len(READS)),
            Block(d_model, len(SLOPES)),
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
    pos[11:, C["live"]] = 1      # the two answer steps are the only ones that compute
    pos[:11, C["phase"]] = -1    # prompt: attend to the fallback
    pos[11, C["phase"]] = 1      # tens step: 1/10 volume; units step stays 0
    L = 15.0  # attention sharpness: selected characters beat the rest by e^L
    place_value = {"x1": {0: L}, "y1": {2: L + math.log(10), 3: L}, "x2": {5: L},
                   "y2": {7: L + math.log(10), 8: L}, "x": {10: L}}
    # Each number lies in its own compass direction; radius is its place value.
    angle = {r: 2 * math.pi * h / len(READS) for h, r in enumerate(READS)}
    for r, slots in place_value.items():
        for p, radius in slots.items():
            pos[p, C["a_cos"]] = radius * math.cos(angle[r])
            pos[p, C["a_sin"]] = radius * math.sin(angle[r])
    pos[1, C["stay"]] = 1000.0   # the ',' at position 1 is the zero-valued fallback,
    pos[11, C["stay"]] = 1100.0  # a fitting head outranks the fallback,
    pos[12, C["stay"]] = 1200.0  # with the newest step outranking the previous one.
    model.token_emb.weight[task.stoi["="], C["out"]] = -0.45

    # READ: head h asks "where is number h?" and returns the digit value there.
    # A two-digit number reads (10a + b) / 11, so W_o rescales by 11.
    # Key dim 0 selects the number. Key dim 1 gives the zero-valued '=' 9x the
    # number's total weight at the '=' step, for 1/10 volume. Muted positions
    # (query -1 on dim 1) go to ','. Position 0 can only see itself, but it reads
    # the same digit for every number, so its e_k is still 0.
    # W_o keeps only what the formula needs: y1, dy, dx and dq = x - x1.
    read = model.blocks[0].attn
    wiring = {"x1": [("dx", -1), ("dq", -1)], "y1": [("y1", 11), ("dy", -11)],
              "x2": [("dx", 1)], "y2": [("dy", 11)], "x": [("dq", 1)]}
    for h, r in enumerate(READS):
        two_digit = r.startswith("y")
        read.W_q.weight[2 * h, C["live"]] = 1
        read.W_k.weight[2 * h, C["a_cos"]] = math.cos(angle[r])
        read.W_k.weight[2 * h, C["a_sin"]] = math.sin(angle[r])
        read.W_q.weight[2 * h + 1, C["phase"]] = 1
        read.W_k.weight[2 * h + 1, C["live"]] = L + math.log(99 if two_digit else 9) + 110.0
        read.W_k.weight[2 * h + 1, C["stay"]] = -0.1
        read.W_v.weight[h, C["dig"]] = 1
        for reg, gain in wiring[r]:
            read.W_o.weight[C[reg], h] = gain

    # FIT: head k scores positions by e_k(t) * -beta e_k(j) + stay(j). A wrong
    # slope has e_k^2 >= 1 (>= 0.01 at 1/10 volume), which outweighs `stay`.
    fit = model.blocks[1].attn
    beta = 40000.0
    for h, k in enumerate(SLOPES):
        for w, c in [(1, "dy"), (-k, "dx")]:
            fit.W_q.weight[2 * h, C[c]] = w
            fit.W_k.weight[2 * h, C[c]] = -beta * w
        fit.W_q.weight[2 * h + 1, C["live"]] = 1
        fit.W_k.weight[2 * h + 1, C["stay"]] = 1
        for w, c in [(1, "y1"), (k, "dq"), (-10, "dig")]:
            fit.W_v.weight[h, C[c]] = w
        fit.W_o.weight[C["out"], h] = 1

    # Head: logit_d = 2d*out - d^2 rounds `out`; punctuation is never printed.
    for tok, i in task.stoi.items():
        if tok.isdigit():
            d = int(tok)
            model.head.weight[i, C["out"]] = 2 * d
            model.head.weight[i, C["live"]] = -d * d
        else:
            model.head.weight[i, C["live"]] = -1e3
    diagnostics(model, task)


def diagnostics(model: SimpleTransformer, task) -> None:
    """Sanity check only (not used by the weights): a larger non-eval-seed set."""
    acc, _ = evaluate(model, task, n_samples=5000, seed=7, device="cpu", verbose=False)
    print(f"diagnostic accuracy on 5000 seed-7 examples: {acc:.4f}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Compass10SharpnessL15"
model_description = "Compass09 with attention sharpness L lowered from 40 to 15, measuring how soft READ can be."


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
