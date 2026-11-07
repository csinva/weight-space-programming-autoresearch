"""
Interpretable transformer for parity of up to 10 bits (weights hand-written, never trained).

Prompt "____101011=" (left-padded with '_') -> answer "0"/"1", the XOR of the bits.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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

# Architecture configuration for this attempt.
D_MODEL, N_HEADS = 1, 1
D_FFS = [3, 2]             # per layer: number of ReLU units
HAS_ATTN = [True, False]   # per layer: does this block have attention?
HAS_MLP = [True, True]     # per layer: does this block have an MLP?
REPEATS = [1, 1]           # per layer: how many times this one block is applied (tied weights)
USE_LN = False
USE_FINAL_LN = False
USE_RELU = True
USE_POS = False
UNIFORM_ATTN = True        # attention with no learned matrices: a fixed uniform causal average


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


class UniformAttention(nn.Module):
    """Causal attention with all scores equal, so each position averages itself and its past."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, x.shape[1] + 1, device=x.device)
        return x.cumsum(dim=1) / steps[None, :, None]


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        return self.fc2(F.relu(h) if USE_RELU else h)


def norm(d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model) if USE_LN else nn.Identity()


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, has_attn: bool, has_mlp: bool):
        super().__init__()
        self.ln1 = norm(d_model)
        attn_cls = UniformAttention() if UNIFORM_ATTN else CausalSelfAttention(d_model, n_heads)
        self.attn = attn_cls if has_attn else None
        self.ln2 = norm(d_model)
        self.mlp = MLP(d_model, d_ff) if has_mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL) if USE_POS else None
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, f, a, m) for f, a, m in zip(D_FFS, HAS_ATTN, HAS_MLP)]
        )
        self.final_ln = norm(D_MODEL) if USE_FINAL_LN else nn.Identity()
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        h = self.token_emb(ids)
        if self.pos_emb is not None:
            h = h + self.pos_emb(torch.arange(T, device=ids.device))[None, :, :]
        for block, repeats in zip(self.blocks, REPEATS):
            for _ in range(repeats):
                h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def exhaustive_check(model: SimpleTransformer, task) -> None:
    """Print accuracy over ALL 2047 possible prompts (read-only sanity check, no fitting)."""
    prompts, targets = [], []
    for n in range(11):
        for bits in itertools.product("01", repeat=n):
            prompts.append("_" * (10 - n) + "".join(bits) + "=")
            targets.append(bits.count("1") % 2)
    ids = torch.tensor([task.encode(p) for p in prompts])
    with torch.no_grad():
        pred = model(ids)[:, -1].argmax(-1)
    print(f"EXHAUSTIVE {(pred == torch.tensor(targets)).sum().item()}/{len(targets)}")


def write_weights(model: SimpleTransformer, task) -> None:
    """Fold budget split as (3, 2) units: 4 * 3 = 12 linear pieces, at least the 10 needed."""
    def set_zigzag(mlp, knots, shift=0.0):
        """Program x -> x + sum_k (-1)^(k+1) 2 relu(x - knots[k]): reflect at each knot in turn."""
        for u, a in enumerate(knots):
            mlp.fc1.weight[u, 0] = 1.0
            mlp.fc1.bias[u] = -a
            mlp.fc2.weight[0, u] = -2.0 * (-1.0) ** u
        mlp.fc2.bias[0] = shift

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.token_emb.weight[task.stoi["1"], 0] = 11.0   # uniform attention over 11 tokens: x = count
        set_zigzag(model.blocks[0].mlp, [3.0, 6.0, 9.0])   # [0, 10] -> [0, 3], reflecting at integers
        set_zigzag(model.blocks[1].mlp, [1.0, 2.0], -0.5)  # 0,1,2,3 -> 0,1,0,1, centered on +-1/2
        model.head.weight[task.stoi["1"], 0] = 1.0
        model.head.weight[task.stoi["0"], 0] = -1.0
    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "origami_2L_5units"
model_description = "Two blocks of 3 and 2 ReLU units: a zigzag folds [0,10] onto [0,3], then a 2-kink zigzag finishes; (3+1)(2+1)=12 pieces, one more unit than the 3-layer version."


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
