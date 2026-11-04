"""Interpretable transformer, hand-written weights (no training) for word-reversal-3x3:
"abc_def_ghi=" -> "ghi_def_abc". Answer slot k simply copies prompt position SOURCE[k].
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
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = x.view(B, T, H, dh).transpose(1, 2)  # keys are the raw embeddings
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = -((q[:, :, :, None, :] - k[:, :, None, :, :]) ** 2).sum(-1)  # distance attention
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).contiguous().view(B, T, D)


class Block(nn.Module):
    """Attention only: no LayerNorm, no MLP."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 1, n_heads: int = 1, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len - 1, d_model)  # last token is never fed back
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        # Tied head: the logit of token i is minus the distance from h to G * its embedding.
        G = self.blocks[0].attn.W_v.weight[0, 0]
        return -(h - G * self.token_emb.weight[:, 0]) ** 2


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

SOURCE = [8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2]  # answer slot k copies prompt position SOURCE[k]


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """d_model = 1 with distance attention. Prompt position s is the number SPACE*(1+s),
    token i is the nudge EPS*i. Answer position k holds -G*SPACE*(1+SOURCE[k]), so query
    -x_t/G lands on the source's number, and x_t + G*x_source cancels position, leaving
    EPS*(c_current + G*c_copied). With G=100 the current token is under half a band, so
    the tied head (nearest G*embedding) reads c_copied. 52 parameters in total."""
    for p in model.parameters():
        p.zero_()
    V = task.vocab_size
    G, EPS, SPACE = 100.0, 1e-2, 10.0
    a = model.blocks[0].attn
    a.W_q.weight[0, 0] = -1.0 / G
    a.W_v.weight[0, 0] = G
    for i in range(V):
        model.token_emb.weight[i, 0] = EPS * i
    for s in range(len(SOURCE)):
        model.pos_emb.weight[s, 0] = SPACE * (1.0 + s)
    for k, s in enumerate(SOURCE):
        model.pos_emb.weight[task.prompt_len - 1 + k, 0] = -G * SPACE * (1.0 + s)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "tied_head_d1"
model_description = "d=1 distance attention with a parameter-free tied head: logit_i = -(h - G*tok_emb_i)^2 and G=100 so the residual current token never crosses a band. 52 parameters."


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
