"""
Interpretable transformer for sort-five-digits: "29270=" -> "02279".
Weights are written in closed form; nothing is trained.

Circuit: the c-th output is the smallest digit d with count(inputs <= d) >= c.
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

@torch.no_grad()
def exhaustive_check(model, task) -> None:
    """Verification only: autoregressive accuracy over all 100000 prompts."""
    prompts = torch.tensor([[int(c) for c in f"{i:05d}"] + [task.stoi["="]] for i in range(100000)])
    ids = prompts
    margin = float("inf")
    for _ in range(task.answer_len):
        logits = model(ids)[:, -1]
        top2 = logits.topk(2, dim=-1).values
        margin = min(margin, (top2[:, 0] - top2[:, 1]).min().item())
        ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], 1)
    ok = (ids[:, task.prompt_len:] == prompts[:, :5].sort(dim=1).values).all(1)
    print(f"EXHAUSTIVE: {ok.sum().item()}/100000  min_margin={margin:.4f}")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Stock architecture. Dims 0-9 score digits, dim 10 scores '=' and flags
    the answer region, dims 11-13 are slack that neutralises LayerNorm.
    Blocks 2-3 and all MLPs stay zero, so they pass x through unchanged."""
    for p in model.parameters():
        p.zero_()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.fill_(1.0)
    role, mean_slack, plus, minus = 10, 11, 12, 13
    D = model.d_model
    M = 1000.0
    attn = model.blocks[0].attn
    d = torch.arange(10.0)

    # Hinge curves, plus slack giving every digit zero mean and equal norm.
    curves = torch.stack([-2 * F.relu(d - a) for a in range(10)])
    means = curves.sum(1)
    norms = (curves ** 2).sum(1) + means ** 2
    slack = ((norms.max() - norms) / 2).sqrt()
    emb = model.token_emb.weight
    emb[:10, :10] = curves
    emb[:10, mean_slack] = -means
    emb[:10, plus] = slack
    emb[:10, minus] = -slack
    emb[task.stoi["="], :10] = -2 * d

    # LayerNorm now multiplies every prompt digit by the same factor g.
    g = math.sqrt(D / norms.max().item())

    # Slope 2c + 1, and a large negative flag on every answer position.
    for t in range(5, task.seq_len):
        c = t - 4
        model.pos_emb.weight[t, :10] = (2 * c + 1) * d
        model.pos_emb.weight[t, role] = -M

    # Prompt tokens have flag 0 after LayerNorm; answer tokens are near -sqrt(D).
    attn.W_q.weight[role, role] = 1.0
    attn.W_k.weight[role, role] = -3.0
    attn.W_v.weight[:10, :10] = torch.eye(10)
    attn.W_o.weight[:10, :10] = 5.0 / g * torch.eye(10)

    model.head.weight[:11, :11] = torch.eye(11)

    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "StockArchPinball"
model_description = "Unmodified 64-dim, 4-head, 3-layer architecture with LayerNorm. Slack dims give each digit zero mean and equal norm so LayerNorm is a constant rescale undone by W_o; one head does the work and everything else is zero."


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
