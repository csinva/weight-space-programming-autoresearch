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

class Block(nn.Module):
    """Attention with uniform scores and identity projections: a causal mean
    over all tokens so far. No LayerNorm, no MLP, no parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, x.shape[1] + 1, device=x.device)
        return x + x.cumsum(dim=1) / steps[None, :, None]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 11,
        n_layers: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_slope = nn.Embedding(max_seq_len, 1)  # one scalar per position
        self.slope_dir = nn.Parameter(torch.zeros(d_model))  # shared direction
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_slope(pos)[None, :, :] * self.slope_dir
        for block in self.blocks:
            h = block(h)
        return h  # d_model == vocab_size: the hidden state is the logit vector


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
    """Every vector lives in logit space: dims 0-9 score the ten digits and
    dim 10 scores '='. Dim 10 doubles as the prompt/answer flag."""
    for p in model.parameters():
        p.zero_()
    role = 10

    # Each digit a embeds as its own loss curve: score(d) = -2 max(d - a, 0).
    # '=' behaves like one more sample below the answer, same curve as 0.
    d = torch.arange(10.0)
    for a in range(10):
        model.token_emb.weight[a, :10] = -2 * F.relu(d - a)
    model.token_emb.weight[task.stoi["="], :10] = -2 * d

    # Attention is uniform (W_q = W_k = 0): the mean of all T tokens seen. In
    # units of 1/T the output has one sample per token, plus T copies of the
    # previous token from the residual. Below the answer that is 2c - 1 + T
    # samples and at the answer 2c + T, so the slope sits halfway between.
    # The mean also averages earlier position embeddings, so solve for each
    # p_t such that p_t + (p_5 + ... + p_t) / T hits the wanted total slope.
    model.slope_dir[:10] = d
    model.slope_dir[role] = -1.0  # every answer step suppresses '='
    earlier = 0.0
    for t in range(5, task.seq_len):
        c, T = t - 4, t + 1
        p = (2 * (2 * c + T - 0.5) / T - earlier / T) / (1 + 1 / T)
        earlier += p
        model.pos_slope.weight[t, 0] = p
    print("slopes:", [round(s, 4) for s in model.pos_slope.weight[5:11, 0].tolist()])

    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CausalMeanRankOnePos"
model_description = "CausalMeanNoHead with position factored into one scalar slope per step times a shared direction, replacing the 16x11 position table."


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
