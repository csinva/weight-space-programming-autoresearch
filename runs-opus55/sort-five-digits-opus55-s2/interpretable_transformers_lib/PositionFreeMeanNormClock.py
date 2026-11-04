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
    """Uniform-score attention keeping only the value projection: x + the
    causal mean of W_v x. No LayerNorm and no MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, x.shape[1] + 1, device=x.device)
        return x + self.W_v(x).cumsum(dim=1) / steps[None, :, None]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 12,
        n_layers: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model) for _ in range(n_layers)])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(ids)  # no position embedding at all
        for block in self.blocks:
            h = block(h)
        return h[:, :, :self.vocab_size]  # first 11 dims are the logits


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
    """Dims 0-9 score the digits. Dim 10 scores '=' and always holds -2, so it
    doubles as a bias. Dim 11 flags the '=' token."""
    for p in model.parameters():
        p.zero_()
    bias, eq_flag = 10, 11
    W_v = model.blocks[0].W_v.weight
    d = torch.arange(10.0)

    # Residual curve of digit a: 2 min(d, a), its hinge plus a slope of 2.
    # '=' has no residual curve: nothing is emitted before it.
    for a in range(10):
        model.token_emb.weight[a, :10] = 2 * torch.minimum(d, torch.tensor(float(a)))
    model.token_emb.weight[:, bias] = -2.0
    model.token_emb.weight[task.stoi["="], eq_flag] = 1.0

    # Attended curve adds 2d to every token (read from the bias) and a further
    # -21d to '='. Totals: slope 4 per digit, -19 for '='. Multiplying the
    # output by T = 5 + c turns the fixed slopes into 2c + T - 1/2 (doubled),
    # so the mean's own normaliser plays the role of the step counter.
    W_v[:10, :10] = torch.eye(10)
    W_v[:10, bias] = -d
    W_v[:10, eq_flag] = -21.0 * d

    exhaustive_check(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "PositionFreeMeanNormClock"
model_description = "No position embedding. The causal mean's 1/T normaliser acts as the step counter: fixed token slopes become step-dependent once the output is rescaled by T = 5 + c. Only token_emb and W_v are parameters."


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
