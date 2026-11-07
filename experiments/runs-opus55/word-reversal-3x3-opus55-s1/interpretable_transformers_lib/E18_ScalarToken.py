"""
Hand-written transformer for word-reversal-3x3: "abc_def_ghi=" -> "ghi_def_abc".

Field model: the embedded sequence is a token field t and an address field a.
One causal attention layer reads the prompt: an answer position stores the
negated address of the character it wants, so query = -a, key = a, value = t.
Output logits compare the attended value with the token table (tied head).
No training anywhere: all weights are written in closed form.
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

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]

# A unique shorthand name + 1-2 sentence description of what this attempt does.
# token/address: "ring" = 2-D dot-product code, "scalar" = 1-D nearest-neighbour code.
model_shorthand_name = "E18_ScalarToken"
model_description = "E17 with each token as one number (its index); the head picks the nearest token value."
CFG = dict(token="scalar", address="ring", beta=500.0)


def source_of(p: int) -> int:
    """Prompt index to read when the token at position p predicts the next char."""
    word, letter = divmod(p - 11, 4)
    return 4 * (2 - word) + letter if letter < 3 else 7 - 4 * word


def token_table() -> torch.Tensor:
    if CFG["token"] == "scalar":
        return torch.arange(28.0)[:, None]
    angle = 2 * math.pi * torch.arange(28) / 28
    return torch.stack([angle.cos(), angle.sin()], dim=1)


def address_of(s: int) -> torch.Tensor:
    """Code of prompt slot s. Slots are numbered from 1 so that -a never equals a."""
    if CFG["address"] == "scalar":
        return torch.tensor([s + 1.0])
    return torch.tensor([math.cos(math.pi * s / 11), math.sin(math.pi * s / 11)])


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """Token and address fields, one causal attention read, tied output head."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, token_table().shape[1])
        self.pos_emb = nn.Embedding(max_seq_len, address_of(0).shape[0])
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        a = self.pos_emb(torch.arange(T, device=ids.device))
        if CFG["address"] == "scalar":
            scores = -(-a[:, None] - a[None, :]).pow(2).sum(-1)  # nearest key to the query
        else:
            scores = -a @ a.T                                    # dot product of query, key
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        attn = (self.beta * scores).masked_fill(mask, float("-inf")).softmax(dim=-1)
        read = attn @ self.token_emb(ids)
        table = self.token_emb.weight
        if CFG["token"] == "scalar":
            return -(read - table.T).pow(2)  # nearest token value
        return read @ table.T


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    model.token_emb.weight.copy_(token_table())
    model.beta.fill_(CFG["beta"])
    model.pos_emb.weight.zero_()
    for p in range(22):
        model.pos_emb.weight[p] = address_of(p) if p < 11 else -address_of(source_of(p))


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
