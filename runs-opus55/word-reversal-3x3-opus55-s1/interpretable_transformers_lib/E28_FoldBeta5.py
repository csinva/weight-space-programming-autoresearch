"""
Hand-written transformer for word-reversal-3x3: "abc_def_ghi=" -> "ghi_def_abc".

Pointer model: attention is the only computation. Each prompt slot's key is its
1-based position, each answer position's query is the slot it should read, the
score is negative squared distance, and the value is the token as one number.
The head then picks the token whose number is nearest to what was read.
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
model_shorthand_name = "E28_FoldBeta5"
model_description = "E27 at beta=5, the lowest value that passed on random prompts, checked on the worst-case a/z stress test."
CFG = dict(query="fold", values="ids", beta=5.0)
COEFFS = dict(linear=[9.0, -4.0, 1.0, -4.0], fold=[9.0, 1.0, -8.0])


def source_of(p: int) -> int:
    """Prompt index to read when the token at position p predicts the next char."""
    word, letter = divmod(p - 11, 4)
    return 4 * (2 - word) + letter if letter < 3 else 7 - 4 * word


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """One causal attention read with computed keys and tabled queries."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.beta = nn.Parameter(torch.zeros(1))
        if CFG["values"] == "table":
            self.token_emb = nn.Embedding(vocab_size, 1)
        self.reads = nn.Parameter(torch.zeros(11 if CFG["query"] == "table" else len(COEFFS[CFG["query"]])))

    def queries(self, pos: torch.Tensor) -> torch.Tensor:
        if CFG["query"] == "table":
            return torch.cat([torch.zeros(11), self.reads])[pos]
        i = (pos - 11).clamp(min=0)
        word, letter = i.div(4, rounding_mode="floor"), i % 4
        if CFG["query"] == "linear":
            features = torch.stack([torch.ones_like(i), word, letter, (letter == 3).long()], dim=1)
        else:
            features = torch.stack([torch.ones_like(i), i, word], dim=1)
        return features.float() @ self.reads

    def values(self, ids: torch.Tensor) -> torch.Tensor:
        if CFG["values"] == "table":
            return self.token_emb(ids)
        return (ids.clamp(max=26) if CFG["query"] == "fold" else ids).float()[..., None]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        scores = -(self.queries(pos)[:, None] - (pos + 1.0)[None, :]).pow(2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        attn = (self.beta * scores).masked_fill(mask, float("-inf")).softmax(dim=-1)
        read = attn @ self.values(ids)
        table = self.values(torch.arange(self.vocab_size, device=ids.device)[None])
        return -(read - table.transpose(-2, -1)).pow(2)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def stress_test(model: SimpleTransformer, task, n: int = 2000) -> None:
    """Report accuracy on prompts built from only 'a' and 'z', the worst case
    for repeated letters and for leakage. Validation only; nothing is fitted."""
    gen = torch.Generator().manual_seed(1)
    letters = torch.tensor([task.stoi["a"], task.stoi["z"]])[torch.randint(2, (n, 9), generator=gen)]
    sep, eq = torch.full((n, 1), task.stoi["_"]), torch.full((n, 1), task.stoi["="])
    ids = torch.cat([letters[:, :3], sep, letters[:, 3:6], sep, letters[:, 6:], eq], dim=1)
    target = torch.cat([ids[:, 8:11], sep, ids[:, 4:7], sep, ids[:, :3]], dim=1)
    for _ in range(11):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    print(f"stress_az:     {(ids[:, 12:] == target).all(dim=1).float().mean():.4f}")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    model.beta.fill_(CFG["beta"])
    if CFG["values"] == "table":
        model.token_emb.weight.copy_(torch.arange(28.0)[:, None])
    table = [source_of(p) + 1.0 for p in range(11, 22)]
    model.reads.copy_(torch.tensor(table if CFG["query"] == "table" else COEFFS[CFG["query"]]))
    stress_test(model, task)


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
