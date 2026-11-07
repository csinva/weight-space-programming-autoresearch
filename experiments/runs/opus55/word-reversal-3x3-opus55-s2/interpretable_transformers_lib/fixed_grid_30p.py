"""Interpretable transformer, hand-written weights (no training) for word-reversal-3x3:
"abc_def_ghi=" -> "ghi_def_abc". The algorithm is a reflection through the '=' pivot.
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

class MirrorAttention(nn.Module):
    """Causal distance attention. Addresses steer it; token values pass through."""

    def __init__(self):
        super().__init__()
        self.q_scale = nn.Parameter(torch.zeros(2))  # diagonal of W_q

    def forward(self, addr: torch.Tensor) -> torch.Tensor:
        """addr is (T, 2). Returns the (T, T) attention matrix."""
        T = addr.shape[0]
        q = addr * self.q_scale
        scores = -((q[:, None, :] - addr[None, :, :]) ** 2).sum(-1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=addr.device), diagonal=1)
        best = scores.masked_fill(mask, float("-inf")).argmax(dim=-1)  # hard attention
        return F.one_hot(best, T).float()


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, 1)  # one number per token
        self.attn = MirrorAttention()
        self.prompt_chars = 11  # length of "www_www_www"; the '=' is the pivot

    def addresses(self, T: int, device) -> torch.Tensor:
        """Fixed positional encoding, computed not stored: (word, letter column)."""
        t = torch.arange(T, device=device)
        answer = t >= self.prompt_chars
        char = torch.where(answer, t - self.prompt_chars, t)  # index within its own text
        word = char // 4 + torch.where(answer, 1, -3)
        letter = char % 4
        letter = torch.where(letter == 3, 4, letter)  # separators go in a far column
        return torch.stack([word, letter], -1).float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        tok = self.token_emb(ids)[:, :, 0]  # (B, T)
        addr = self.addresses(T, ids.device)  # (T, 2)
        copied = (tok @ self.attn(addr).T)[:, :, None]  # (B, T, 1)
        # Tied head: the logit of token i is minus the distance from the copy to its embedding.
        return -(copied - self.token_emb.weight[:, 0]) ** 2


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Word timeline centred on the '=' pivot: prompt -3,-2,-1 and answer +1,+2,+3.
    The two attention parameters say: word -> -word, letter -> letter.
    Separators sit at letter column 4, so the missing (-1, sep) lands on (-2, sep).
    The addresses are a fixed formula, so only token values and the reflection are stored."""
    model.attn.q_scale.copy_(torch.tensor([-1.0, 1.0]))
    for i in range(task.vocab_size):
        model.token_emb.weight[i, 0] = float(i)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "fixed_grid_30p"
model_description = "Position table replaced by a fixed computed (word, letter) encoding, like a sinusoidal PE. Stored parameters: 28 token values plus the 2-number reflection. Hard attention."


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
