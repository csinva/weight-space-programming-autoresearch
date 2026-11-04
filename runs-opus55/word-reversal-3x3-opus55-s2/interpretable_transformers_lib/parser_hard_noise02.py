"""Interpretable transformer, hand-written weights (no training) for word-reversal-3x3:
"abc_def_ghi=" -> "ghi_def_abc". The model parses words by counting separators.
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

class SimpleTransformer(nn.Module):
    """Two steps: a prefix-sum attention counts separators, then a hard distance
    attention copies from the (word, letter) address computed from that count."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, 2)  # (value, is_separator)
        self.pos_emb = nn.Embedding(max_seq_len - 1, 1)  # the time index t
        self.W_q = nn.Parameter(torch.zeros(2, 3))  # (t, cum, 1) -> (word, letter) wanted
        self.W_k = nn.Parameter(torch.zeros(2, 3))  # (t, cum, sep) -> (word, letter) held

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        emb = self.token_emb(ids)
        value, sep = emb[:, :, 0], emb[:, :, 1]  # (B, T) each
        t = self.pos_emb(torch.arange(T, device=ids.device))[:, 0].expand(B, T)
        causal = torch.tril(torch.ones(T, T, device=ids.device))
        cum = sep @ causal.T  # prefix sum: attention with weight 1 on every past position
        q = torch.stack([t, cum, torch.ones_like(t)], -1) @ self.W_q.T  # (B, T, 2)
        k = torch.stack([t, cum, sep], -1) @ self.W_k.T  # (B, T, 2)
        scores = -((q[:, :, None, :] - k[:, None, :, :]) ** 2).sum(-1)
        best = scores.masked_fill(causal == 0, float("-inf")).argmax(dim=-1)  # hard attention
        attn = F.one_hot(best, T).float()
        copied = attn @ value[:, :, None]  # (B, T, 1)
        # Tied head: '=' shares '_' value, and ties go to the lower index, '_'.
        return -(copied - self.token_emb.weight[:, 0]) ** 2


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """'_' and '=' are both separators. With cum separators seen up to time t:
    a position holds word = cum - sep and letter = t - 4*word.
    In the answer, the character being predicted is in output word cum-3, so it wants
    prompt word 5-cum and letter t+1-4*cum. Answer positions hold word >= 3, which no
    query asks for. Output slot 3 wants (2, 3), the '=', which reads out as '_'."""
    SPACE = 10.0
    #                                       t     cum    1
    model.W_q.copy_(SPACE * torch.tensor([[0.0, -1.0, 5.0],    # word wanted
                                          [1.0, -4.0, 1.0]]))  # letter wanted
    #                                       t     cum   sep
    model.W_k.copy_(SPACE * torch.tensor([[0.0, 1.0, -1.0],    # word held
                                          [1.0, -4.0, 4.0]]))  # letter held
    sep_id = task.stoi["_"]
    for i in range(task.vocab_size):
        is_sep = task.itos[i] in "_="
        model.token_emb.weight[i, 0] = float(sep_id if is_sep else i)
        model.token_emb.weight[i, 1] = float(is_sep)
    for t in range(model.pos_emb.num_embeddings):
        model.pos_emb.weight[t, 0] = float(t)

    NOISE = 0.02  # robustness probe: seeded multiplicative noise on every parameter
    torch.manual_seed(0)
    for p in model.parameters():
        p.mul_(1.0 + NOISE * torch.randn_like(p))
    model.token_emb.weight[task.stoi["="], 0] = model.token_emb.weight[sep_id, 0]  # re-tie


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "parser_hard_noise02"
model_description = "Diagnosis of parser_noise02_retied: hard attention with the same 2% noise. Blending is impossible, so a failure means the address arithmetic itself drifted."


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
