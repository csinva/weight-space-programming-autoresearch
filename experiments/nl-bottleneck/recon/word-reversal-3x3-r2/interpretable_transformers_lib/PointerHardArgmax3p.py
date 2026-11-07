"""
Interpretable transformer for character-level sequence tasks.

Task here: word-reversal-3x3. "abc_def_ghi=" -> "ghi_def_abc".

Model: no MLP. One causal attention head with hard argmax acts as a pointer
into the prompt; the value is the token id as a scalar (capped at the "_" id),
and the readout is a nearest-value lookup over the vocabulary. 3 parameters,
all written in closed form (no training).
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

class PointerAttention(nn.Module):
    """One causal head, hard argmax, score = -(query - key)^2.

    key_j   = j + 1                                  (1-based position)
    query_t = q_offset + i - q_stride * floor(i/4),  i = t - 11
              (t is the 0-based position; "=" sits at t = 11)
    """

    def __init__(self):
        super().__init__()
        self.q_offset = nn.Parameter(torch.tensor(0.0))
        self.q_stride = nn.Parameter(torch.tensor(0.0))

    def forward(self, ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        t = torch.arange(T, device=ids.device, dtype=torch.float32)
        i = t - 11.0
        word = torch.floor(i / 4.0)
        q = self.q_offset + i - self.q_stride * word             # (T,)
        k = t + 1.0                                              # (T,)
        scores = -(q[:, None] - k[None, :]) ** 2                 # (T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        idx = scores.argmax(dim=-1)                              # hard argmax: single best earlier position
        return values[:, idx]                                    # (B, T)


class SimpleTransformer(nn.Module):
    """No-MLP pointer transformer: 1 hard-attention head + nearest-value readout. 3 params."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.attn = PointerAttention()
        self.v_cap = nn.Parameter(torch.tensor(0.0))             # value = min(token_id, v_cap)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        values = torch.minimum(ids.float(), self.v_cap)          # scalar value per token
        read = self.attn(ids, values)                            # (B, T)
        vocab_ids = torch.arange(self.vocab_size, device=ids.device, dtype=torch.float32)
        logits = -(read[..., None] - vocab_ids) ** 2             # exact match = 0; ties -> lower id
        return logits


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights. No training."""
    with torch.no_grad():
        model.attn.q_offset.fill_(9.0)                   # first answer step reads 1-based key 9 (index 8)
        model.attn.q_stride.fill_(8.0)                   # each new word jumps back 8 positions
        model.v_cap.fill_(float(task.stoi["_"]))         # cap at separator id so "=" reads as "_"


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "PointerHardArgmax3p"
model_description = ("No-MLP: one hard-argmax head, key=1-based pos, query=9+i-8*floor(i/4), "
                     "score=-(q-k)^2; value=min(id,'_'), logits=-(v-id)^2. 3 params.")


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
