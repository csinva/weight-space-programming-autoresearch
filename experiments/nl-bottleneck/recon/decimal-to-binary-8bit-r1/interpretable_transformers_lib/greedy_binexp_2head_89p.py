"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 1-layer, 2-head, attention-only transformer for
`decimal-to-binary-8bit`: exact greedy binary expansion, MSB first.

Usage:
    uv run interpretable_transformer.py --task decimal-to-binary-8bit
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
    """1-layer, 2-head attention-only transformer (no MLP, no LayerNorm).

    Parameters (89 total):
      token_emb : (V=11, 2)  per-token [value, is_eq]            -> 22
      pos_emb   : (P=16, 4)  per-position [L, B0, B1, Qeq]       -> 64
      readout   : (3,)       [w_head0, w_head1, bias]            -> 3
    Head h key score at position p from query q:
        head 0: L[p] + B0[p]
        head 1: L[p] + B1[p] + is_eq[p] * Qeq[q]
    Both heads use the scalar token value as V (causal softmax).
    "1" logit = w0*h0 + w1*h1 + b ; "0" logit = its negative ; others 0.
    """

    def __init__(self, vocab_size: int, max_seq_len: int = 16, **kw):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, 2)
        self.pos_emb = nn.Embedding(max_seq_len, 4)
        self.readout = nn.Parameter(torch.zeros(3))
        self.id0 = 0
        self.id1 = 1

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        te = self.token_emb(ids)                                  # (B,T,2)
        pe = self.pos_emb(torch.arange(T, device=ids.device))     # (T,4)
        val, is_eq = te[..., 0], te[..., 1]
        L, B0, B1, Qeq = pe[:, 0], pe[:, 1], pe[:, 2], pe[:, 3]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        s0 = (L + B0)[None, None, :].expand(B, T, T)
        s1 = (L + B1)[None, None, :] + Qeq[None, :, None] * is_eq[:, None, :]
        s0 = s0.masked_fill(mask, float("-inf"))
        s1 = s1.masked_fill(mask, float("-inf"))
        h0 = (s0.softmax(-1) * val[:, None, :]).sum(-1)           # ~ n/111
        h1 = (s1.softmax(-1) * val[:, None, :]).sum(-1)           # ~ (S+2^k)/256
        w0, w1, b = self.readout
        z = w0 * h0 + w1 * h1 + b                                 # ~ 1 + 2(r - 2^k)
        logits = torch.zeros(B, T, self.vocab_size, device=ids.device)
        logits[..., self.id1] = z
        logits[..., self.id0] = -z
        return logits


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    ln2 = math.log(2.0)
    V = task.vocab_size
    stoi = task.stoi
    model.id0, model.id1 = stoi["0"], stoi["1"]
    eq_pos = 3          # prompt "DDD=" -> '=' at position 3, bits fed back at 4..10
    with torch.no_grad():
        te = torch.zeros(V, 2)
        for d in range(10):
            te[stoi[str(d)], 0] = float(d)       # digit face value (0/1 for bits)
        te[stoi["="], 0] = 0.5
        te[stoi["="], 1] = 1.0
        model.token_emb.weight.copy_(te)

        P = model.max_seq_len
        pe = torch.zeros(P, 4)
        # log-place-value feature
        pe[0, 0] = math.log(100.0)
        pe[1, 0] = math.log(10.0)
        pe[2, 0] = 0.0
        for p in range(eq_pos, min(P, 12)):
            pe[p, 0] = (11 - p) * ln2            # '=' -> 8 ln2, bit at p -> 2^(11-p)
        # +-6 positional region biases
        for p in range(P):
            digit = p < eq_pos
            pe[p, 1] = 6.0 if digit else -6.0    # head 0 favours digits
            pe[p, 2] = -6.0 if digit else 6.0    # head 1 favours '=' and bits
        # query-side shift so '=' weight is 2^(k+1) = 2^(11-q) at query q
        for q in range(eq_pos, min(P, 12)):
            pe[q, 3] = (eq_pos - q) * ln2
        model.pos_emb.weight.copy_(pe)

        model.readout.copy_(torch.tensor([222.0, -512.0, 1.0]))


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "greedy_binexp_2head_89p"
model_description = ("1-layer 2-head attention-only (89 params): head0 log-place weights 100:10:1 -> n/111, "
                     "head1 bit weights 2^j + '=' 2^(k+1) -> (S+2^k)/256; logit1 = 222h0-512h1+1, "
                     "logit0 = -logit1; +-6 region bias.")


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
