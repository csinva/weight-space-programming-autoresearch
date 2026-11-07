"""
Hand-written transformer for word-reversal-3x3: "abc_def_ghi=" -> "ghi_def_abc".

Every answer character is a copy of one prompt character, so the circuit is a
single attention layer acting as a pointer: the position decides WHERE to read
(query), prompt positions are addresses (keys), and values carry the token.
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
# Experiment knobs: dt = token dims, dp = position dims, d_model = dt + dp.
model_shorthand_name = "E16_RingNegated"
model_description = "One 2-D address field: prompt positions store r(s) on a half ring, answer positions store -r(source), and W_q=-I turns that into a query. d_model 4."
CFG = dict(token="ring", position="ring_negated", beta=500.0)


def source_of(p: int) -> int:
    """Prompt index to read when the token at position p predicts the next char."""
    word, letter = divmod(p - 11, 4)
    return 4 * (2 - word) + letter if letter < 3 else 7 - 4 * word


def token_codes() -> torch.Tensor:
    """Rows are token vectors; the head decodes by dot product with these rows."""
    if CFG["token"] == "onehot":
        return torch.eye(28)
    if CFG["token"] == "binary":
        bits = torch.tensor([[(c >> b) & 1 for b in range(5)] for c in range(28)], dtype=torch.float32)
        return (2 * bits - 1) / math.sqrt(5)
    angle = 2 * math.pi * torch.arange(28) / 28  # "ring"
    return torch.stack([angle.cos(), angle.sin()], dim=1)


def position_codes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (P, M_k, M_q): position table, key map and query map over position dims."""
    if CFG["position"] == "onehot":
        M_q = torch.zeros(23, 23)
        for p in range(11, 22):
            M_q[source_of(p), p] = 1
        return torch.eye(23), torch.eye(23), M_q
    if CFG["position"] == "wordletter":
        # Dims: word(3) | letter-slot(3 letters + separator) | phase(+1 prompt, -1 answer).
        P = torch.zeros(23, 8)
        for p in range(22):
            prompt = p < 11
            word, letter = divmod(p if prompt else p - 11, 4)
            P[p, 3 + letter] = 1
            P[p, 7] = 1 if prompt else -1
            if letter < 3 or not CFG.get("sep_any_word"):
                P[p, word] = 1
        M_q = torch.eye(8)
        M_q[:3] = M_q[[2, 1, 0]]  # read the mirrored word, same letter slot
        M_q[7, 7] = -1            # answer positions query the prompt phase
        return P, torch.eye(8), M_q
    ring = lambda s, arc: torch.tensor([math.cos(arc * s / 11), math.sin(arc * s / 11)])
    if CFG["position"] == "ring_two_fields":
        # Keys on one ring, queries on a second; W_q moves field 2 onto field 1.
        P = torch.zeros(23, 4)
        for p in range(22):
            if p < 11:
                P[p, :2] = ring(p, 2 * math.pi)
            else:
                P[p, 2:] = ring(source_of(p), 2 * math.pi)
        M_k, M_q = torch.zeros(4, 4), torch.zeros(4, 4)
        M_k[:2, :2] = torch.eye(2)
        M_q[:2, 2:] = torch.eye(2)
        return P, M_k, M_q
    if CFG["position"] == "ring_negated":
        # One 2-D field. Prompt position s is the point r(s) on a half ring; an
        # answer position stores -r(source), so W_q = -I turns it into a query
        # while its own key points away from everything it could read.
        P = torch.zeros(23, 2)
        for p in range(22):
            P[p] = ring(p, math.pi) if p < 11 else -ring(source_of(p), math.pi)
        return P, torch.eye(2), -torch.eye(2)
    raise ValueError(CFG["position"])


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape[1:]
        scores = (self.W_q(x) @ self.W_k(x).transpose(-2, -1)) / math.sqrt(D)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ self.W_v(x)


class SimpleTransformer(nn.Module):
    """One causal attention layer between embeddings and the output head."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model = token_codes().shape[1] + position_codes()[0].shape[1]
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn = CausalSelfAttention(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        return self.head(self.attn(self.token_emb(ids) + self.pos_emb(pos)[None, :, :]))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    for param in model.parameters():
        param.zero_()
    TOK = token_codes()
    P, M_k, M_q = position_codes()
    dt, D = TOK.shape[1], model.d_model
    model.token_emb.weight[:, :dt] = TOK
    model.pos_emb.weight[:, dt:] = P
    model.head.weight[:, :dt] = TOK
    model.attn.W_k.weight[dt:, dt:] = M_k
    model.attn.W_q.weight[dt:, dt:] = CFG["beta"] * math.sqrt(D) * M_q
    model.attn.W_v.weight[:dt, :dt] = torch.eye(dt)


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
