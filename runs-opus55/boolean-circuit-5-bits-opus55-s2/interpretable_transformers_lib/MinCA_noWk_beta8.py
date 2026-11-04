"""
Interpretable transformer for boolean-circuit-5-bits (hand-written weights, NO training).

Task: "b0 o1 b1 o2 b2 o3 b3 o4 b4 =" -> strict left-to-right fold of {&,|,^}.

The fold rule is 3 ReLU neurons (S=state, b=bit, A/X = op is &/^, op | is default):
    S' = S + relu(b-S-A) - relu(S+A-b-1) - relu(S+X+b-2)
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
    """Single head. The key is the raw position; W_q rotates it; values live in logic space."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.W_q = nn.Linear(2, 2, bias=False)
        self.W_v = nn.Linear(d_model + 1, d_model, bias=False)  # (logic channels + state) -> logic

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        scores = self.W_q(pe) @ pe.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return attn @ self.W_v(x)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, 1)  # output is the scalar state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """One CA step: read the previous cell, then recompute this cell's state from scratch."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, s: torch.Tensor, e: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        return self.mlp(e + self.attn(torch.cat([e, s], dim=-1), pe))


class SimpleTransformer(nn.Module):
    """A single tied Block applied n_layers times (a causal 1-D cellular automaton)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 3,
                 n_heads: int = 1, n_layers: int = 10, d_ff: int = 2, theta: float = math.pi / 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # closed-form sinusoidal position (a buffer, like the original Transformer)
        angles = theta * torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer("pe", torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1))
        self.block = Block(d_model, n_heads, d_ff)
        self.head = nn.Linear(1, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:ids.shape[1]]
        e = self.token_emb(ids)
        s = self.block.mlp.fc2.bias.expand(*e.shape[:2], 1)  # start from "all neurons off"
        for _ in range(self.n_layers):
            s = self.block(s, e, pe)
        return self.head(s)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def verify_exhaustive(model, task) -> None:
    """Report accuracy on all unique prompts the task generator produces (no fitting)."""
    seen = {e.prompt: e.answer for e in task.generate_examples(30000, seed=1)}
    prompts = sorted(seen)
    ids = torch.tensor([task.encode(p) for p in prompts])
    with torch.no_grad():
        pred = model(ids)[:, -1].argmax(-1).tolist()
    ok = sum(task.is_correct(task.decode([i]), seen[p]) for i, p in zip(pred, prompts))
    print(f"EXHAUSTIVE: {ok}/{len(prompts)} unique prompts correct")


def write_weights(model: SimpleTransformer, task) -> None:
    """Tied CA. Logic channels: A,X = op | b = bit | S = state. Position (c,n) is a separate plane.

    S is stored centred (S-1/2) so the readout needs no bias. W_v routes the previous cell's
    S onto the bit channel, so h_b = b + S - 1/2 =: k - 1/2, and the gate is
        S_new = relu(h_b + 1/2 - A) - relu(2 h_b + X - A - 2)
    Op tokens carry their neutral bit (&:1, |:0, ^:0), so they pass S through unchanged.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.zeros_(p)
    A, X, b, S = range(4)
    c, n = 0, 1
    tok, at, m = task.stoi, model.block.attn, model.block.mlp
    theta, beta = math.pi / 6, 8.0  # beta = attention sharpness
    with torch.no_grad():
        model.token_emb.weight[tok["1"], b] = 1
        model.token_emb.weight[tok["&"], A] = 1
        model.token_emb.weight[tok["&"], b] = 1  # neutral element of AND
        model.token_emb.weight[tok["^"], X] = 1
        # query = position rotated back one cell, key = position: "look at cell p-1"
        at.W_q.weight[c, c], at.W_q.weight[c, n] = beta * math.cos(theta), beta * math.sin(theta)
        at.W_q.weight[n, c], at.W_q.weight[n, n] = -beta * math.sin(theta), beta * math.cos(theta)
        # values: op flags stay on their channels, state lands on the bit channel
        at.W_v.weight[A, A], at.W_v.weight[X, X], at.W_v.weight[b, S] = 1, 1, 1
        m.fc1.weight[0, b], m.fc1.weight[0, A] = 1, -1
        m.fc1.bias[0] = 0.5
        m.fc1.weight[1, b], m.fc1.weight[1, A], m.fc1.weight[1, X] = 2, -1, 1
        m.fc1.bias[1] = -2
        m.fc2.weight[0, 0], m.fc2.weight[0, 1] = 1, -1
        m.fc2.bias[0] = -0.5
        model.head.weight[tok["1"], 0] = 4
        model.head.weight[tok["0"], 0] = -4
    verify_exhaustive(model, task)


model_shorthand_name = "MinCA_noWk_beta8"
model_description = "Noise-margin sweep: same CA with beta=8 (neighbouring cell leak about e^-1.07, roughly 34% relative weight)."


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
