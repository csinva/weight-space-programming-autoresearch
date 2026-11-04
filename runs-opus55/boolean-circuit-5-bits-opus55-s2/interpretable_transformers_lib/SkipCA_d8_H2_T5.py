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
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """One CA step: read the previous cell, then recompute this cell's state from scratch."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.mlp(x + self.attn(x))


class SimpleTransformer(nn.Module):
    """A single tied Block applied n_layers times (a causal 1-D cellular automaton)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 8,
                 n_heads: int = 2, n_layers: int = 5, d_ff: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.block = Block(d_model, n_heads, d_ff)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        base = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        h = base
        for _ in range(self.n_layers):
            h = self.block(h, base)
        return self.head(h)


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
    """Skip CA: state lives on odd cells (ops and '='); each step hops two cells.

    Cell p reads (S, A, X) from p-2 with head 0 and the bit from p-1 with head 1.
    channels: c0,n0 / c1,n1 = position circles for each head | A,X = own op flags
              b = own bit, also collects k=S+b | S = state | rA,rX = op flags read from p-2
    gate: S_new = relu(k-rA) - relu(2k-3+rX-rA)
    """
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.zeros_(p)
    tok, at, m = task.stoi, model.block.attn, model.block.mlp
    dh = model.d_model // model.n_heads
    theta, beta = math.pi / 6, (200 * math.sqrt(dh)) ** 0.5

    def look_back(head_dim, offset):
        # query = position rotated back `offset` cells, key = position, in dims head_dim, head_dim+1
        ca, sa = beta * math.cos(offset * theta), beta * math.sin(offset * theta)
        at.W_q.weight[head_dim, C], at.W_q.weight[head_dim, N] = ca, sa
        at.W_q.weight[head_dim + 1, C], at.W_q.weight[head_dim + 1, N] = -sa, ca
        at.W_k.weight[head_dim, C], at.W_k.weight[head_dim + 1, N] = beta, beta

    C, N, A, X, b, S, rA, rX = range(8)
    with torch.no_grad():
        model.token_emb.weight[tok["1"], b] = 1
        model.token_emb.weight[tok["&"], A] = 1
        model.token_emb.weight[tok["^"], X] = 1
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, C] = math.cos(theta * p)
            model.pos_emb.weight[p, N] = math.sin(theta * p)
        look_back(0, 2)
        look_back(dh, 1)
        # head 0 values (dims 2,3 + one spare): op flags and state. Head 1 value: bit.
        at.W_v.weight[2, A], at.W_v.weight[3, X], at.W_v.weight[dh + 2, S] = 1, 1, 1
        at.W_v.weight[dh + 3, b] = 1
        at.W_o.weight[rA, 2], at.W_o.weight[rX, 3] = 1, 1
        at.W_o.weight[b, dh + 2], at.W_o.weight[b, dh + 3] = 1, 1
        # gate
        m.fc1.weight[0, b], m.fc1.weight[0, rA] = 1, -1
        m.fc1.weight[1, b], m.fc1.weight[1, rA], m.fc1.weight[1, rX] = 2, -1, 1
        m.fc1.bias[1] = -3
        m.fc2.weight[S, 0], m.fc2.weight[S, 1] = 1, -1
        # readout: threshold S at 1/2
        model.head.weight[tok["1"], S] = 2
        model.head.weight[tok["0"], S] = -2
        model.head.bias[tok["0"]] = 1
    verify_exhaustive(model, task)


model_shorthand_name = "SkipCA_d8_H2_T5"
model_description = "Tied skip CA: head 0 reads state+op from p-2, head 1 reads the bit at p-1, so state hops odd cells and 5 iterations suffice (half the depth of the 1-head CA)."


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
