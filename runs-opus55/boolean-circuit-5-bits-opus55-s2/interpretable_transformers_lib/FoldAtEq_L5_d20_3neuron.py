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
    """Attention is a transient read: it feeds the MLP but is not persisted."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attn(x))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 20,
                 n_heads: int = 2, n_layers: int = 5, d_ff: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
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
    """Fold at the '=' cell: layer t reads op@2t-1 and bit@2t, then applies the 3-neuron gate."""
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.zeros_(p)
    s = task.stoi
    dh = model.d_model // model.n_heads
    # channels: 0-8 pos one-hot | 9 B | 10 A | 11 X | 12 EQ | 13 S | 14 rB | 15 rA | 16 rX
    B, A, X, EQ, S, rB, rA, rX = 9, 10, 11, 12, 13, 14, 15, 16
    beta = (30 * math.sqrt(dh)) ** 0.5  # hard attention: match score = 30
    with torch.no_grad():
        model.token_emb.weight[s["1"], B] = 1
        model.token_emb.weight[s["&"], A] = 1
        model.token_emb.weight[s["^"], X] = 1
        model.token_emb.weight[s["="], EQ] = 1
        for p in range(9):
            model.pos_emb.weight[p, p] = 1
        for t, block in enumerate(model.blocks):
            at, m = block.attn, block.mlp
            op_pos, bit_pos = max(2 * t - 1, 0), 2 * t
            # head 0 reads the op cell, head 1 reads the bit cell (keys = position one-hot)
            at.W_q.weight[op_pos, EQ] = beta
            at.W_q.weight[dh + bit_pos, EQ] = beta
            for p in range(9):
                at.W_k.weight[p, p] = beta
                at.W_k.weight[dh + p, p] = beta
            at.W_v.weight[0, A] = 1
            at.W_v.weight[1, X] = 1
            at.W_v.weight[dh, B] = 1
            at.W_o.weight[rA, 0] = 1
            at.W_o.weight[rX, 1] = 1
            at.W_o.weight[rB, dh] = 1
            # 3-neuron gate
            m.fc1.weight[0, rB], m.fc1.weight[0, S], m.fc1.weight[0, rA] = 1, -1, -1
            m.fc1.weight[1, S], m.fc1.weight[1, rA], m.fc1.weight[1, rB] = 1, 1, -1
            m.fc1.bias[1] = -1
            m.fc1.weight[2, S], m.fc1.weight[2, rX], m.fc1.weight[2, rB] = 1, 1, 1
            m.fc1.bias[2] = -2
            m.fc2.weight[S, 0], m.fc2.weight[S, 1], m.fc2.weight[S, 2] = 1, -1, -1
        # readout: S=1 -> '1', S=0 -> '0' (EQ gives the threshold)
        model.head.weight[s["1"], S] = 2
        model.head.weight[s["0"], S] = -2
        model.head.weight[s["0"], EQ] = 1
    verify_exhaustive(model, task)


model_shorthand_name = "FoldAtEq_L5_d20_3neuron"
model_description = "5 layers fold at the '=' cell: position-addressed hard attention reads op+bit, a 3-ReLU gate updates state S; no LayerNorm, attention is a transient read."


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
