"""Hand-written transformer for boolean-circuit-5-bits. No training anywhere."""

from __future__ import annotations

import argparse
import csv
import itertools
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

N_STEPS = 5


class TapeAttention(nn.Module):
    """Causal cross-attention: queries from the registers, keys/values from the
    read-only tape (the embedded prompt)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, tape: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(tape).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(tape).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = TapeAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, tape: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attn(x, tape))  # reads are transient wires


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 9,
                 n_heads: int = 1, n_layers: int = 1, d_ff: int = 4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        h = tape = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for _ in range(N_STEPS):
            for block in self.blocks:
                h = block(h, tape)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Fold '0 | b0 op1 b1 ... op4 b4' at '='. A scalar pointer C addresses the
    tape through a parabolic attention score -(p - C)^2 / 2 and moves +2 per
    clock. C sits halfway between an op and its bit, so one head reads both."""
    ONE, AND, OR, XOR, EQ, S, C, P, P2 = range(9)   # named registers in x
    BETA = 90.0 * 3.0                         # score gap of 1 -> e^-90 (sqrt(9)=3)
    a, m = model.blocks[0].attn, model.blocks[0].mlp
    for p in model.parameters():
        p.data.zero_()
    # Token '0' is the zero vector: a bit is just the presence of ONE.
    for t, r in {1: ONE, 2: AND, 3: OR, 4: XOR, 5: EQ}.items():
        model.token_emb.weight.data[t, r] = 1.0
    for p in range(10):
        addr = -1.0 if p == 9 else float(p)   # '=' sits just left of b0
        model.pos_emb.weight.data[p, P] = addr
        model.pos_emb.weight.data[p, P2] = -addr * addr / 2.0
    model.pos_emb.weight.data[9, C] = -0.5    # boot between '=' and b0

    # Fetch: q = (C, 1), k = (p, -p^2/2), so q.k = -(p - C)^2 / 2 + C^2 / 2.
    a.W_q.weight.data[P, C] = BETA
    a.W_q.weight.data[P2, EQ] = BETA
    a.W_k.weight.data[P, P] = a.W_k.weight.data[P2, P2] = 1.0
    for r in (ONE, AND, OR, XOR):             # two equal hits, so W_o = 2
        a.W_v.weight.data[r, r] = 1.0
        a.W_o.weight.data[r, r] = 2.0
    a.W_v.weight.data[OR, EQ] = 1.0           # '=' reads as '|': state starts 0

    # Execute: (weights on registers, bias, delta on S).
    rules = [({S: -1, OR: 1, XOR: 1, ONE: 1}, -1, 1.0),   # 0 | 1, 0 ^ 1 -> 1
             ({S: 1, XOR: 1, ONE: 1}, -2, -1.0),          # 1 ^ 1 -> 0
             ({S: 1, AND: 1, ONE: -1}, -1, -1.0)]         # 1 & 0 -> 0
    for u, (w, b, delta) in enumerate(rules):
        for r, v in w.items():
            m.fc1.weight.data[u, r] = float(v)
        m.fc1.bias.data[u] = float(b)
        m.fc2.weight.data[S, u] = delta
    m.fc1.weight.data[3, EQ] = 1.0            # clock: C <- C + 2
    m.fc2.weight.data[C, 3] = 2.0

    w = model.head.weight.data                # S -> logit, EQ supplies the bias
    w[0, S], w[0, EQ], w[1, S], w[1, EQ] = -1.0, 0.5, 1.0, -0.5
    verify(model, task)


@torch.no_grad()
def verify(model, task) -> None:
    """Exhaustive check on all 2592 prompts (test only)."""
    def fold(s):
        v = int(s[0])
        for i in range(1, 9, 2):
            b = int(s[i + 1])
            v = v & b if s[i] == "&" else v | b if s[i] == "|" else v ^ b
        return str(v)
    ex = task.generate_examples(5000, seed=123)
    agree = sum(fold(e.prompt) == e.answer for e in ex) / len(ex)
    prompts = ["".join(c) + "=" for c in itertools.product(
        "01", "&|^", "01", "&|^", "01", "&|^", "01", "&|^", "01")]
    ids = torch.tensor([task.encode(p) for p in prompts])
    pred = model(ids)[:, -1].argmax(-1).tolist()
    ok = sum(task.itos[i] == fold(p) for i, p in zip(pred, prompts))
    print(f"VERIFY fold_vs_task={agree:.4f} exhaustive={ok}/{len(prompts)}")


model_shorthand_name = "E05_parabolic_pointer"
model_description = "Scalar pointer C addresses the tape via parabolic attention -(p-C)^2/2, moving +2 per clock; 3 state rules + 1 clock unit, d_model 9, d_ff 4."


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
