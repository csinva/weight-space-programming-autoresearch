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
NOISE = 0.01  # std of a fixed Gaussian perturbation of every weight (robustness test)


class TapeAttention(nn.Module):
    """Two bare attention heads; the tape is keys and values, reads are summed."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)   # head 1: the op cell
        self.W_q2 = nn.Linear(d_model, d_model, bias=False)  # head 2: the bit cell

    def forward(self, x: torch.Tensor, tape: torch.Tensor) -> torch.Tensor:
        k = tape.transpose(-2, -1)
        v = tape.clone()
        v[..., 3:] = 0.0   # address dims are keys only: never read out as values
        return (self.W_q(x) @ k).softmax(dim=-1) @ v + (self.W_q2(x) @ k).softmax(dim=-1) @ v


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.attn = TapeAttention(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, tape: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attn(x, tape))  # reads are transient wires


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 5,
                 n_layers: int = 1, d_ff: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(10, d_model)  # exactly the 10 prompt cells
        self.blocks = nn.ModuleList([Block(d_model, d_ff) for _ in range(n_layers)])
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
    """Fold '0 | b0 op1 b1 ... op4 b4' at '='. Pointer P addresses the tape via
    a parabolic score and moves +2 per clock. Head 1 peaks on the op cell and
    head 2 on the next (bit) cell: full reads, no ties. '=' reads as '|'."""
    ONE, O, S, P, P2 = range(5)               # named registers in x
    BETA = 10.0                               # score gap to a neighbour is BETA/2
    a, m = model.blocks[0].attn, model.blocks[0].mlp
    for p in model.parameters():
        p.data.zero_()
    # Token '0' is the zero vector: a bit is just the presence of ONE.
    # Scalar opcode o: '&' = 0, '|' = 1, '^' = 2. '=' IS '|', leaving 1 in O at
    # the CPU, so during a read O = 1 + o and ONE = b.
    model.token_emb.weight.data[1, ONE] = 1.0
    for t, o in {3: 1.0, 4: 2.0, 5: 1.0}.items():
        model.token_emb.weight.data[t, O] = o
    for p in range(10):
        addr = -1.0 if p == 9 else float(p)   # '=' sits just left of b0
        model.pos_emb.weight.data[p, P] = addr
        model.pos_emb.weight.data[p, P2] = -addr * addr / 2.0

    # Fetch: q = beta (P, 1), k = (p, -p^2/2): head 1 peaks at P, head 2 at P + 1
    # (the O register holds a constant 1 at the CPU, used as the +1 offset).
    a.W_q.weight.data[P, P] = a.W_q2.weight.data[P, P] = BETA
    a.W_q2.weight.data[P, O] = BETA
    a.W_q.weight.data[P2, P2] = a.W_q2.weight.data[P2, P2] = -2.0 * BETA  # P2 = -1/2 at '='

    # Execute: (weights on registers, bias, delta on S).
    rules = [({O: 1, ONE: 2, S: -3}, -3, 1.0),   # relu(o + 2b - 3s - 2): 0|1 -> 1
             ({O: 1, ONE: 2, S: -3}, -4, -1.0),  # relu(o + 2b - 3s - 3): 0^1 caps at +1
             ({O: 1, ONE: 1, S: 1}, -4, -1.0),   # relu(o + b + s - 3): 1 ^ 1 -> 0
             ({O: -1, ONE: -1, S: 1}, 1, -1.0)]  # relu(s - o - b):     1 & 0 -> 0
    for u, (w, b, delta) in enumerate(rules):
        for r, v in w.items():
            m.fc1.weight.data[u, r] = float(v)
        m.fc1.bias.data[u] = float(b)
        m.fc2.weight.data[S, u] = delta
    m.fc2.bias.data[P] = 2.0                  # clock: P <- P + 2

    w = model.head.weight.data                # S -> logit, O supplies the bias
    w[0, S], w[0, O], w[1, S], w[1, O] = -1.0, 0.5, 1.0, -0.5
    torch.manual_seed(0)
    for p in model.parameters():
        p.data += NOISE * torch.randn_like(p)
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


model_shorthand_name = "E22_keys_not_values_noise"
model_description = "Tests the amplification hypothesis: a fixed mask stops address dims (up to 32) reaching the MLP; same std 0.01 noise and seed as E20."


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
