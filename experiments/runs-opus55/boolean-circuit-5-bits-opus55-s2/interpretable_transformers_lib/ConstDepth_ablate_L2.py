"""
Interpretable transformer for boolean-circuit-5-bits (hand-written weights, NO training).

Task: "b0 o1 b1 o2 b2 o3 b3 o4 b4 =" -> strict left-to-right fold of {&,|,^}.

Constant-depth view: the answer is the value of the LAST RESET (&0, |1, or the first bit)
XOR the parity of the FLIPS (^1) after it.
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
    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        scores = self.W_q(x) @ self.W_k(x).transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return attn @ self.W_v(x)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Attention is a transient read: it feeds the MLP but is not persisted."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attn(x))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 16,
                 n_layers: int = 3, d_ff: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None, :, :]
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
    """Three layers, each with a single job.

    L1 classify: each bit reads its op; R = is-a-reset, F = is-a-flip, Z = is-first-cell.
    L2 count:    every cell gets C = number of flips at or before it. Cell 0 has a huge score,
                 so all other cells share one fixed attention weight e^-L and the attention
                 output is a true sum; F at cell 0 is 0, so its large weight contributes nothing.
    L3 decide:   '=' jumps to the last reset r, reads its bit v and C_r, and outputs the parity
                 of y = v + C_= - C_r with a 5-neuron triangle wave.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.zeros_(p)
    A, X, b, tau, pos, one, R, Fl, Z, C, rA, rX, rF, rb, rC, S = range(16)
    tok = task.stoi
    beta, M, L = 30.0, 20.0, 2.0
    l1, l2, l3 = model.blocks
    with torch.no_grad():
        model.token_emb.weight[tok["0"], tau] = 1
        model.token_emb.weight[tok["1"], tau] = 1
        model.token_emb.weight[tok["1"], b] = 1
        model.token_emb.weight[tok["&"], A] = 1
        model.token_emb.weight[tok["^"], X] = 1
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, pos] = p
            model.pos_emb.weight[p, one] = 1

        # L1: attend to the most recent token of the opposite kind
        at = l1.attn
        at.W_q.weight[0, one] = 1
        at.W_q.weight[1, one], at.W_q.weight[1, tau] = 1, -2
        at.W_k.weight[0, pos], at.W_k.weight[1, tau] = beta, beta * M
        at.W_v.weight[rA, A], at.W_v.weight[rX, X] = 1, 1
        f1, f2 = l1.mlp.fc1, l1.mlp.fc2
        f1.weight[0, rA], f1.weight[0, b], f1.weight[0, tau] = 1, -1, 1          # &0
        f1.bias[0] = -1
        f1.weight[1, b], f1.weight[1, rA], f1.weight[1, rX], f1.weight[1, tau] = 1, -1, -1, 1  # |1
        f1.bias[1] = -1
        f1.weight[2, pos], f1.weight[2, b] = -1, -1                               # first bit is 0
        f1.bias[2] = 1
        f1.weight[3, rX], f1.weight[3, b], f1.weight[3, tau] = 1, 1, 1            # ^1
        f1.bias[3] = -2
        f1.weight[4, pos] = -1                                                    # first cell
        f1.bias[4] = 1
        f2.weight[R, 0], f2.weight[R, 1], f2.weight[R, 2] = 1, 1, 1
        f2.weight[Fl, 3] = 1
        f2.weight[Z, 4] = 1

        # L2: sum F over the prefix
        at = l2.attn
        at.W_q.weight[0, one], at.W_k.weight[0, Z] = 1, L
        at.W_v.weight[rF, Fl] = 1
        l2.mlp.fc1.weight[0, rF] = math.exp(L)
        l2.mlp.fc2.weight[C, 0] = 1

        # L3: jump to the last reset, then take parity
        at = l3.attn
        at.W_q.weight[0, one] = 1
        at.W_k.weight[0, R], at.W_k.weight[0, pos] = beta * M, beta
        at.W_v.weight[rb, b], at.W_v.weight[rC, C] = 1, 1
        f1, f2 = l3.mlp.fc1, l3.mlp.fc2
        for j in range(5):
            f1.weight[j, rb], f1.weight[j, C], f1.weight[j, rC] = 1, 1, -1
            f1.bias[j] = -j
            f2.weight[S, j] = 1 if j == 0 else (-2 if j % 2 else 2)
        f2.bias[S] = -0.5

        model.head.weight[tok["1"], S] = 4
        model.head.weight[tok["0"], S] = -4
    verify_exhaustive(model, task)


model_shorthand_name = "ConstDepth_ablate_L2"
model_description = "Ablation: dominated softmax weakened to L=2, so the flip count is no longer a true sum (denominator depends on position)."


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
