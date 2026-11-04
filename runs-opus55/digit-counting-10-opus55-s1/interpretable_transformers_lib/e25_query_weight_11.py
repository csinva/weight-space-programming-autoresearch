"""Hand-written transformer for digit-counting-10. No training anywhere."""

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
        self.gain = nn.Parameter(torch.ones(1))   # scales the attention output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = x.view(B, T, H, dh).transpose(1, 2)   # keys are the raw hidden state
        v = torch.zeros_like(x)
        v[..., :10] = x[..., :10]                 # values are the digit channels
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return self.gain * (attn @ v).transpose(1, 2).contiguous().view(B, T, D)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Attention then MLP, with no LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    """1-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 15,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 10,
    ):
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Count digits with one query-weighted histogram and threshold gates.

    Attention, at pos 11 and 12: one head averages the digit one-hots of pos
    0-10, with pos 10 weighted Q times as much as each other position. Scaled
    by 10 + Q this gives H[d] = count[d] + Q * [d is the query].
    MLP: gate d is ReLU(H[d] - Q). Since count[d] <= 10 < Q, only the query's
    gate can fire, and it outputs the integer count c, stored in X.
    Head: logit_k = 2 k X - k^2, the same parabola decoder on integer templates.
    """
    # D = token one-hots, also the histogram; U = units step; T = token is '1'.
    D, DIG, ANS, X, U, T = 0, 10, 11, 12, 13, 14
    BETA = 20.0                               # attention sharpness
    Q = 11.0                                  # extra weight on the query position
    scale = BETA * math.sqrt(model.d_model)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        for d in range(10):
            model.token_emb.weight[d, D + d] = 1.0
        model.token_emb.weight[1, T] = 1.0
        model.pos_emb.weight[:10, DIG] = 1.0      # pos 0-9: the digit string
        # pos 10: a score bonus of log Q gives the query weight Q.
        model.pos_emb.weight[10, DIG] = 1.0 + math.log(Q) / BETA
        model.pos_emb.weight[11:13, ANS] = 1.0    # pos 11-12: answer steps
        model.pos_emb.weight[12, U] = 1.0
        # At pos 12 the token is normally '0'; cancel it so H[0] stays clean.
        model.pos_emb.weight[12, D] = -1.0

        a = model.blocks[0].attn
        a.W_q.weight[DIG, ANS] = scale            # look at the digits and query
        a.gain.fill_(10.0 + Q)                    # undo the weighted average

        # One threshold gate per digit, summed into X.
        m = model.blocks[0].mlp
        for d in range(10):
            m.fc1.weight[d, D + d] = 1.0
            m.fc1.bias[d] = -Q
            m.fc2.weight[X, d] = 1.0

        # Head. ANS-only terms apply at pos 11, and U cancels them at pos 12 to
        # leave the units parabola.
        BIG = 1000.0
        model.head.weight[1, ANS] = -19.0         # at pos 11, '1' iff 2 X > 19
        model.head.weight[2:, ANS] = -BIG
        for k in range(10):
            model.head.weight[k, X] = 2.0 * k
            model.head.weight[k, U] = -k * k - model.head.weight[k, ANS]
        model.head.weight[0, T] = 2 * BIG         # after a tens '1', units is 0
    stress_test(model, task)


def stress_test(model, task, n_per_case: int = 20) -> None:
    """Verification only: greedy-decode crafted prompts for every (query, count)."""
    import random
    rng = random.Random(123)
    wrong, total = 0, 0
    with torch.no_grad():
        for q in range(10):
            for c in range(11):
                for _ in range(n_per_case):
                    others = [d for d in range(10) if d != q]
                    digits = [q] * c + [rng.choice(others) for _ in range(10 - c)]
                    rng.shuffle(digits)
                    ids = task.encode("".join(map(str, digits)) + f"{q}=")
                    for _ in range(2):
                        logits = model(torch.tensor([ids]))
                        ids.append(int(logits[0, -1].argmax()))
                    total += 1
                    if task.decode(ids[-2:]) != f"{c:02d}":
                        wrong += 1
                        if wrong <= 5:
                            print("STRESS FAIL", digits, q, c, task.decode(ids[-2:]))
    print(f"STRESS {total - wrong}/{total} correct")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e25_query_weight_11"
model_description = "Query-weight sweep: Q=11, the smallest value allowed by count <= 10 < Q."


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
