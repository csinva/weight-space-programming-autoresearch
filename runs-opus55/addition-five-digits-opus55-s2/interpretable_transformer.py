"""Hand-written transformer for 5-digit addition. No training anywhere."""

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

class CausalSelfAttention(nn.Module):
    """Cross-attention to "aaaaa+bbbbb": keys from position, values from the token."""

    PROMPT_LEN = 11

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        T, P = keys.shape[1], self.PROMPT_LEN
        scores = keys @ keys[:, :P].transpose(-2, -1)   # a position's query is its embedding
        mask = torch.triu(torch.ones(T, P, dtype=torch.bool, device=keys.device), diagonal=1)
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ values[:, :P]


class SimpleTransformer(nn.Module):
    """One attention read, decoded by a head tied to the token embeddings."""

    # Which embedding slot each position uses: a_c and b_c are the same place value.
    SLOTS = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_emb = nn.Embedding(12, d_model)    # 5 columns, '+', 6 output steps
        self.attn = CausalSelfAttention()
        self.digit_value = nn.Parameter(torch.zeros(1))     # embedding of digit d is d * this
        self.head_turn = nn.Parameter(torch.zeros(1))       # angle between adjacent digits
        self.head_phase = nn.Parameter(torch.zeros(1))      # angle of digit 0's sector

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        slots = torch.tensor(self.SLOTS[:T], device=ids.device)
        keys = self.pos_emb(slots)[None].expand(B, -1, -1)
        digits = ids.masked_fill(ids >= 10, 0)[..., None].to(keys.dtype)    # '+' and '=' are 0
        z = self.attn(keys, self.digit_value * digits)      # z is the angle of the running sum
        tokens = torch.arange(self.vocab_size, device=ids.device, dtype=keys.dtype)
        logits = torch.cos(z - (self.head_turn * tokens + self.head_phase))
        logits[..., 10:] = float("-inf")                    # only digits are ever written
        return logits


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    # Circuit: the digit written for column j is floor(x_j) mod 10, where
    # x_j = (A + B) / 10^(4 - j) is the whole sum, shifted. Each 10 of x_j is one
    # full turn of a ring, so higher columns vanish and carries are automatic.
    model.double()  # float32 scores 0.5%, see RingSharedSlotsFloat32
    for p in model.parameters():
        p.data.zero_()

    LOG10 = math.log(10.0)
    GAIN = 1e-5     # attention output is GAIN x_j; undone once in the token values
    DELTA = 5e-6    # half the finest step of x (1e-5 at step 0), keeps x off borders

    # Dot-product attention: a column-c key (1, -c LOG10) against the step-i query
    # (log t, 1) scores log(t 10^-c), the place value of column c. The '+' slot stays
    # at the origin, so it always scores 0 and acts as a value-0 sink of weight 1.
    # That makes the column weight t 10^-c / (1 + S t), with S = 2 sum 10^-c. The
    # one free number per step, t, therefore sets the gain: t / (1 + S t) = GAIN 10^j
    # shifts the decimal point to column j = i - 1.
    slots = model.pos_emb.weight.data
    S = 2 * sum(10.0 ** -c for c in range(5))
    for c in range(5):
        slots[c] = torch.tensor([1.0, -c * LOG10], dtype=slots.dtype)
    for i in range(6):
        t = 1 / (1 / (GAIN * 10.0 ** (i - 1)) - S)
        slots[6 + i] = torch.tensor([math.log(t), 1.0], dtype=slots.dtype)
    model.digit_value.data[0] = 2 * math.pi / 10 / GAIN     # a digit is 1/10 turn per unit

    # Head: digit d owns the sector centred on d + 0.5 tenths of a turn, rotated back
    # by DELTA so exact integers land in the upper one.
    model.head_turn.data[0] = 2 * math.pi / 10
    model.head_phase.data[0] = 2 * math.pi / 10 * (0.5 - DELTA)

    _stress_test(model, n_random=1000000)


def _stress_test(model, n_random: int = 50000) -> None:
    """Check the model on random pairs plus hand-picked long carry chains."""
    gen = torch.Generator().manual_seed(1)
    a = torch.randint(0, 100000, (n_random,), generator=gen).tolist()
    b = torch.randint(0, 100000, (n_random,), generator=gen).tolist()
    edges = [0, 1, 9, 10, 99, 100, 999, 1000, 9999, 10000, 49999, 50000, 50001, 89999, 90000, 90001, 99990, 99998, 99999]
    for p in edges:
        for q in edges:
            a.append(p)
            b.append(q)
    for s in (9999, 10000, 19999, 99999, 100000, 150000, 199998):   # complements
        for p in range(max(0, s - 99999), min(s, 99999) + 1, 997):
            a.append(p)
            b.append(s - p)
    for sums in itertools.product((0, 7, 8, 9, 10, 11, 18), repeat=5):  # column patterns
        p, q = 0, 0
        for s in sums:
            d = int(torch.randint(max(0, s - 9), min(s, 9) + 1, (1,), generator=gen))
            p, q = 10 * p + d, 10 * q + s - d
        a.append(p)
        b.append(q)
    wrong, shown, margin = 0, [], float("inf")
    with torch.no_grad():
        for start in range(0, len(a), 4096):
            aa, bb = a[start:start + 4096], b[start:start + 4096]
            ids = torch.tensor([[int(c) for c in f"{p:05d}"] + [10] + [int(c) for c in f"{q:05d}"] + [11]
                                for p, q in zip(aa, bb)])
            for _ in range(6):
                top = model(ids)[:, -1].topk(2, dim=-1)
                margin = min(margin, (top.values[:, 0] - top.values[:, 1]).min().item())
                ids = torch.cat([ids, top.indices[:, [0]]], dim=1)
            for row, p, q in zip(ids[:, 12:].tolist(), aa, bb):
                if "".join(map(str, row)) != f"{p + q:06d}":
                    wrong += 1
                    if len(shown) < 5:
                        shown.append((p, q, row))
    print(f"STRESS: {wrong} wrong of {len(a)}; smallest logit margin {margin:.3e}; examples: {shown}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "RingFinalMillionCheck"
model_description = "RingLinearTokens verified on 1,000,000 random pairs plus the edge, complement and column-pattern sets, also reporting the smallest top-1 vs top-2 logit margin."


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
