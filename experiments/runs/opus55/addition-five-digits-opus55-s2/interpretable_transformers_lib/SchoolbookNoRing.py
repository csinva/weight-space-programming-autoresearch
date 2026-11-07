"""Hand-written transformer for 5-digit addition. No training anywhere."""

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
    """One head of cross-attention to "aaaaa+bbbbb"."""

    PROMPT_LEN = 11

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        T, P = keys.shape[1], self.PROMPT_LEN
        scores = self.W_q(keys) @ keys[:, :P].transpose(-2, -1)
        mask = torch.triu(torch.ones(T, P, dtype=torch.bool, device=keys.device), diagonal=1)
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ values[:, :P]


class SimpleTransformer(nn.Module):
    """Two heads: the current column, and the columns below it. Then a carry bit."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 6):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, 1)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.column_attn = CausalSelfAttention(d_model)
        self.tail_attn = CausalSelfAttention(d_model)
        self.carry_fc = nn.Linear(1, 2)     # two ReLUs make a step
        self.mix = nn.Linear(2, 1, bias=False)
        self.wrap_fc = nn.Linear(1, 2)      # two ReLUs detect z >= 10
        self.fold = nn.Linear(3, 1, bias=False)
        self.head = nn.Linear(1, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        keys = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        values = self.token_emb(ids)
        column = self.column_attn(keys, values)
        ramps = F.relu(self.carry_fc(self.tail_attn(keys, values)))
        carry = ramps[..., [0]] - ramps[..., [1]]
        z = self.mix(torch.cat([column, carry], dim=-1))    # column sum, 0 to 19
        wraps = F.relu(self.wrap_fc(z))
        return self.head(self.fold(torch.cat([z, wraps], dim=-1)))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    # Circuit: schoolbook addition, in float32. For column j, one head averages
    # a_j and b_j. The other reads the tail t_j = sum_{k>j} (a_k + b_k) 10^(j-k),
    # and the carry into column j is exactly the test t_j >= 1. The digit is then
    # z = a_j + b_j + carry, an integer in 0..19, and z - 10 [z >= 10] is the digit.
    for p in model.parameters():
        p.data.zero_()

    KEY_SINK = 5
    ALPHA = 0.25    # common attention scale of the tail head
    OFF = -80.0     # log-weight of anything that must be ignored
    DELTA = 5e-6    # half the finest step of the tail
    STEEP = 1.6e6   # ramp width, narrower than the gap below the threshold

    column_table = torch.full((6, 6), OFF, dtype=torch.float64)
    tail_table = torch.full((6, 6), OFF, dtype=torch.float64)
    column_table[0, KEY_SINK] = 0.0     # step 0 has no column: it is only the carry
    for i in range(6):
        j = i - 1
        if j >= 0:
            column_table[i, j] = 0.0    # equal weight on a_j and b_j
        weights = {k: ALPHA * 10.0 ** (j - k) for k in range(j + 1, 5)}
        for k, w in weights.items():
            tail_table[i, k] = math.log(w)
        tail_table[i, KEY_SINK] = math.log(1 - 2 * sum(weights.values()))

    pos, tok = model.pos_emb.weight.data, model.token_emb.weight.data
    for c in range(5):
        pos[c, c] = pos[6 + c, c] = 1.0     # a_c and b_c share the key of column c
    pos[5, KEY_SINK] = 1.0
    for i in range(6):
        pos[11 + i, i] = 1.0                # step i reuses key dim i as its query
    for d in range(10):
        tok[d, 0] = float(d)

    model.column_attn.W_q.weight.data[:] = column_table.T.float()
    model.tail_attn.W_q.weight.data[:] = tail_table.T.float()

    # Carry bit: relu(s + 1) - relu(s), a unit step spread over a width of 1 / STEEP.
    threshold = ALPHA * (1 - DELTA)
    model.carry_fc.weight.data[:, 0] = STEEP
    model.carry_fc.bias.data[0] = 1 - STEEP * threshold
    model.carry_fc.bias.data[1] = -STEEP * threshold

    model.mix.weight.data[0, 0] = 2.0   # the column head returned (a_j + b_j) / 2
    model.mix.weight.data[0, 1] = 1.0

    # Mod 10: relu(z - 9) - relu(z - 10) is 0 up to z = 9 and 1 from z = 10 on.
    model.wrap_fc.weight.data[:, 0] = 1.0
    model.wrap_fc.bias.data[:] = torch.tensor([-9.0, -10.0])
    model.fold.weight.data[0] = torch.tensor([1.0, -10.0, 10.0])

    # Head: linear templates 2 d R - d^2, whose argmax is the nearest integer d.
    for d in range(10):
        model.head.weight.data[d, 0] = 2.0 * d
        model.head.bias.data[d] = -float(d * d)
    model.head.bias.data[10:] = -1e6    # '+' and '=' are never written

    _stress_test(model)


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
    wrong, shown = 0, []
    with torch.no_grad():
        for start in range(0, len(a), 4096):
            aa, bb = a[start:start + 4096], b[start:start + 4096]
            ids = torch.tensor([[int(c) for c in f"{p:05d}"] + [10] + [int(c) for c in f"{q:05d}"] + [11]
                                for p, q in zip(aa, bb)])
            for _ in range(6):
                ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
            for row, p, q in zip(ids[:, 12:].tolist(), aa, bb):
                if "".join(map(str, row)) != f"{p + q:06d}":
                    wrong += 1
                    if len(shown) < 5:
                        shown.append((p, q, row))
    print(f"STRESS: {wrong} wrong of {len(a)}; examples: {shown}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SchoolbookNoRing"
model_description = "Schoolbook circuit in float32 with only attention, ReLU and linear layers: carry bit from the tail, two more ReLUs subtract 10 when z >= 10, and a linear template head reads the digit."


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
