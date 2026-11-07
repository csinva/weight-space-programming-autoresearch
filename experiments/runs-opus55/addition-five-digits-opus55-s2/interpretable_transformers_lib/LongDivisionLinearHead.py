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
    """Plain causal dot-product attention over all tokens seen so far."""

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        T = keys.shape[1]
        scores = queries @ keys.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=keys.device), diagonal=1)
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ values


class SimpleTransformer(nn.Module):
    """One attention read of the remainder, decoded by a plain linear head."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 6):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, 1)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn = CausalSelfAttention()
        self.head = nn.Linear(1, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        p = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        q, k = p[..., :3], p[..., 3:]               # each position's query and key
        sign = k[..., [0]] - k[..., [2]]            # + for prompt digits, - for answer digits
        return self.head(self.attn(q, k, sign * self.token_emb(ids)))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    # Circuit: long division. With x_j = (A + B) / 10^(4 - j) and P the number
    # already written, the remainder R = x_j - 10 P lies in [0, 10) and the next
    # digit is floor(R). No periodic unit is needed: the written digits do the wrap.
    model.double()  # R is a difference of terms up to 2e5
    for p in model.parameters():
        p.data.zero_()

    LOG10 = math.log(10.0)
    GAIN = 1e-6     # attention output is GAIN R; undone once in the token values
    DELTA = 5e-6    # half the finest step of R (1e-5 at step 0), keeps R off borders

    # Prompt column c has key (1, -c LOG10, 0) and answer digit m has key
    # (0, -m LOG10, 1); '+' and '=' have the zero key, so they score 0 and sink.
    # The step-i query (log t, 1, log u) then weights them t 10^-c and u 10^-m. With
    # total mass Z, solve t / Z = GAIN 10^j and u / Z = GAIN 10^i for j = i - 1.
    pos, tok = model.pos_emb.weight.data, model.token_emb.weight.data
    vec = lambda *xs: torch.tensor(xs, dtype=pos.dtype)
    S = 2 * sum(10.0 ** -c for c in range(5))
    for c in range(5):
        pos[c, 3:] = pos[6 + c, 3:] = vec(1.0, -c * LOG10, 0.0)
        pos[12 + c, 3:] = vec(0.0, -c * LOG10, 1.0)
    for i in range(6):
        j = i - 1
        S_written = sum(10.0 ** -m for m in range(i))
        Z = 2 / (1 - GAIN * (S * 10.0 ** j + S_written * 10.0 ** i))
        pos[11 + i, :3] = vec(math.log(GAIN * 10.0 ** j * Z), 1.0, math.log(GAIN * 10.0 ** i * Z))
    for d in range(10):
        tok[d, 0] = d / GAIN

    # Head: linear templates. -(R - c)^2 = 2 c R - c^2 - R^2, and R^2 is shared by
    # every class, so the argmax is the nearest centre c = d + 0.5, i.e. floor(R).
    for d in range(10):
        c = d + 0.5 - DELTA
        model.head.weight.data[d, 0] = 2 * c
        model.head.bias.data[d] = -c * c
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
model_shorthand_name = "LongDivisionLinearHead"
model_description = "No ring or periodic unit: causal attention subtracts the digits already written to get a remainder in [0,10), and a plain linear head with quadratic templates reads floor(R)."


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
