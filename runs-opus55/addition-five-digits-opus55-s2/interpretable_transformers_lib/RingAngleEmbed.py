"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
    """Keys come from position only and values from the token only."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        T = keys.shape[1]
        scores = self.W_q(keys) @ keys.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=keys.device), diagonal=1)
        return scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ values


class Block(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model)

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Attention output is an angle; the parameter-free MLP reads it on the circle.
        z = self.attn(keys, values)
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class SimpleTransformer(nn.Module):
    """One causal block: position keys, scalar token values, (cos, sin) readout."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 7):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, 1)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.block = Block(d_model)
        self.head = nn.Linear(2, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        keys = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        return self.head(self.block(keys, self.token_emb(ids)))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Circuit: x_j = sum_{k>=j} (a_k + b_k) 10^(j-k) is read by attention, and
    # the answer digit for column j is floor(x_j) mod 10. The carry never has to
    # ripple: it is just the fractional part of x_j crossing an integer.
    model.double()  # float32 fails 89999+90001, see RingAttnFloat32
    for p in model.parameters():
        p.data.zero_()

    KEY_SINK, KEY_OFF = 5, 6
    ALPHA = 0.25    # common attention scale, undone once in the token values
    OFF = -80.0     # log-weight of anything that must be ignored
    DELTA = 5e-6    # half the finest step of x (1e-5 at step 0), keeps x off borders

    # Attention table: row = output step i (column j = i - 1, j = -1 is the final
    # carry); entry = log attention weight on a key. '+' is the value-0 sink that
    # soaks up the remaining mass so every row normalises to the same ALPHA.
    table = torch.full((6, 7), OFF, dtype=torch.float64)
    for i in range(6):
        j = i - 1
        weights = {k: ALPHA * 10.0 ** (j - k) for k in range(max(j, 0), 5)}
        for k, w in weights.items():
            table[i, k] = math.log(w)
        table[i, KEY_SINK] = math.log(1 - 2 * sum(weights.values()))

    pos, tok = model.pos_emb.weight.data, model.token_emb.weight.data
    for c in range(5):
        pos[c, c] = pos[6 + c, c] = 1.0     # a_c and b_c share the key of column c
    pos[5, KEY_SINK] = 1.0
    for i in range(6):
        pos[11 + i, KEY_OFF] = 1.0          # '=' and answer tokens never get read
        pos[11 + i, i] = 1.0                # step i reuses key dim i as its query
    for d in range(10):
        tok[d, 0] = 2 * math.pi / 10 * d / ALPHA    # a digit is an angle, 1/10 turn per unit

    model.block.attn.W_q.weight.data[:, :6] = table.T

    # Head: digit d owns the ring sector centred on d + 0.5 tenths of a turn,
    # rotated back by DELTA so exact integers fall inside the upper sector.
    dtype = model.head.weight.dtype
    ring = lambda n: torch.tensor([math.cos(2 * math.pi * n / 10), math.sin(2 * math.pi * n / 10)], dtype=dtype)
    for d in range(10):
        model.head.weight.data[d] = ring(d + 0.5 - DELTA)

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
model_shorthand_name = "RingAngleEmbed"
model_description = "fc1 removed: token embeddings are angles, attention takes their place-weighted mean, a parameter-free (cos, sin) reads it, and the head is rotated by the border offset."


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
