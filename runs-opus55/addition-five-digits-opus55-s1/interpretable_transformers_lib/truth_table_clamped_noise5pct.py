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

NOISE_SIGMA = 0.05  # multiplicative weight noise, used only for robustness probes


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Rectangular attention: d_key-dim queries/keys and d_val-dim values per head."""

    def __init__(self, d_model: int, n_heads: int, d_key: int, d_val: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_val = d_val
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_val, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dk, dv = self.n_heads, self.d_key, self.d_val
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dv)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_key=6, d_val=10)
        self.W_o = nn.Linear(n_heads * 10, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.W_o(self.attn(x))
        return x + self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """One weight-tied causal block applied n_layers times."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 65,
        n_heads: int = 2,
        n_layers: int = 6,
        d_ff: int = 343,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, 10)            # one-hot digit
        self.pos_emb = nn.Embedding(max_seq_len, d_model - 10)   # all other stream dims
        self.block = Block(d_model, n_heads, d_ff)               # one cell, applied n_layers times
        self.final_ln = nn.Identity()
        self.head = nn.Linear(11, vocab_size, bias=False)        # reads (BIAS, OUT one-hot)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        h = torch.cat([self.token_emb(ids), pos], dim=-1)
        for _ in range(self.n_layers):
            h = self.block(h)
        return self.head(h[..., [10] + list(range(55, 65))])


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Digital ripple-carry adder. Digits are one-hot everywhere and the MLP is the
    literal truth table of a full adder, so each row is an AND gate. Carry rows
    saturate at 1 (two ReLU units form a clamped ramp), so a weakened carry is
    restored at every place instead of decaying along a propagate chain.
    One cell is applied six times with shared weights.
      STEP: one-hot of the place being added, the query of both heads
      TTL:  one-hot countdown; TTL[0] is set when the cell reaches the place
            this position must print
    Each application does:
      head 0 copies digit a_L into register A, head 1 copies b_L into B
      carry rows:  AND(A[i], B[j]) for i + j >= 10, AND(A[i], B[j], CARRY) for
                   i + j == 9
      print rows:  AND(A[i], B[j], CARRY or not, TTL[0]) set OUT[(i+j+c) mod 10]
      A, B, CARRY are cleared and the registers shift.
    Above place 4 the heads read '+', whose embedding is the digit 0.
    """
    torch.manual_seed(0)
    DIG, BIAS, KEY = 0, 10, 11
    STEP, TTL = KEY + 11, KEY + 17
    A, B, CARRY, OUT = KEY + 23, KEY + 33, KEY + 43, KEY + 44
    T = model.max_seq_len
    SCORE = 30.0   # score of the one position a head should read
    b = model.block

    def pos(t, dim, value):
        model.pos_emb.weight[t, dim - 10] = value

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Tokens are one-hot digits, and '+' counts as the digit 0.
        for i in range(10):
            model.token_emb.weight[task.stoi[str(i)], DIG + i] = 1.0
        model.token_emb.weight[task.stoi["+"], DIG] = 1.0

        for t in range(T):
            pos(t, BIAS, 1.0)
            pos(t, STEP, 1.0)
            if t < 11:
                pos(t, KEY + t, 1.0)
            elif t < 17:
                pos(t, TTL + (16 - t), 1.0)

        # Attention: head h, query L is STEP[L], and the one key for that place.
        for L in range(6):
            for h, j in enumerate((4 - L, 10 - L) if L < 5 else (5, 5)):
                b.attn.W_q.weight[h * 6 + L, STEP + L] = 1.0
                b.attn.W_k.weight[h * 6 + L, KEY + j] = SCORE
        for h in range(2):
            for i in range(10):
                b.attn.W_v.weight[h * 10 + i, DIG + i] = 1.0
                b.W_o.weight[(A if h == 0 else B) + i, h * 10 + i] = 1.0

        unit = 0

        def gate(inputs, threshold, outputs):
            """An AND gate: relu(sum(inputs) - threshold) * 2."""
            nonlocal unit
            for dim, sign in inputs:
                b.fc1.weight[unit, dim] = sign
            b.fc1.bias[unit] = -threshold
            for dim, sign in outputs:
                b.fc2.weight[dim, unit] = 2.0 * sign
            unit += 1

        def clamped_gate(inputs, threshold, output):
            """A saturating AND gate, 0 below threshold - 0.25 and 1 above threshold."""
            nonlocal unit
            for offset, sign in ((0.25, 1.0), (0.0, -1.0)):
                for dim, weight in inputs:
                    b.fc1.weight[unit, dim] = 4.0 * weight
                b.fc1.bias[unit] = -4.0 * (threshold - offset)
                b.fc2.weight[output, unit] = sign
                unit += 1

        # Truth table, carry rows.
        for i in range(10):
            for j in range(10):
                if i + j >= 10:
                    clamped_gate([(A + i, 1), (B + j, 1)], 1.5, CARRY)
                elif i + j == 9:
                    clamped_gate([(A + i, 1), (B + j, 1), (CARRY, 1)], 2.5, CARRY)

        # Truth table, print rows.
        for i in range(10):
            for j in range(10):
                for c in (0, 1):
                    carry_in = (CARRY, 1 if c else -1)
                    gate([(A + i, 1), (B + j, 1), carry_in, (TTL, 1)], 3.5 if c else 2.5,
                         [(OUT + (i + j + c) % 10, 1)])

        # Identity readers: clear A, B and CARRY, and shift STEP and TTL.
        def reader(dim, writes):
            nonlocal unit
            b.fc1.weight[unit, dim] = 1.0
            for target, sign in writes:
                b.fc2.weight[target, unit] = sign
            unit += 1

        for i in range(10):
            reader(A + i, [(A + i, -1)])
            reader(B + i, [(B + i, -1)])
        reader(CARRY, [(CARRY, -1)])
        for i in range(6):
            reader(STEP + i, [(STEP + i, -1)] + ([(STEP + i + 1, 1)] if i < 5 else []))
            reader(TTL + i, [(TTL + i, -1)] + ([(TTL + i - 1, 1)] if i > 0 else []))
        assert unit == model.d_ff, unit

        # Head reads (BIAS, OUT): the digit with the set bit wins.
        for k in range(10):
            model.head.weight[task.stoi[str(k)], 1 + k] = 10.0
        for sym in "+=":
            model.head.weight[task.stoi[sym], 0] = -1e6

        # Robustness probe: seeded multiplicative noise on every nonzero weight.
        for p in model.parameters():
            p.mul_(1.0 + NOISE_SIGMA * torch.randn_like(p))

    self_test(model, task)


@torch.no_grad()
def self_test(model, task, n_random: int = 50000) -> None:
    """Diagnostics only: greedy decode on random and adversarial carry-chain pairs."""
    g = torch.Generator().manual_seed(1)
    a = torch.randint(0, 100000, (n_random,), generator=g).tolist()
    b = torch.randint(0, 100000, (n_random,), generator=g).tolist()
    for p in range(0, 6):
        for k in range(1, 100):
            x = 10 ** p - k
            for y in (k, k - 1, k + 1):
                if 0 <= x < 100000 and 0 <= y < 100000:
                    a.append(x); b.append(y)
    for x in (0, 1, 9, 49999, 50000, 99999, 90909, 9999):
        for y in (0, 1, 9, 49999, 50000, 99999, 90909, 1):
            a.append(x); b.append(y)
    # Sums landing exactly on, or one away from, a multiple of each power of ten.
    for p in range(1, 6):
        xs = torch.randint(0, 100000, (2000,), generator=g).tolist()
        for x in xs:
            for delta in (-1, 0, 1):
                for m in (1, 3, 7):
                    y = (x // 10 ** p + m) * 10 ** p - x + delta
                    if 0 <= y < 100000:
                        a.append(x); b.append(y)
    prompts = [f"{x:05d}+{y:05d}=" for x, y in zip(a, b)]
    ids = torch.tensor([[task.stoi[c] for c in s] for s in prompts])
    was_training = model.training
    model.eval()
    out = ids
    for _ in range(task.answer_len):
        nxt = model(out)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    preds = ["".join(task.itos[i] for i in row[12:].tolist()) for row in out]
    wrong = [(x, y, r) for x, y, r in zip(a, b, preds) if r != f"{x + y:06d}"]
    print(f"SELF-TEST: {len(a) - len(wrong)}/{len(a)} correct; first failures: {wrong[:5]}")
    model.train(was_training)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "truth_table_clamped_noise5pct"
model_description = "Breaking-point search: saturating truth-table adder under 5% seeded multiplicative weight noise."


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
