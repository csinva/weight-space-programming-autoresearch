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
    """Rectangular attention: each head has d_key-dim queries/keys and a 1-dim value."""

    def __init__(self, d_model: int, n_heads: int, d_key: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_key
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_key
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, 1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).contiguous().view(B, T, H)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_key=1)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.W_o(self.attn(x))
        return x + self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """1-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 23,
        n_heads: int = 1,
        n_layers: int = 6,
        d_ff: int = 7,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, 1)            # numeric value only
        self.pos_emb = nn.Embedding(max_seq_len, d_model - 1)   # all stream dims except VAL
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(2, vocab_size, bias=False)        # reads (BIAS, OUT)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        h = torch.cat([self.token_emb(ids), pos], dim=-1)
        for block in self.blocks:
            h = block(h)
        return self.head(h[..., [1, 4]])


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Ripple-carry adder: block L is the full adder for place L, least significant
    first, and runs identically at every answer position.
      attention: PAIR = (a_L + b_L) / 2, with a fixed query (no position needed)
      MLP, with Y = 2 PAIR + CARRY:
        CARRY <- relu(Y - 9) - relu(Y - 10)       i.e. [Y >= 10]
        OUT   += the digit Y mod 10, but only if this position prints place L
        PAIR  <- 0, ready for the next block
    The print gate subtracts 20 from Y unless the position's one-hot flag is set.
    Every signal is a small integer, so all margins are at least 0.5.
    """
    torch.manual_seed(0)
    VAL, BIAS, PAIR, CARRY, OUT, POS = 0, 1, 2, 3, 4, 5  # pos_emb column c is dim c + 1
    T = model.max_seq_len
    NEG = -30.0   # score meaning "ignore this position"

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Tokens carry their numeric value; positions are BIAS plus one-hot.
        for i in range(10):
            model.token_emb.weight[task.stoi[str(i)], 0] = float(i)
        for t in range(T):
            model.pos_emb.weight[t, BIAS - 1] = 1.0
            model.pos_emb.weight[t, POS - 1 + t] = 1.0

        for L, block in enumerate(model.blocks):
            # Attention: both digits of place L, or the zero-valued '+' above place 4.
            readable = (4 - L, 10 - L) if L < 5 else (5,)
            block.attn.W_q.weight[0, BIAS] = 1.0
            block.attn.W_k.weight[0, POS:POS + T] = NEG
            for j in readable:
                block.attn.W_k.weight[0, POS + j] = 0.0
            block.attn.W_v.weight[0, VAL] = 1.0
            block.W_o.weight[PAIR, 0] = 1.0

            # MLP units: Y - c, ungated (0, 1) and print-gated (2, 3, 4), plus
            # identity readers (5, 6) that let fc2 clear CARRY and PAIR.
            flag = POS + 16 - L
            for unit, (c, gated) in enumerate(((9, 0), (10, 0), (0, 1), (9, 1), (10, 1))):
                block.fc1.weight[unit, PAIR] = 2.0
                block.fc1.weight[unit, CARRY] = 1.0
                block.fc1.bias[unit] = -float(c) - (20.0 if gated else 0.0)
                if gated:
                    block.fc1.weight[unit, flag] = 20.0
            block.fc1.weight[5, CARRY] = 1.0
            block.fc1.weight[6, PAIR] = 1.0

            block.fc2.weight[CARRY, 0] = 1.0
            block.fc2.weight[CARRY, 1] = -1.0
            block.fc2.weight[CARRY, 5] = -1.0
            block.fc2.weight[OUT, 2] = 1.0
            block.fc2.weight[OUT, 3] = -10.0
            block.fc2.weight[OUT, 4] = 10.0
            block.fc2.weight[PAIR, 6] = -1.0

        # Head reads (BIAS, OUT): logit_k = 2 k OUT - k^2, the parabola around k.
        for k in range(10):
            i = task.stoi[str(k)]
            model.head.weight[i, 0] = -float(k * k)
            model.head.weight[i, 1] = 2.0 * k
        for sym in "+=":
            model.head.weight[task.stoi[sym], 0] = -1e6

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
model_shorthand_name = "ripple_adder_v1"
model_description = "Six blocks forming a ripple-carry chain, one full adder per place: attention fetches the digit pair, the MLP computes the carry and writes the digit only at the position printing that place. All signals are small integers."


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
