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
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the input stream and append one result channel per head.
        return torch.cat([x, self.attn(x)], dim=-1)


class SimpleTransformer(nn.Module):
    """1-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 6,
        n_heads: int = 2,
        n_layers: int = 1,
        d_ff: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, 1)            # numeric value only
        self.pos_emb = nn.Embedding(max_seq_len, d_model - 1)   # BIAS, KEY_A, KEY_B, QUERY, IS_DIGIT
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(3, vocab_size, bias=False)        # reads (BIAS, A, B)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = self.pos_emb(torch.arange(T, device=ids.device))[None].expand(B, -1, -1)
        h = torch.cat([self.token_emb(ids), pos], dim=-1)
        for block in self.blocks:
            h = block(h)
        return self.head(h[..., [1, -2, -1]])


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Long division in attention. At answer place p the model computes
        R_p = (a + b - already_printed) / 10^p
    which always lies in [0, 10), so its floor is the next digit.
      head A: + (a + b) / 10^p        (reads the prompt digits)
      head B: - printed / 10^p        (reads the answer so far; the causal mask
                                       is what restricts it to printed digits)
    The '+' token has value 0 and is a huge sink of attention mass C, so each
    head's softmax normaliser is effectively the constant C. Scores are rank 2:
        score(query place p, key place q) = ln(10) * (q - p)
    so a query is just (1, -p ln 10) and a key is (q ln 10, 1).
    The output head classifies R_p.
    """
    torch.manual_seed(0)
    VAL, BIAS, KEY_A, KEY_B, QUERY, IS_DIGIT = range(6)   # input stream to attention
    T = model.max_seq_len
    NEG = -2000.0  # key value meaning "ignore this position"
    GAP = 1e-5     # smallest possible spacing of R_p (at place 5)
    C = 1e18       # sink mass: makes the digit mass negligible in the normaliser
    LN10, LNC = math.log(10.0), math.log(C)

    model.double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Tokens carry their numeric value.
        for i in range(10):
            model.token_emb.weight[task.stoi[str(i)], 0] = float(i)

        # Positions: which head may read them, and with what place value.
        # pos_emb omits VAL, so column c of it is stream dim c + 1.
        def pos(t, dim, value):
            model.pos_emb.weight[t, dim - 1] = value

        for t in range(T):
            pos(t, BIAS, 1.0)
            pos(t, KEY_A, NEG)
            pos(t, KEY_B, NEG)
        for j in range(5):
            for t in (j, 6 + j):                    # digit j of a, digit j of b
                pos(t, KEY_A, (4 - j) * LN10)
                pos(t, IS_DIGIT, 1.0)
        pos(5, KEY_A, LNC)                          # '+' is the sink of both heads
        pos(5, KEY_B, LNC)
        for t in range(12, T):
            pos(t, KEY_B, (17 - t) * LN10)          # token at t was printed for place 17 - t
            pos(t, IS_DIGIT, 1.0)
        for t in range(11, T):
            pos(t, QUERY, -(16 - t) * LN10)         # position t asks for place 16 - t

        attn = model.blocks[0].attn
        for head, key in ((0, KEY_A), (1, KEY_B)):
            attn.W_q.weight[2 * head, BIAS] = 1.0
            attn.W_q.weight[2 * head + 1, QUERY] = 1.0
            attn.W_k.weight[2 * head, key] = 1.0
            attn.W_k.weight[2 * head + 1, IS_DIGIT] = 1.0
            attn.W_v.weight[head, VAL] = 1.0

        # Head reads (BIAS, A, B). Parabolic classifier -(R - c_k)^2 with
        # R = C * (A - B); the R^2 term is shared, so it is linear in R with a bias.
        for k in range(10):
            c = k + 0.5 - GAP / 2
            i = task.stoi[str(k)]
            model.head.weight[i, 0] = -c * c
            model.head.weight[i, 1] = 2 * c * C
            model.head.weight[i, 2] = -2 * c * C
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
model_shorthand_name = "long_division_rect"
model_description = "Long division with rectangular attention (2-dim queries/keys, 1-dim values), a 1-dim token embedding, a 5-dim position embedding and a 3-input head, removing every weight the circuit never uses."


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
