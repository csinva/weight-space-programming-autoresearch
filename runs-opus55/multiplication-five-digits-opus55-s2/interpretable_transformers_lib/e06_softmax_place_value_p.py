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
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.Identity()  # LayerNorm would rescale the numeric slots
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 32,
        n_heads: int = 32,
        n_layers: int = 2,
        d_ff: int = 18,
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
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Schoolbook multiplication followed by long division, written in closed form.

    Slots of the hidden vector (each has one human meaning):
      BIAS   constant 1
      DIG    numeric value of the token (0 for '*', '=')
      POS    one-hot position
      TAU    place value of the digit being predicted from this cell
      A      the number a, gathered by attention
      PP     partial product (token digit) * a
      P      the product a*b, gathered by attention
      E      value of the answer digits emitted so far
      DOUT   next digit = sum_m (R >= m*TAU)
    Layer 1: attention reads a; MLP gates a by each b digit (repeated addition).
    Layer 2: attention reads the partial products and the emitted digits;
             MLP compares the remainder R = P - E against m*TAU, m = 1..9.
    """
    torch.manual_seed(0)
    model.double()
    for p in model.parameters():
        nn.init.zeros_(p)

    BIAS, DIG = 0, 1
    POS = 2  # 22 slots
    TAU, A, PP, P, E, DOUT, KA, KB = 24, 25, 26, 27, 28, 29, 30, 31
    BIG = 1e6    # exceeds any value a gated neuron must suppress
    BETA2 = 40.0  # attention sharpness: a tier of score gap gives e^-40 leakage
    EQ = 11       # '=' cell: holds DIG = 0, the fallback for unavailable answer digits

    te, pe = model.token_emb.weight.data, model.pos_emb.weight.data
    pe[:, BIAS] = 1.0
    for d in range(10):
        te[d, DIG] = d
        te[d, PP] = -BIG * d * (d - 1) / 2
    for t in range(pe.shape[0]):
        pe[t, POS + t] = 1.0
        pe[t, KA] = (4 - t) * math.log(10.0) if t <= 4 else -60.0
        pe[t, KB] = (10 - t) * math.log(10.0) if 6 <= t <= 10 else -60.0
        if 11 <= t <= 20:
            pe[t, TAU] = 10.0 ** (20 - t)

    def read(attn, head, cell, src, dst, weight, fallback=None):
        """Head `head` reads slot `src` of `cell` into slot `dst`, scaled by `weight`."""
        attn.W_q.weight.data[head, BIAS] = math.sqrt(BETA2)
        attn.W_k.weight.data[head, POS + cell] = 2 * math.sqrt(BETA2)
        if fallback is not None:
            attn.W_k.weight.data[head, POS + fallback] = math.sqrt(BETA2)
        attn.W_v.weight.data[head, src] = 1.0
        attn.W_o.weight.data[dst, head] = weight

    b1, b2 = model.blocks

    # Layer 1: gather a, then PP = sum_{m=1..9} (digit >= m) * a on b's cells.
    # One head: scores log(10^(4-i)) on a's cells make softmax weights 10^(4-i)/11111,
    # and the far-negative KA on every other cell (tiny weight) keeps them out.
    b1.attn.W_q.weight.data[0, BIAS] = 1.0
    b1.attn.W_k.weight.data[0, KA] = 1.0
    b1.attn.W_v.weight.data[0, DIG] = 1.0
    b1.attn.W_o.weight.data[A, 0] = 11111.0
    # With d integer and 0 <= A < BIG: relu(A + BIG*(d-m)) - relu(BIG*(d-m)) = (d >= m) * A.
    # Summed over m, the second terms total BIG*d(d-1)/2, which the token embedding
    # seeds into PP with a minus sign; the residual connection adds the MLP on top.
    for m in range(1, 10):
        n = m - 1
        b1.mlp.fc1.weight.data[n, A] = 1.0
        b1.mlp.fc1.weight.data[n, DIG] = BIG
        b1.mlp.fc1.bias.data[n] = -m * BIG
        b1.mlp.fc2.weight.data[PP, n] = 1.0

    # Layer 2: gather P and E, then step(R - m*TAU) with step(x) = relu(x+1) - relu(x).
    # Head 0 forms P the same way as a, over b's cells; the remaining heads read answer digits.
    b2.attn.W_q.weight.data[0, BIAS] = 1.0
    b2.attn.W_k.weight.data[0, KB] = 1.0
    b2.attn.W_v.weight.data[0, PP] = 1.0
    b2.attn.W_o.weight.data[P, 0] = 11111.0
    for p in range(12, 21):
        read(b2.attn, p - 11, p, DIG, E, 10.0 ** (21 - p), fallback=EQ)
    for m in range(1, 10):
        for n, offset, sign in ((2 * (m - 1), 1.0, 1.0), (2 * (m - 1) + 1, 0.0, -1.0)):
            b2.mlp.fc1.weight.data[n, P] = 1.0
            b2.mlp.fc1.weight.data[n, E] = -1.0
            b2.mlp.fc1.weight.data[n, TAU] = -float(m)
            b2.mlp.fc1.bias.data[n] = offset
            b2.mlp.fc2.weight.data[DOUT, n] = sign

    # Head: logit_c = c*d - c^2/2, tangents to the parabola d^2/2, so argmax_c = d.
    h = model.head.weight.data
    for c in range(10):
        h[c, DOUT] = float(c)
        h[c, BIAS] = -c * c / 2.0
    h[10, BIAS] = h[11, BIAS] = -1e3  # '*' and '=' are never emitted

    stress_test(model, task)


def stress_test(model, task, n: int = 3000) -> None:
    """Sanity check only (does not touch the weights): greedy decode on random and edge pairs."""
    g = torch.Generator().manual_seed(123)
    a = torch.randint(0, 100000, (n,), generator=g).tolist()
    b = torch.randint(0, 100000, (n,), generator=g).tolist()
    edge = [0, 1, 9, 10, 99, 100, 9999, 10000, 50000, 99990, 99999]
    pairs = list(zip(a, b)) + [(x, y) for x in edge for y in edge]
    ids = torch.tensor([task.encode(f"{x:05d}*{y:05d}=") for x, y in pairs])
    model.eval()
    with torch.no_grad():
        for _ in range(task.answer_len):
            next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    outputs = [task.decode(row[task.prompt_len:].tolist()) for row in ids]
    wrong = [(x, y, o) for (x, y), o in zip(pairs, outputs) if o != f"{x * y:010d}"]
    print(f"STRESS {len(pairs) - len(wrong)}/{len(pairs)} correct; first failures: {wrong[:5]}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e06_softmax_place_value_p"
model_description = "e05 with one head summing partial products into P via log-place-value scores; answer digits still use one hard head each."


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
