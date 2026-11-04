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
        self.n_heads = n_heads  # each head is one scalar query, key and value
        self.W_q = nn.Linear(d_model, n_heads, bias=False)
        self.W_k = nn.Linear(d_model, n_heads, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, 1).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, 1).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, 1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
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
        d_model: int = 6,
        n_heads: int = 1,
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
        d_ffs = (9, d_ff)  # layer 1 multiplies with 9 gates; layer 2 needs a step pair per m
        self.blocks = nn.ModuleList([Block(d_model, n_heads, f) for f in d_ffs[:n_layers]])
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
      Y      BIG*d + a': the token digit in the high part, a' in the low part.
             a' is a + 1 on b's cells and 0 on answer cells
      KA     layer 1 key: log place value on a's cells, far negative on '*'.
             Also the query (-KA/30): +1 on b's cells, -1 on answer cells.
             Constant 30 on every cell that predicts, so KA/30 acts as the bias
      KR     layer 2 key: log place value on b's cells, '=' (ballast), answer cells
      ZETA   TAU / Z, the place value being predicted on the remainder's scale
      V      d*a on b's cells, -d on answer cells: signed contribution to the remainder
      O      next digit in the units place, with R / (1000 Z) below it as a small fraction
    Layer 1: one head reads a; MLP multiplies it by the cell's digit.
    Layer 2: one head sums V, giving the remainder; MLP compares it with m*ZETA.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.zeros_(p)

    Y, KA, KR, ZETA, V, O = range(6)
    BIG = 1e6     # exceeds any value a gated neuron must suppress
    SHRINK = 1e-3  # keeps R/Z well below 1 so it cannot disturb the digit in the same slot
    SCALE = 3e9 / SHRINK  # turns O - m*ZETA back into a step of at least 1 per integer of R
    LOG10 = math.log(10.0)

    te, pe = model.token_emb.weight.data, model.pos_emb.weight.data
    for d in range(10):
        te[d, Y] = BIG * d
        te[d, V] = -BIG * d * (d - 1) / 2 - d
    z = 11111.0 + 1e9  # softmax normalizer: b's weights plus the ballast on '='
    for t in range(pe.shape[0]):
        pe[t, KA] = (4 - t) * LOG10 if t <= 4 else {5: -120.0}.get(t, -30.0 if t <= 10 else 30.0)
        pe[t, KR] = -60.0
        pe[t, Y] = 1.0 if 6 <= t <= 10 else 0.0
        if 6 <= t <= 10:
            pe[t, KR] = (10 - t) * LOG10
        if 11 <= t <= 20:
            pe[t, KR] = (21 - t if t > 11 else 9) * LOG10
            if t > 11:
                z += 10.0 ** (21 - t)
            pe[t, ZETA] = SHRINK * 10.0 ** (20 - t) / z

    b1, b2 = model.blocks

    # Layer 1: scores log(10^(4-i)) make softmax weights 10^(4-i)/11111 on a's cells, so
    # the head's output is a/11111. On answer cells the query flips and lands on '*' (DIG 0).
    b1.attn.W_q.weight.data[0, KA] = -1.0 / 30.0
    b1.attn.W_k.weight.data[0, KA] = 1.0
    b1.attn.W_v.weight.data[0, Y] = 1.0 / BIG
    b1.attn.W_o.weight.data[Y, 0] = 11111.0
    # With d integer and 0 <= A < BIG: relu(A + BIG*(d-m)) - relu(BIG*(d-m)) = (d >= m) * A.
    # The second terms total BIG*d(d-1)/2, seeded into V with a minus sign by the token
    # embedding, as is -d: V = d*(a+1) - d = d*a on b's cells, and 0 - d = -d on answer cells.
    for m in range(1, 10):
        n = m - 1
        b1.mlp.fc1.weight.data[n, Y] = 1.0
        b1.mlp.fc1.bias.data[n] = -m * BIG
        b1.mlp.fc2.weight.data[V, n] = 1.0

    # Layer 2: the same log-place-value head sums V over b's cells and the emitted digits,
    # so O = (P - emitted) / Z = R / Z. The digit is sum_m step(O - m*ZETA), where
    # step(x) = relu(SCALE*x + 1) - relu(SCALE*x) on the integer grid of R.
    b2.attn.W_q.weight.data[0, KA] = 1.0 / 30.0
    b2.attn.W_k.weight.data[0, KR] = 1.0
    b2.attn.W_v.weight.data[0, V] = 1.0
    b2.attn.W_o.weight.data[O, 0] = SHRINK
    for m in range(1, 10):
        for n, offset, sign in ((2 * (m - 1), 1.0, 1.0), (2 * (m - 1) + 1, 0.0, -1.0)):
            b2.mlp.fc1.weight.data[n, O] = SCALE
            b2.mlp.fc1.weight.data[n, ZETA] = -m * SCALE
            b2.mlp.fc1.bias.data[n] = offset
            b2.mlp.fc2.weight.data[O, n] = sign

    # Head: logit_c = c*d - c^2/2, tangents to the parabola d^2/2, so argmax_c = d.
    h = model.head.weight.data
    for c in range(10):
        h[c, O] = float(c)
        h[c, KA] = -c * c / 60.0
    h[10, KA] = h[11, KA] = -100.0  # '*' and '=' are never emitted

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
model_shorthand_name = "e14_float32_negative_control"
model_description = "Negative control: e13 weights evaluated in float32. Tests whether float64 is really needed for a remainder of about 10^10."


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
