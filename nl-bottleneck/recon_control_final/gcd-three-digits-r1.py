"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) transformer for `gcd-three-digits`:
prompt "AAA,BBB=" -> answer "GGG".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task gcd-three-digits
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
#
# Algorithm (gcd via the divisor lattice):
#   L1 attn : 6 heads (d_head=1) copy the 6 operand digits into dedicated dims.
#   L1 MLP  : ReLU ramps t_j = relu(a - j) give an exact one-hot of a (2nd difference);
#             fc2 sums the one-hot over multiples of d  ->  D_a[d] = [d | a]  (same for b).
#   L2 MLP  : C[d] = relu(D_a[d] + D_b[d] - 1) = [d is a common divisor].
#   L3 MLP  : common divisors are exactly the divisors of g, so
#             C[d] - sum_{k>=2} C[k d] is 1 iff d == g (<= 0 otherwise).  Gated by
#             position (answer slot 0/1/2) the unit writes digit_slot(d) into an output dim.
#   head    : logits = scale * output-digit dims.
# No LayerNorm (removed so that values stay exact).

N = 999


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 0, d_head: int = 1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if n_heads > 0 else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Layout:
    """Named residual-stream subspaces."""

    def __init__(self, max_seq_len):
        i = 0

        def take(n):
            nonlocal i
            s = i
            i += n
            return s

        self.ONE = take(1)
        self.VAL = take(1)
        self.POS = take(max_seq_len)
        self.DIG = take(6)          # a0 a1 a2 b0 b1 b2
        self.DA = take(N)           # [d | a], index d-1
        self.DB = take(N)           # [d | b]
        self.C = take(N)            # [d | a and d | b]
        self.OUT = take(10)         # output digit one-hot
        self.d_model = i


class SimpleTransformer(nn.Module):
    """3-layer causal transformer (no LayerNorm), vocab/seq-len configured from the task."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.L = Layout(max_seq_len)
        d = self.L.d_model
        self.d_model = d
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        self.blocks = nn.ModuleList([
            Block(d, 2 * (N + 3), n_heads=6, d_head=1),
            Block(d, N),
            Block(d, 3 * N),
        ])
        self.head = nn.Linear(d, vocab_size, bias=False)

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

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    L = model.L
    for p in model.parameters():
        p.zero_()

    # --- embeddings: constant 1, digit value, one-hot position
    for c, i in task.stoi.items():
        model.token_emb.weight[i, L.ONE] = 1.0
        if c.isdigit():
            model.token_emb.weight[i, L.VAL] = float(c)
    for p in range(model.max_seq_len):
        model.pos_emb.weight[p, L.POS + p] = 1.0

    # --- L1 attention: head h attends (sharply) to operand digit position src_pos[h]
    src_pos = [0, 1, 2, 4, 5, 6]
    att = model.blocks[0].attn
    SHARP = 100.0
    for h, sp in enumerate(src_pos):
        att.W_q.weight[h, L.ONE] = SHARP
        att.W_k.weight[h, L.POS + sp] = 1.0
        att.W_v.weight[h, L.VAL] = 1.0
        att.W_o.weight[L.DIG + h, h] = 1.0

    # --- L1 MLP: ramps t_j = relu(x - j), j = -1..N+1, for x = a and x = b
    m1 = model.blocks[0].mlp
    R = N + 3
    for o, base in enumerate([L.DA, L.DB]):
        for r in range(R):
            j = r - 1
            u = o * R + r
            for k, w in enumerate([100.0, 10.0, 1.0]):
                m1.fc1.weight[u, L.DIG + 3 * o + k] = w
            m1.fc1.bias[u] = -float(j)
        # onehot[m] = t_{m-1} - 2 t_m + t_{m+1};  D[d] = sum_{d | m} onehot[m]
        for d in range(1, N + 1):
            for m in range(d, N + 1, d):
                for jj, w in [(m - 1, 1.0), (m, -2.0), (m + 1, 1.0)]:
                    m1.fc2.weight[base + d - 1, o * R + jj + 1] += w

    # --- L2 MLP: C[d] = relu(D_a[d] + D_b[d] - 1)
    m2 = model.blocks[1].mlp
    for d in range(1, N + 1):
        m2.fc1.weight[d - 1, L.DA + d - 1] = 1.0
        m2.fc1.weight[d - 1, L.DB + d - 1] = 1.0
        m2.fc1.bias[d - 1] = -1.0
        m2.fc2.weight[L.C + d - 1, d - 1] = 1.0

    # --- L3 MLP: H[s,d] = relu(C[d] - sum_{k>=2} C[kd] + pos_{P-1+s} - 1) -> OUT[digit_s(d)]
    m3 = model.blocks[2].mlp
    P = task.prompt_len  # first answer digit is predicted at position P-1 ('=')
    for s in range(3):
        for d in range(1, N + 1):
            u = s * N + d - 1
            m3.fc1.weight[u, L.C + d - 1] = 1.0
            for kd in range(2 * d, N + 1, d):
                m3.fc1.weight[u, L.C + kd - 1] = -1.0
            m3.fc1.weight[u, L.POS + P - 1 + s] = 1.0
            m3.fc1.bias[u] = -1.0
            digit = (d // [100, 10, 1][s]) % 10
            m3.fc2.weight[L.OUT + digit, u] = 1.0

    # --- unembedding: digit token logit = 10 * OUT[digit]
    for c, i in task.stoi.items():
        if c.isdigit():
            model.head.weight[i, L.OUT + int(c)] = 10.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "gcd_divisor_lattice_v1"
model_description = ("Attn copies digits; MLP1 builds exact one-hot of a,b via ReLU ramps and sums over multiples "
                     "to get [d|a],[d|b]; MLP2 ANDs to common divisors; MLP3 picks the common divisor with no "
                     "common-divisor multiple (=gcd), position-gated digit output.")


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
