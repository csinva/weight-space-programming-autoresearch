"""
Interpretable transformer for 5-digit multiplication: "50494*99346=" -> "5016376...".
The model is never trained: write_weights() sets every parameter in closed form.
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
    """LayerNorm-free block: registers in x are real numbers and must pass through unchanged."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_mlp: bool = True):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff) if use_mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 20,
        n_heads: int = 2,
        d_ff: int = 4,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, use_mlp=True),   # A*b_j at each multiplier digit
            Block(d_model, n_heads, d_ff, use_mlp=False),  # weighted sum, minus emitted digits
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

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

# Hidden-state channels ("registers")
DIG, B0, B1, B2, B3, ONE, A, PP, X = range(9)                   # values
KA, KP, ISB, NUL, SP, NP, ISOUT, QV, KV, NV = range(9, 19)      # attention address codes

LN10 = math.log(10.0)
OFF = -1000.0   # log-weight of "never attend here"
M = 1e5         # big-M gate; A < M
ALPHA = 1e5     # attention scale for the partial-product sum
BETA = 2e9      # attention scale for the emitted-digit sum


def place(t: int) -> int:
    """Decimal place of the digit predicted at sequence position t (t=11 -> 10^9)."""
    return max(20 - t, 0)


def write_weights(model: SimpleTransformer, task) -> None:
    model.double()  # the registers hold integers up to 1e10
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Tokens: numeric digit value + its 4 binary bits. '*' and '=' are all zero.
        for d in range(10):
            model.token_emb.weight[d, DIG] = d
            for b, ch in enumerate((B0, B1, B2, B3)):
                model.token_emb.weight[d, ch] = (d >> b) & 1

        # Positions: address codes used by attention to form exact weighted sums.
        pe = model.pos_emb.weight
        pe[:, ONE] = 1.0
        for t in range(pe.shape[0]):
            pe[t, KA] = (4 - t) * LN10 if t < 5 else OFF              # A = sum 10^(4-i) a_i
            pe[t, KP] = (10 - t) * LN10 if 6 <= t <= 10 else (0.0 if t == 11 else OFF)
            pe[t, ISB] = float(6 <= t <= 10)
            pe[t, NUL] = float(t == 11)                               # '=' is the zero-valued sink
            pe[t, ISOUT] = float(t >= 12)
            pe[t, KV] = (1 - t) * LN10 if t >= 12 else (0.0 if t == 11 else OFF)
            if 11 <= t <= 20:  # position 21 is never an input
                p = place(t)
                pe[t, SP] = -p * LN10 - math.log(ALPHA)
                pe[t, NP] = math.log1p(-11111.0 * 10.0 ** (-p) / ALPHA)
                pe[t, QV] = t * LN10 - math.log(BETA)
                pe[t, NV] = math.log1p(-sum(10.0 ** (t + 1 - i) for i in range(12, t + 1)) / BETA)
                pe[t, X] = 0.5 * 10.0 ** (-p)                         # tie-break: floor, not round

        sq = math.sqrt(model.blocks[0].attn.d_head)
        dh = model.blocks[0].attn.d_head

        # Block 1 attention, head 0: every position reads the multiplicand A / 11111.
        at = model.blocks[0].attn
        at.W_q.weight[0, ONE] = sq
        at.W_k.weight[0, KA] = 1.0
        at.W_v.weight[0, DIG] = 1.0
        at.W_o.weight[A, 0] = 11111.0

        # Block 1 MLP: A*b = sum_k 2^k ReLU(A + M bit_k - M). Position-wise, so each
        # multiplier digit gets its own partial product, using the same 4 neurons.
        mlp = model.blocks[0].mlp
        for b, ch in enumerate((B0, B1, B2, B3)):
            mlp.fc1.weight[b, A] = 1.0
            mlp.fc1.weight[b, ch] = M
            mlp.fc1.bias[b] = -M
            mlp.fc2.weight[PP, b] = 2.0 ** b

        # Block 2 attention. Head 0 reads P * 10^-p, head 1 reads emitted-prefix * 10^-p.
        at = model.blocks[1].attn
        at.W_q.weight[0, ONE] = sq;  at.W_k.weight[0, KP] = 1.0
        at.W_q.weight[1, SP] = sq;   at.W_k.weight[1, ISB] = 1.0
        at.W_q.weight[2, NP] = sq;   at.W_k.weight[2, NUL] = 1.0
        at.W_v.weight[0, PP] = 1.0
        at.W_o.weight[X, 0] = ALPHA
        at.W_q.weight[dh + 0, QV] = sq;   at.W_k.weight[dh + 0, ISOUT] = 1.0
        at.W_q.weight[dh + 1, ONE] = sq;  at.W_k.weight[dh + 1, KV] = 1.0
        at.W_q.weight[dh + 2, NV] = sq;   at.W_k.weight[dh + 2, NUL] = 1.0
        at.W_v.weight[dh, DIG] = 1.0
        at.W_o.weight[X, dh] = -BETA

        # Tropical decoder: argmax_k (k x - k(k+1)/2) = floor(x). No neurons needed.
        for k in range(10):
            model.head.weight[k, X] = k
            model.head.weight[k, ONE] = -k * (k + 1) / 2
        model.head.weight[10, ONE] = -1e6
        model.head.weight[11, ONE] = -1e6


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ShiftAddTropical_v1"
model_description = ("No LayerNorm, float64. Block1: attention reads A, 4-neuron gated MLP makes A*b_j at each "
                     "multiplier digit. Block2: attention sums partial products minus emitted digits, scaled to [0,10); "
                     "tropical linear head gives floor(x).")


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
