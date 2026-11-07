"""
Interpretable transformer for gcd-three-digits (hand-compiled, never trained).

Ablation snapshot: N_STEPS = 10, TABLE_MAX = 63 (1-variable lookup tables cover 0..63 only).
See Stein21Lookup.py for the full algorithm.
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
# Architecture (no LayerNorm: the residual stream holds exact integer scalars)
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
    """Pre-residual block without LayerNorm: x <- x + attn(x); x <- x + mlp(x)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer whose depth and per-block MLP width come from the compiled program."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, n_heads: int,
                 d_ff_per_block: list[int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, f) for f in d_ff_per_block])
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

N_STEPS = 10           # Stein iterations (each = 2 blocks). 20 is enough for all inputs <= 999.
ATTN_SCORE = 40.0      # attention logit for the selected prompt position (weight ~ 1 - e^-40)
TABLE_MAX = 63         # largest input covered by the 1-variable lookup tables
USE_POWER_OF_TWO = True  # multiply the odd gcd by 2^min(v2 a, v2 b)
D_MODEL = 54
N_HEADS = 6            # gather heads; d_head = 9
D_HEAD = D_MODEL // N_HEADS

# residual-stream layout: one-hot position dims, constant dim, token-value dim, then scalar registers
POS_DIM = 0            # dims 0..15: position one-hot
CONST_DIM = 16         # always 1
TOK_DIM = 17           # digit value of the token (0 for ',' and '=')
_REG_NAMES = (["A", "B", "VA", "VB", "U", "V", "D", "K"]
              + [f"I{j}" for j in range(10)]
              + ["X", "HH", "HT", "HU", "DG"]
              + [f"L{c}" for c in range(10)])
REG = {name: 18 + i for i, name in enumerate(_REG_NAMES)}
assert max(REG.values()) < D_MODEL


def v2_of(s: int) -> int:
    """Exponent of 2 in s (0 for s = 0). Used only to fill a 1-variable lookup table."""
    if s == 0:
        return 0
    c = 0
    while s % 2 == 0:
        s //= 2
        c += 1
    return c


def odd_of(s: int) -> int:
    """s with all factors of 2 removed (0 for s = 0). Used only to fill a 1-variable table."""
    if s == 0:
        return 0
    while s % 2 == 0:
        s //= 2
    return s


V2_TABLE = [v2_of(s) for s in range(TABLE_MAX + 1)]
ODD_TABLE = [odd_of(s) for s in range(TABLE_MAX + 1)]


class BlockSpec:
    """Hidden units of one MLP. Each unit: (input weights over dims, bias, output coefficients)."""

    def __init__(self):
        self.units: list[tuple[dict, float, dict]] = []
        self.attn: list[tuple[int, str, float]] | None = None  # (key position, dst register, coeff) per head

    def relu(self, inw: dict, bias: float, outw: dict) -> None:
        self.units.append((dict(inw), float(bias), dict(outw)))

    def table(self, src: int, dst: int, table: list[int], overwrite: bool = False) -> None:
        """dst += f(src) where f is the integer table on 0..len(table)-1 (exact piecewise-linear).

        f(s) = T[0] + slope_0 * ReLU(s) + sum_{t>=1} (slope_t - slope_{t-1}) * ReLU(s - t).
        overwrite=True means dst := f(src), i.e. subtract the old dst value (dst >= 0).
        """
        S = len(table) - 1
        if table[0] != 0:
            self.relu({}, table[0], {dst: 1.0})
        slopes = [table[t + 1] - table[t] for t in range(S)]
        if slopes[0] != 0:
            self.relu({src: 1.0}, 0.0, {dst: float(slopes[0])})
        for t in range(1, S):
            c = slopes[t] - slopes[t - 1]
            if c != 0:
                self.relu({src: 1.0}, -float(t), {dst: float(c)})
        if overwrite:
            self.relu({dst: 1.0}, 0.0, {dst: -1.0})


def compile_program(n_steps: int) -> list[BlockSpec]:
    R = REG
    blocks: list[BlockSpec] = []

    # Block 0: gather digits of a and b from fixed prompt positions; odd/v2 tables.
    b = BlockSpec()
    b.attn = [(0, "A", 100.0), (1, "A", 10.0), (2, "A", 1.0),
              (4, "B", 100.0), (5, "B", 10.0), (6, "B", 1.0)]
    b.table(R["A"], R["VA"], V2_TABLE)
    b.table(R["A"], R["U"], ODD_TABLE)
    b.table(R["B"], R["VB"], V2_TABLE)
    b.table(R["B"], R["V"], ODD_TABLE)
    blocks.append(b)

    for step in range(n_steps):
        # Block A: u <- min(u,v), d <- |u-v|   (all reads use the pre-block values)
        ba = BlockSpec()
        ba.relu({R["U"]: 1.0, R["V"]: -1.0}, 0.0, {R["U"]: -1.0, R["D"]: 1.0})   # u - ReLU(u-v) = min
        ba.relu({R["V"]: 1.0, R["U"]: -1.0}, 0.0, {R["D"]: 1.0})                # + ReLU(v-u)
        ba.relu({R["D"]: 1.0}, 0.0, {R["D"]: -1.0})                              # remove old d
        if step == 0:
            # k = min(va, vb) = va - ReLU(va - vb)
            ba.relu({R["VA"]: 1.0}, 0.0, {R["K"]: 1.0})
            ba.relu({R["VA"]: 1.0, R["VB"]: -1.0}, 0.0, {R["K"]: -1.0})
        blocks.append(ba)

        # Block B: v <- odd(d)
        bb = BlockSpec()
        bb.table(R["D"], R["V"], ODD_TABLE, overwrite=True)
        if step == 0 and USE_POWER_OF_TWO:
            for j in range(10):
                bb.table(R["K"], R[f"I{j}"], [1 if s == j else 0 for s in range(10)])
        blocks.append(bb)

    # X = 2^k * (u + v), via sum_j 2^j * ReLU(G + 999*[k=j] - 999)
    bx = BlockSpec()
    if USE_POWER_OF_TWO:
        for j in range(10):
            bx.relu({R["U"]: 1.0, R["V"]: 1.0, R[f"I{j}"]: 999.0}, -999.0, {R["X"]: float(2 ** j)})
    else:
        bx.relu({R["U"]: 1.0, R["V"]: 1.0}, 0.0, {R["X"]: 1.0})
    blocks.append(bx)

    # Decimal digits of X
    bd = BlockSpec()
    bd.table(R["X"], R["HH"], [s // 100 for s in range(1000)])
    bd.table(R["X"], R["HT"], [(s // 10) % 10 for s in range(1000)])
    bd.table(R["X"], R["HU"], [s % 10 for s in range(1000)])
    blocks.append(bd)

    # Select the digit for this output position: position 7 -> hundreds, 8 -> tens, 9 -> units
    be = BlockSpec()
    for pos, name in ((7, "HH"), (8, "HT"), (9, "HU")):
        be.relu({R[name]: 1.0, POS_DIM + pos: 9.0}, -9.0, {R["DG"]: 1.0})
    blocks.append(be)

    # One-hot logits over digit values 0..9
    bl = BlockSpec()
    for c in range(10):
        bl.table(R["DG"], R[f"L{c}"], [1 if s == c else 0 for s in range(10)])
    blocks.append(bl)
    return blocks


def write_weights(model: SimpleTransformer, blocks: list[BlockSpec]) -> None:
    """Populate `model`'s parameters in-place. No training."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for c in range(10):
            model.token_emb.weight[c, TOK_DIM] = float(c)
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, POS_DIM + t] = 1.0
            model.pos_emb.weight[t, CONST_DIM] = 1.0
        for block, spec in zip(model.blocks, blocks):
            if spec.attn is not None:
                att = block.attn
                for h, (kpos, reg, coeff) in enumerate(spec.attn):
                    r0 = h * D_HEAD
                    att.W_q.weight[r0, CONST_DIM] = math.sqrt(D_HEAD)   # score = 1 * ATTN_SCORE / ...
                    att.W_k.weight[r0, POS_DIM + kpos] = ATTN_SCORE
                    att.W_v.weight[r0, TOK_DIM] = 1.0
                    att.W_o.weight[REG[reg], r0] += coeff
            for i, (inw, bias, outw) in enumerate(spec.units):
                for dim, w in inw.items():
                    block.mlp.fc1.weight[i, dim] = w
                block.mlp.fc1.bias[i] = bias
                for dim, c in outw.items():
                    block.mlp.fc2.weight[dim, i] += c
        for c in range(10):
            model.head.weight[c, REG[f"L{c}"]] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Stein10Tab063"
model_description = ("Ablation at N_STEPS=10: 1-variable lookup tables cover inputs 0..63 only "
                     "(TABLE_MAX=63; larger inputs are extrapolated). "
                     "attention gather, odd/v2 tables, 2^k times odd pair sum. No training.")


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
    blocks = compile_program(N_STEPS)
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff_per_block=[max(1, len(b.units)) for b in blocks],
    )
    write_weights(model, blocks)
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
