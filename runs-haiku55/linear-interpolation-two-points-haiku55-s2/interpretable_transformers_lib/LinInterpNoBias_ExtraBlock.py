"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task `linear-interpolation-two-points`:
prompt "x1,y1;x2,y2;xq=" (12 chars) -> answer two digits (y at xq).
The points lie on a line y = m*x + b with m in 1..5, b in 0..9.

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
# Residual-stream layout (used by write_weights; the architecture is generic)
# ---------------------------------------------------------------------------

TOK0 = 0              # 10 dims: one-hot of a digit token (zero for ',', ';', '=')
POS0 = 10             # 13 dims: one-hot of positions 0..12 (only these are ever read)
FIELDS = ["X1", "Y1T", "Y1U", "X2", "Y2T", "Y2U", "XQ"]
FIELD_POS = {"X1": 0, "Y1T": 2, "Y1U": 3, "X2": 5, "Y2T": 7, "Y2U": 8, "XQ": 10}
G0 = 23               # 7 blocks x 10 dims: prompt digits copied here by attention
YV1_0 = 93            # 55 dims: one-hot of point 1's y-value (tens*10 + units)
YV2_0 = 148           # 55 dims: one-hot of point 2's y-value
K1_0 = 203            # 50 dims: hypothesis h=(m,b) matches point 1
K2_0 = 253            # 50 dims: hypothesis h=(m,b) matches point 2
VAL0 = 303            # 55 dims: one-hot of the answer value v = y_q in 0..54
OUTT_0 = 358          # 10 dims: one-hot tens digit of the answer (read at position 11)
OUTU_0 = 368          # 10 dims: one-hot units digit of the answer (read at position 12)
CONST = 378           # 1 dim: constant 1 at every position; MLP thresholds are weights on it (no biases)
D_MODEL = 379
N_HEADS = 7           # one gather head per prompt field
D_HEAD = 11           # keys only for positions 0..10 (the only positions any head reads)
D_FFS = [1, 110, 1000, 500, 16, 1]   # per-block MLP widths; the last block is an all-zero identity
N_LAYERS = 5
HYPS = [(m, b) for m in range(1, 6) for b in range(10)]   # 50 candidate lines


def G_OF(name: str) -> int:
    return G0 + 10 * FIELDS.index(name)


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
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
    """Residual block without LayerNorm (keeps the circuit exactly linear + ReLU)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 16,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_head: int = D_HEAD,
        n_layers: int = N_LAYERS,
        d_ff: int = 1000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_head, w) for w in D_FFS])
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Circuit (value-first encoding, hypothesis enumeration, digit extraction; 5 blocks):
      B0 attn : gather the 7 prompt digits into residual blocks G_f (fixed prompt positions).
      B1 MLP  : per point, AND(tens digit, units digit) -> one-hot of the point's y-value YV.
      B2 MLP  : per point, K[h] = OR_a AND(x == a, YV == m*a+b): does line h pass through the point?
      B3 MLP  : VAL[m*q+b] += AND(K1[h], K2[h], xq == q): the answer value as a one-hot.
      B4 MLP  : tens digit (gated by position 11) and units digit (gated by position 12).
      head    : logits = tens block + units block (digit tokens only).
    """
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: digit tokens -> TOK block; positions -> POS one-hot + CONST.
        for d in range(10):
            model.token_emb.weight[d, TOK0 + d] = 1.0
        for j in range(13):
            model.pos_emb.weight[j, POS0 + j] = 1.0
            model.pos_emb.weight[j, CONST] = 1.0

        # B0 attention: head hi attends (score 30) to prompt position FIELD_POS[f]
        # and copies that position's digit one-hot into G block f.
        attn = model.blocks[0].attn
        S = 30.0 * math.sqrt(D_HEAD)   # attention score margin at the target position
        for hi, f in enumerate(FIELDS):
            base = hi * D_HEAD
            # POS one-hot sums to 1 at every position, so this gives a constant query.
            for j in range(13):
                attn.W_q.weight[base + FIELD_POS[f], POS0 + j] = S
            for j in range(D_HEAD):
                attn.W_k.weight[base + j, POS0 + j] = 1.0
            for k in range(10):
                attn.W_v.weight[base + k, TOK0 + k] = 1.0
                attn.W_o.weight[G_OF(f) + k, base + k] = 1.0

        # B1 MLP: YV1[v] = AND(Y1T == v//10, Y1U == v%10), v in 0..54; same for point 2.
        mlp = model.blocks[1].mlp
        u = 0
        for v in range(55):
            for tf, uf, out0 in (("Y1T", "Y1U", YV1_0), ("Y2T", "Y2U", YV2_0)):
                mlp.fc1.weight[u, G_OF(tf) + v // 10] = 1.0
                mlp.fc1.weight[u, G_OF(uf) + v % 10] = 1.0
                mlp.fc1.weight[u, CONST] = -1.0
                mlp.fc2.weight[out0 + v, u] = 1.0
                u += 1

        # B2 MLP: K1[h] = OR_a AND(X1 == a, YV1 == m*a+b); same for point 2 with X2, YV2.
        mlp = model.blocks[2].mlp
        u = 0
        for h, (m, b) in enumerate(HYPS):
            for a in range(10):
                v = m * a + b
                for xf, yv, out0 in (("X1", YV1_0, K1_0), ("X2", YV2_0, K2_0)):
                    mlp.fc1.weight[u, G_OF(xf) + a] = 1.0
                    mlp.fc1.weight[u, yv + v] = 1.0
                    mlp.fc1.weight[u, CONST] = -1.0
                    mlp.fc2.weight[out0 + h, u] = 1.0
                    u += 1

        # B3 MLP: VAL[m*q+b] += AND(K1[h], K2[h], XQ[q]).
        mlp = model.blocks[3].mlp
        u = 0
        for h, (m, b) in enumerate(HYPS):
            for q in range(10):
                v = m * q + b
                mlp.fc1.weight[u, K1_0 + h] = 1.0
                mlp.fc1.weight[u, K2_0 + h] = 1.0
                mlp.fc1.weight[u, G_OF("XQ") + q] = 1.0
                mlp.fc1.weight[u, CONST] = -2.0
                mlp.fc2.weight[VAL0 + v, u] = 1.0
                u += 1

        # B4 MLP: digit extraction from the value one-hot, gated by position.
        mlp = model.blocks[4].mlp
        u = 0
        for d in range(6):                       # tens digits of values 0..54
            for v in range(55):
                if v // 10 == d:
                    mlp.fc1.weight[u, VAL0 + v] = 1.0
            mlp.fc1.weight[u, POS0 + 11] = 1.0
            mlp.fc1.weight[u, CONST] = -1.0
            mlp.fc2.weight[OUTT_0 + d, u] = 1.0
            u += 1
        for d in range(10):                      # units digits
            for v in range(55):
                if v % 10 == d:
                    mlp.fc1.weight[u, VAL0 + v] = 1.0
            mlp.fc1.weight[u, POS0 + 12] = 1.0
            mlp.fc1.weight[u, CONST] = -1.0
            mlp.fc2.weight[OUTU_0 + d, u] = 1.0
            u += 1

        # Head: digit logit = tens-block + units-block value (only one is active per position).
        for d in range(10):
            model.head.weight[d, OUTT_0 + d] = 1.0
            model.head.weight[d, OUTU_0 + d] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "LinInterpNoBias_ExtraBlock"
model_description = ("NoBias 5-block circuit (canonical margin 30, causal mask, no noise) plus one extra, all-zero "
                     "residual block (6 blocks): depth-padding invariance check.")


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
