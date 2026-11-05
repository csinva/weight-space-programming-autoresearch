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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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
        self.final_ln = nn.LayerNorm(d_model)
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
    """Left-to-right fold v_l = op_l(v_{l-1}, b_l) over "b0 o1 b1 o2 b2 ... o4 b4 =".

    Residual layout (d_model=64): tok one-hot 0-5 ('0','1','&','|','^','='),
    pos one-hot 6-16, then one 9-dim block per layer l=1..4:
      O_l (3: op &,|,^), B_l (2: bit), P_l (2: previous value), V_l (2: new value).
    Layer l (l<=3) acts at position 2l; layer 4 acts at the '=' position 9.
    Heads: 0 fetches the operator, 1 the previous value, 2 the bit.
    MLP_l: 12 neurons = AND(op, bit, prev, position) lookup -> writes V_l.
    """
    E, G, SV, U, M = 1.0, 10.0, SCALE_V, UNIT, 20.0
    H = model.d_model // model.n_heads
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        for i in range(6):
            model.token_emb.weight[i, i] = E
        for p in range(11):
            model.pos_emb.weight[p, 6 + p] = E
        for l in range(1, 5):
            blk = model.blocks[l - 1]
            O, B, P, V = 17, 20, 22, 24 + 2 * (l - 1)
            q = 9 if l == 4 else 2 * l
            keys = [7, 6, 8] if l == 4 else [q - 1, q - 2, q]
            src_prev = 0 if l == 1 else 24 + 2 * (l - 2)
            at = blk.attn
            nulls = [0, 1 if l == 1 else 0, 1]   # keys whose read value is 0 (idle queries park here)
            for h, kp in enumerate(keys):
                at.W_q.weight[H * h + kp, 6 + q] = G
                for p_ in range(11):
                    if p_ != q:
                        at.W_q.weight[H * h + nulls[h], 6 + p_] = G
                for j in range(11):
                    at.W_k.weight[H * h + j, 6 + j] = 1.0
            for j in range(3):
                at.W_v.weight[0 * H + j, 2 + j] = SV
                at.W_o.weight[O + j, 0 * H + j] = 1.0
            for j in range(2):
                at.W_v.weight[1 * H + j, src_prev + j] = SV if l == 1 else SV_STATE
                at.W_o.weight[P + j, 1 * H + j] = 1.0
                at.W_v.weight[2 * H + j, j] = SV
                at.W_o.weight[B + j, 2 * H + j] = 1.0
            n = 0
            for op in range(3):
                for b in range(2):
                    for v in range(2):
                        res = [b & v, b | v, b ^ v][op]
                        blk.mlp.fc1.weight[n, O + op] = 1.0
                        blk.mlp.fc1.weight[n, B + b] = 1.0
                        blk.mlp.fc1.weight[n, P + v] = 1.0
                        blk.mlp.fc1.bias[n] = -2.5 * U
                        blk.mlp.fc2.weight[V + res, n] = M / (0.5 * U)
                        n += 1
        model.head.weight[0, 30] = 1.0
        model.head.weight[1, 31] = 1.0


DEBUG = False
SCALE_V = 0.242      # token one-hot after LN1 has amplitude ~5.75 -> copy with 1/5.75
SV_STATE = 0.176  # running-value dims (amplitude 8, dominate the norm) -> LN1 gives ~7.9
UNIT = 2.35  # post-LN2 amplitude of an active feature

model_shorthand_name = "FoldChain4L_final_d33_seed1_n500"
model_description = "4-layer left-to-right fold, d_model=33, 3 heads, d_ff=12 (22k params): shared op/bit/prev slots, per-layer 2-dim running value V_l; idle queries park on null keys; 12-neuron (op,bit,prev) AND-lookup MLP."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, n_layers=4, d_ff=12, d_model=33, n_heads=3)
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
