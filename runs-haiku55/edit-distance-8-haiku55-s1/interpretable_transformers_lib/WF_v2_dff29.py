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
# NOTE: LayerNorm is removed (identity) so that hand-written one-hot / thermometer
# codes stay exact in the residual stream.

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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer (no LayerNorm), vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 400,
        n_heads: int = 2,
        n_layers: int = 18,
        d_ff: int = 29,
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
#
# Task (edit-distance-8): prompt "AAAAAAAA,BBBBBBBB=" (18 tokens, letters a-d),
# answer = Levenshtein distance (single digit 0-8).
#
# Circuit: wavefront dynamic program for the Levenshtein DP table.
#   Positions: a_i at p=i-1 (i=1..8), ',' at p=8 (acts as column j=0),
#              b_j at p=8+j (j=1..8), '=' at p=17 (the readout position).
#   DP cell (i,j) lives at position p=8+j and is computed in block m=i+j.
#   Values are stored as thermometer codes [value >= k], k=1..8.
#   DP[i][j] = min(up+1, left+1, diag+cost) with
#       [min >= k] = [up>=k-1] & [left>=k-1] & ([diag>=k] | (cost & [diag>=k-1]))
#   which is a 1-hidden-layer ReLU unit set (exact on binary inputs).
#
# Residual layout (d_model = 400 = 2 heads x 200):
#   dims  0..17  one-hot position index (POS(p) = p)
#   dims 18..21  one-hot letter (TOK(t) = 18 + t, t = a,b,c,d)
#   block b (base 22 + 21*b):
#       CELL(b,k)  k=1..8   thermometer of DP cell computed in block b
#       CC(b)               substitution cost used by block b (computed in block b-1)
#       LEFT(b,k)           copy of CELL(b-1,k) from position p-1  -> cell (i, j-1)
#       AF(b,t)             letter a_{b-j} fetched from position b+8-p (used for CC(b+1))
#   Up neighbour (i-1, j) is read from the own position's CELL(b-1).
#   Diagonal neighbour (i-1, j-1) = LEFT(b-1) at own position (the same copy, one block earlier).
#
# Output: at p=17, LEFT(17,·) copies CELL(16,·) from p=16, i.e. DP[8][8].
# Logit of digit d is T_d - T_{d+1} (T_0 = 1 = sum of position one-hots).

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    assert model.d_model == 400 and model.n_heads == 2 and model.n_layers == 18
    assert model.pos_emb.num_embeddings >= 18

    S = 142.0    # attention sharpness: logit gap = S / sqrt(200) ~ 10
    BIG = 5.0    # gate strength for border positions
    dh = model.d_model // model.n_heads  # 200
    stoi = task.stoi

    def POS(p): return p
    def TOK(t): return 18 + t
    def blk(b): return 22 + 21 * b
    def CELL(b, k): return blk(b) + (k - 1)
    def CC(b): return blk(b) + 8
    def LEFT(b, k): return blk(b) + 9 + (k - 1)
    def DIAG(b, k): return LEFT(b - 1, k)   # diag(i-1,j-1) = left-copy of previous block
    def AF(b, t): return blk(b) + 17 + t
    def HB(h): return h * dh

    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        # ---- embeddings ----
        for p in range(18):
            model.pos_emb.weight[p, POS(p)] = 1.0
        for t, ch in enumerate("abcd"):
            model.token_emb.weight[stoi[ch], TOK(t)] = 1.0

        # ---- readout: digit d <-> T_d - T_{d+1}, with T_0 = 1 ----
        for d in range(9):
            row = stoi[str(d)]
            if d == 0:
                for p in range(18):
                    model.head.weight[row, POS(p)] = 1.0
            else:
                model.head.weight[row, LEFT(17, d)] = 1.0
            if d + 1 <= 8:
                model.head.weight[row, LEFT(17, d + 1)] = -1.0

        for b, block in enumerate(model.blocks):
            # ---- attention ----
            at = block.attn
            Wq, Wk, Wv, Wo = at.W_q.weight, at.W_k.weight, at.W_v.weight, at.W_o.weight
            # head 0 (LEFT): copy CELL(b-1) from position p-1 -> cell (i, j-1)
            if b >= 1:
                hb = HB(0)
                for p in range(1, 18):
                    Wq[hb + p - 1, POS(p)] = S
                for r in range(18):
                    Wk[hb + r, POS(r)] = 1.0
                for k in range(1, 9):
                    Wv[hb + k - 1, CELL(b - 1, k)] = 1.0
                    Wo[LEFT(b, k), hb + k - 1] = 1.0
            # head 1: fetch letter at key index b+8-p (a_{b-j}) into AF(b,.)
            hb = HB(1)
            for p in range(18):
                r = b + 8 - p
                if 0 <= r <= 17:
                    Wq[hb + r, POS(p)] = S
            for r in range(18):
                Wk[hb + r, POS(r)] = 1.0
            for t in range(4):
                Wv[hb + t, TOK(t)] = 1.0
                Wo[AF(b, t), hb + t] = 1.0

            # ---- MLP ----
            fc1, fc2 = block.mlp.fc1, block.mlp.fc2
            ctr = [0]

            def add_unit(inputs, bias, outputs):
                h = ctr[0]
                ctr[0] += 1
                for d, w in inputs.items():
                    fc1.weight[h, d] = w
                fc1.bias[h] = bias
                for d, w in outputs.items():
                    fc2.weight[d, h] = w

            def and_unit(dims, outputs, gates=()):
                inputs = {d: 1.0 for d in dims}
                for g in gates:
                    inputs[g] = -BIG
                add_unit(inputs, 1.0 - len(dims), outputs)

            # interior DP cells (i>=1, j>=1) computed in block b
            if b >= 2:
                gates = [POS(8)] + ([POS(8 + b)] if b <= 8 else [])
                for k in range(1, 9):
                    U = [CELL(b - 1, k - 1)] if k >= 2 else []
                    Lf = [LEFT(b, k - 1)] if k >= 2 else []
                    Dk1 = [DIAG(b, k - 1)] if k >= 2 else []
                    Dk = [DIAG(b, k)]
                    c = [CC(b)]
                    out = CELL(b, k)
                    and_unit(U + Lf + Dk, {out: 1.0}, gates)
                    and_unit(U + Lf + c + Dk1, {out: 1.0}, gates)
                    and_unit(U + Lf + c + Dk, {out: -1.0}, gates)

            # substitution cost for block b+1: c = 1 - sum_t [a_t & b_t]
            if 1 <= b <= 16:
                for t in range(4):
                    and_unit([AF(b, t), TOK(t)], {CC(b + 1): -1.0})
                fc2.bias[CC(b + 1)] = 1.0

            # border cells: (b,0) at position 8 and (0,b) at position 8+b, value b
            if 1 <= b <= 8:
                add_unit({POS(8): 1.0, POS(8 + b): 1.0}, 0.0,
                         {CELL(b, k): 1.0 for k in range(1, b + 1)})


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "WF_v2_dff29"
model_description = (
    "Wavefront v2 (S=142, BIG=5) with MLP width cut to 29 = exactly the number of hidden units "
    "the circuit uses per block."
)


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
