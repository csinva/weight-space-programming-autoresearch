"""
Interpretable transformer for character-level sequence tasks.

Hand-written (untrained) transformer for `edit-distance-8`: Levenshtein distance between two
8-letter strings over {a,b,c,d}, computed by running the DP table one row per pass of a single
shared block.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py --task edit-distance-8
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
# Residual layout (width 7, no LayerNorm):
#   0 COS, 1 SIN  : letter as a point on the unit ring (a,b,c,d at 90° steps)
#   2 W           : DP value W[l] = D[i][l] - l  (first-string tokens start at 16)
#   3 Y, 4 Y2     : countdown Y = position - pass and its square (frozen from the comma on)
#   5 F           : 1 on first-string positions (0..7)
#   6 ONE         : constant 1
COS, SIN, WD, YD, Y2D, FD, ONE = range(7)
D_MODEL = 7
N_PASSES = 9


class TwoScalarHeadAttention(nn.Module):
    """Two heads, each emitting one scalar.

    head 1: 1-dim query/key, attends to strictly earlier tokens (position 0 falls back to itself).
    head 2: 3-dim query/key, ordinary causal attention.
    The two scalars are fed to the MLP (not added to the residual).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.q1 = nn.Linear(d_model, 1, bias=False)
        self.k1 = nn.Linear(d_model, 1, bias=False)
        self.v1 = nn.Linear(d_model, 1, bias=False)
        self.q2 = nn.Linear(d_model, 3, bias=False)
        self.k2 = nn.Linear(d_model, 3, bias=False)
        self.v2 = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        strict = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
        strict[0, 0] = False  # position 0 has no earlier token: let it see itself
        s1 = self.q1(x) @ self.k1(x).transpose(-2, -1)
        s1 = s1.masked_fill(strict, float("-inf")).softmax(-1)
        o1 = s1 @ self.v1(x)
        s2 = self.q2(x) @ self.k2(x).transpose(-2, -1)
        s2 = s2.masked_fill(causal, float("-inf")).softmax(-1)
        o2 = s2 @ self.v2(x)
        return torch.cat([o1, o2], dim=-1)  # (B, T, 2): softmin S, match m


class MLP(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 3):
        super().__init__()
        self.attn = TwoScalarHeadAttention(d_model)
        self.mlp = MLP(d_model + 2, d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(x)
        return x + self.mlp(torch.cat([x, a], dim=-1))


class SimpleTransformer(nn.Module):
    """One shared block applied N_PASSES times (one DP row per pass)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = D_MODEL,
                 n_passes: int = N_PASSES, d_ff: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_passes = n_passes
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.block = Block(d_model, d_ff)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for _ in range(self.n_passes):
            h = self.block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form weights implementing the row-by-row Levenshtein DP.

    Pass p (0..7) processes first-string letter p (the one whose countdown Y is 0):
        W[l] <- min(W[l] + 1, softmin_{l' < l} W[l'] - match(a_p, b_l))
    which is exact because W = D - l is non-increasing along a row.  Pass 8 lets '=' pick up
    the soft minimum of the final row, i.e. W[8] = D - 8.
    """
    torch.manual_seed(0)
    stoi = task.stoi
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings ---
        te = model.token_emb.weight
        for i, ch in enumerate("abcd"):
            ang = i * math.pi / 2
            te[stoi[ch], COS] = round(math.cos(ang))
            te[stoi[ch], SIN] = round(math.sin(ang))
        te[:, ONE] = 1.0
        pe = model.pos_emb.weight
        for pos in range(model.max_seq_len):
            if pos < 8:                            # first string
                pe[pos, WD] = 16.0                 # never wins a minimum
                pe[pos, FD] = 1.0
            y = 0.0 if pos == 8 else float(pos)    # comma Y = 0; later tokens frozen at large Y
            pe[pos, YD] = y
            pe[pos, Y2D] = y * y

        att = model.block.attn
        # head 1: soft minimum of W over strictly earlier tokens: score -4*W_k, value W_k
        att.q1.weight[0, ONE] = 1.0
        att.k1.weight[0, WD] = -4.0
        att.v1.weight[0, WD] = 1.0
        # head 2: first-string letter with Y = 0 scores +8 (match) / -8 (mismatch) vs 0 for comma
        att.q2.weight[0, ONE] = 1.0
        att.k2.weight[0, Y2D] = -32.0
        att.k2.weight[0, FD] = -8.0
        att.q2.weight[1, COS] = 16.0
        att.q2.weight[2, SIN] = 16.0
        att.k2.weight[1, COS] = 1.0
        att.k2.weight[2, SIN] = 1.0
        att.v2.weight[0, FD] = 1.0                 # 1 on letters, 0 on comma -> soft match bit

        # MLP: input = [x (7), S, m]
        S_IN, M_IN = D_MODEL, D_MODEL + 1
        fc1, fc2 = model.block.mlp.fc1, model.block.mlp.fc2
        # u0 = relu(W + 1 - (S - m));  W <- W + 1 - u0 = min(W + 1, S - m)
        fc1.weight[0, WD] = 1.0
        fc1.weight[0, S_IN] = -1.0
        fc1.weight[0, M_IN] = 1.0
        fc1.bias[0] = 1.0
        fc2.weight[WD, 0] = -1.0
        fc2.bias[WD] = 1.0
        # u1 = relu(Y) on first-string tokens only (Y - 32 < 0 elsewhere)
        fc1.weight[1, YD] = 1.0
        fc1.weight[1, FD] = 32.0
        fc1.bias[1] = -32.0
        # u2 = relu(F): countdown step only for first-string tokens (frozen from the comma on)
        fc1.weight[2, FD] = 1.0
        # Y <- Y - F ;  Y2 <- Y2 - 2 relu(Y) + F  (= (Y-1)^2 while Y >= 0, stays >= 1 afterwards)
        fc2.weight[YD, 2] = -1.0
        fc2.weight[Y2D, 2] = 1.0
        fc2.weight[Y2D, 1] = -2.0

        # readout at '=': v = W + 8, digit logits 2d*v - d^2 (= argmax round(v))
        hd = model.head.weight
        hd[:, ONE] = -1000.0
        for d in range(9):
            t = stoi[str(d)]
            hd[t, WD] = 2.0 * d
            hd[t, ONE] = 2.0 * d * 8 - d * d


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "EditDistDP_softmin_v1"
model_description = ("Width-7 shared block x9, one Levenshtein DP row per pass: head1 softmin(-4W) over strictly "
                     "earlier tokens, head2 countdown-selected match bit (+-8 vs comma), MLP W<-min(W+1,S-m), "
                     "parabolic readout of W+8 at '='.")


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
