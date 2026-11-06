"""
Interpretable transformer for character-level sequence tasks.

Hand-written GCD transformer for `gcd-three-digits`:
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
# Residual state per position: [VAL, A, B, X]  (+ a fixed one-hot position code
# that is a buffer, not a parameter).
#
#   1. Embedding: VAL = digit value (',' and '=' -> 0).
#   2. Parser attention (6 heads, logits ~40): head h reads one operand digit
#      and writes place_value * digit into A or B.  At the comma every head
#      attends to the comma itself (VAL=0), so A=B=0 there ("zero slot").
#   3. Euclid: 12 tied rounds of 4 residual ReLU MLP folds (s = 64,16,4,1) on
#      (A,B).  fold_s(A) = A - 2r(A - sB/2) + 2r(A - sB) - 2r(A - 3sB/2)
#      + 2r(A - 2sB): triangle wave = distance to nearest multiple of sB
#      (valid for A <= 2sB).  The s=1 fold also swaps: (A,B) -> (B, fold).
#      Result 2-cycles between (g,0) and (0,g), so A+B = g.
#   4. Readout attention (3 heads): X = (A+B)/100, /10, /1 by mixing self with
#      the zero slot, minus 100x / 10x the previously emitted digits.
#   5. Head: logit_d = -(X - d - 0.495)^2 (dropping the d-independent -X^2).

N_PARSE_HEADS = 6
N_READ_HEADS = 3
N_ROUNDS = 12
FOLD_SCALES = (64, 16, 4, 1)
COMMA_POS = 3
ANS_POS = (7, 8, 9)          # positions whose next-token prediction is a gcd digit


def _causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
    return scores.masked_fill(mask, float("-inf")).softmax(dim=-1)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16, **_):
        super().__init__()
        T = max_seq_len
        self.vocab_size, self.max_seq_len = vocab_size, T
        self.register_buffer("pos_code", torch.eye(T))            # fixed, not a parameter

        # token embedding: one scalar (VAL)
        self.tok_val = nn.Parameter(torch.zeros(vocab_size))

        # parser attention: q (shared across heads, 2-dim), per-head k (2-dim)
        self.parse_q = nn.Parameter(torch.zeros(T, 2))
        self.parse_k = nn.Parameter(torch.zeros(N_PARSE_HEADS, T, 2))
        self.parse_out = nn.Parameter(torch.zeros(N_PARSE_HEADS, 2))   # VAL -> (A, B)

        # Euclid folds (tied across rounds). fc1: (A,B) -> knots; fc2: knots -> (dA,dB)
        self.fold_w1 = nn.ParameterList()
        self.fold_b1 = nn.ParameterList()
        self.fold_w2 = nn.ParameterList()
        for s in FOLD_SCALES:
            n_hidden = 6 if s == 1 else 4
            self.fold_w1.append(nn.Parameter(torch.zeros(n_hidden, 2)))
            self.fold_w2.append(nn.Parameter(torch.zeros(2, n_hidden)))

        # readout attention: shared key features (4-dim), per-head query
        self.read_k = nn.Parameter(torch.zeros(T, 4))
        self.read_q = nn.Parameter(torch.zeros(N_READ_HEADS, T, 4))
        self.read_v = nn.Parameter(torch.zeros(N_READ_HEADS, 3))       # from [VAL, A, B]
        self.read_out = nn.Parameter(torch.zeros(N_READ_HEADS))        # -> X

        # unembedding on [X, 1]
        self.head = nn.Parameter(torch.zeros(vocab_size, 2))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        Bsz, T = ids.shape
        P = self.pos_code[:T, :T]                      # (T, T) one-hot rows
        val = self.tok_val[ids]                        # (B, T)

        # --- parser attention ---
        q = self.parse_q[:T]                           # (T, 2)
        k = self.parse_k[:, :T]                        # (H, T, 2)
        scores = torch.einsum("id,hjd->hij", q, k)     # (H, T, T)
        attn = _causal_softmax(scores)                 # (H, T, T)
        heads = torch.einsum("hij,bj->bhi", attn, val) # (B, H, T)
        ab = torch.einsum("bhi,hc->bic", heads, self.parse_out)   # (B, T, 2)
        a, b = ab[..., 0], ab[..., 1]

        # --- Euclid: tied residual ReLU MLP folds ---
        for _ in range(N_ROUNDS):
            for w1, w2 in zip(self.fold_w1, self.fold_w2):
                x = torch.stack([a, b], dim=-1)
                d = F.relu(x @ w1.T) @ w2.T
                a, b = a + d[..., 0], b + d[..., 1]

        # --- readout attention ---
        kr = self.read_k[:T]                            # (T, 4)
        qr = self.read_q[:, :T]                         # (H, T, 4)
        scores = torch.einsum("hid,jd->hij", qr, kr)
        attn = _causal_softmax(scores)
        state = torch.stack([val, a, b], dim=-1)        # (B, T, 3)
        v = torch.einsum("btc,hc->bht", state, self.read_v)
        heads = torch.einsum("hij,bhj->bhi", attn, v)
        X = torch.einsum("bhi,h->bi", heads, self.read_out)

        feats = torch.stack([X, torch.ones_like(X)], dim=-1)    # (B, T, 2)
        return feats @ self.head.T


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    T = model.max_seq_len
    L = 40.0

    # embedding
    for tok, i in task.stoi.items():
        model.tok_val[i] = float(tok) if tok.isdigit() else 0.0

    # parser: query feature 0 = "read operand" (all non-comma positions),
    #         query feature 1 = "attend comma" (comma position only)
    for i in range(T):
        if i == COMMA_POS:
            model.parse_q[i, 1] = L
        else:
            model.parse_q[i, 0] = L
    src = [(0, 0, 100.0), (1, 0, 10.0), (2, 0, 1.0),     # (key position, A/B, place value)
           (4, 1, 100.0), (5, 1, 10.0), (6, 1, 1.0)]
    for h, (p, c, pv) in enumerate(src):
        model.parse_k[h, p, 0] = 1.0
        model.parse_k[h, COMMA_POS, 1] = 1.0
        model.parse_out[h, c] = pv

    # Euclid folds. hidden j: relu(A - m*s*B/2), m=1..4, output coeffs -2,+2,-2,+2 on A
    for f, s in enumerate(FOLD_SCALES):
        w1, w2 = model.fold_w1[f], model.fold_w2[f]
        for j, (m, c) in enumerate(zip((1, 2, 3, 4), (-2.0, 2.0, -2.0, 2.0))):
            w1[j, 0] = 1.0
            w1[j, 1] = -m * s / 2.0
            if s == 1:
                # new B = fold(A) = A + sum c*relu(...) ; new A = B
                w2[1, j] = c
            else:
                w2[0, j] = c
        if s == 1:
            w1[4, 0] = 1.0   # relu(A) = A
            w1[5, 1] = 1.0   # relu(B) = B
            w2[0, 4], w2[0, 5] = -1.0, 1.0   # dA = B - A
            w2[1, 4], w2[1, 5] = 1.0, -1.0   # dB = A - B + knots

    # readout: key features = [is comma, is pos7, is pos8, is pos9]
    model.read_k[COMMA_POS, 0] = 1.0
    for f, p in enumerate(ANS_POS, start=1):
        model.read_k[p, f] = 1.0
    l99, l9 = math.log(99.0), math.log(9.0)
    q = model.read_q
    # head 0: G = A+B mixed with zero slot -> /100 at pos7, /10 at pos8, /1 at pos9
    q[0, 7, 0], q[0, 7, 1] = L + l99, L
    q[0, 8, 0], q[0, 8, 2] = L + l9, L
    q[0, 9, 3] = L
    # head 1: first emitted digit (token at pos 8): weight 0.1 at pos8, 1 at pos9; x(-100)
    q[1, 7, 0] = L
    q[1, 8, 0], q[1, 8, 2] = L + l9, L
    q[1, 9, 2] = L
    # head 2: second emitted digit (token at pos 9): weight 1 at pos9; x(-10)
    q[2, 7, 0] = L
    q[2, 8, 0] = L
    q[2, 9, 3] = L
    model.read_v[0, 1], model.read_v[0, 2] = 1.0, 1.0
    model.read_v[1, 0] = 1.0
    model.read_v[2, 0] = 1.0
    model.read_out[0], model.read_out[1], model.read_out[2] = 1.0, -100.0, -10.0

    # head: -(X - d - 0.495)^2  ->  2(d+.495) X - (d+.495)^2
    for tok, i in task.stoi.items():
        if tok.isdigit():
            c = int(tok) + 0.495
            model.head[i, 0] = 2.0 * c
            model.head[i, 1] = -c * c
        else:
            model.head[i, 1] = -1e6


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "gcd_euclid_radix4_fold_v1"
model_description = ("6-head parser attention (logit 40) builds exact A,B; comma self-attends as zero slot; "
                     "12 tied rounds of 4 residual ReLU triangle-wave folds (s=64,16,4,1; s=1 swaps) give A+B=g; "
                     "readout mixes with zero slot minus 100x/10x emitted digits; head -(X-d-0.495)^2.")


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
