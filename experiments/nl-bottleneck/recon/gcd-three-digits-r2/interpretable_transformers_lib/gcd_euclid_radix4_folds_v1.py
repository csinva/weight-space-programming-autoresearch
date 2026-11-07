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
# Residual state per position: [VAL, A, B, X] plus a fixed one-hot position
# code P (11 dims, positions 0..10).  Layers:
#   1. parser attention (6 heads, logits ~40): A = 100a0+10a1+a2, B likewise.
#      The comma (pos 3) attends to itself -> A=B=0 there ("zero slot").
#   2. Euclid: 12 tied rounds of 4 residual ReLU folds (s = 64, 16, 4, 1).
#      fold_s: A <- dist(A, nearest multiple of s*B) using 4 knots at
#      m*s*B/2 (m=1..4) with slopes -2,+2,-2,+2; s=1 fold also swaps (A,B).
#   3. readout attention (3 heads) writes X = (A+B)/100, /10, /1 (mix with
#      the zero slot) minus 100x / 10x previously emitted digits.
#   4. head: logit_d = -(X - d - 0.495)^2 (the -X^2 term is dropped: common).

NPOS = 11
POS_COMMA = 3


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16, **kw):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tok_val = nn.Embedding(vocab_size, 1)                 # VAL scalar
        pos = torch.zeros(max_seq_len, NPOS)
        pos[:NPOS, :NPOS] = torch.eye(NPOS)
        self.register_buffer("pos_code", pos)                      # fixed one-hot
        # parser attention: 6 heads, score = P_i^T Q_h P_j ; value = v_h * VAL
        self.parse_q = nn.Parameter(torch.zeros(6, NPOS, NPOS))
        self.parse_v = nn.Parameter(torch.zeros(6))
        self.parse_o = nn.Parameter(torch.zeros(6, 2))             # -> (A, B)
        # 4 distinct fold MLPs (tied across 12 rounds), inputs (A, B)
        self.fold_fc1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(4, 2)) for _ in range(3)] + [nn.Parameter(torch.zeros(6, 2))])
        self.fold_fc2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(2, 4)) for _ in range(3)] + [nn.Parameter(torch.zeros(2, 6))])
        self.n_rounds = 12
        # readout attention: 3 heads, value = w_h . (VAL, A, B), output -> X
        self.read_q = nn.Parameter(torch.zeros(3, NPOS, NPOS))
        self.read_v = nn.Parameter(torch.zeros(3, 3))
        # unembedding on [X, P]
        self.head = nn.Linear(1 + NPOS, vocab_size, bias=False)

    @staticmethod
    def _attend(P, Q, V):
        # P: (B,T,NPOS), Q: (H,NPOS,NPOS), V: (B,H,T)
        T = P.shape[1]
        scores = torch.einsum("bip,hpq,bjq->bhij", P, Q, P)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=P.device), 1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
        return torch.einsum("bhij,bhj->bhi", attn, V)              # (B,H,T)

    def forward(self, ids):
        Bsz, T = ids.shape
        P = self.pos_code[:T][None].expand(Bsz, T, NPOS)
        val = self.tok_val(ids)[..., 0]                            # (B,T)
        # --- parser ---
        Vp = self.parse_v[None, :, None] * val[:, None, :]
        heads = self._attend(P, self.parse_q, Vp)                  # (B,6,T)
        AB = torch.einsum("bht,hc->btc", heads, self.parse_o)      # (B,T,2)
        # --- Euclid: residual ReLU folds (tied across rounds) ---
        for _ in range(self.n_rounds):
            for W1, W2 in zip(self.fold_fc1, self.fold_fc2):
                AB = AB + F.relu(AB @ W1.T) @ W2.T
        # --- readout ---
        feats = torch.stack([val, AB[..., 0], AB[..., 1]], -1)     # (B,T,3)
        Vr = torch.einsum("btc,hc->bht", feats, self.read_v)
        X = self._attend(P, self.read_q, Vr).sum(1)                # (B,T)
        return self.head(torch.cat([X[..., None], P], -1))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    L = 40.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # token embedding: digit value, ',' and '=' -> 0
        for i, c in enumerate(task.vocab):
            model.tok_val.weight[i, 0] = float(int(c)) if c.isdigit() else 0.0
        # parser: head h reads one operand digit, scaled by place value
        targets = [0, 1, 2, 4, 5, 6]
        places = [100.0, 10.0, 1.0, 100.0, 10.0, 1.0]
        for h, (tp, pv) in enumerate(zip(targets, places)):
            for i in range(NPOS):
                tgt = POS_COMMA if i == POS_COMMA else tp
                model.parse_q[h, i, tgt] = L
            model.parse_v[h] = pv
            model.parse_o[h, 0 if h < 3 else 1] = 1.0
        # folds
        slopes = [-2.0, 2.0, -2.0, 2.0]
        for k, s in enumerate([64.0, 16.0, 4.0, 1.0]):
            W1, W2 = model.fold_fc1[k], model.fold_fc2[k]
            for m in range(1, 5):                    # knot ReLU(A - m*s*B/2)
                W1[m - 1, 0] = 1.0
                W1[m - 1, 1] = -m * s / 2.0
            if s != 1.0:
                for m in range(4):
                    W2[0, m] = slopes[m]             # dA = fold(A) - A
            else:
                W1[4, 0] = 1.0                       # ReLU(A) = A  (A >= 0)
                W1[5, 1] = 1.0                       # ReLU(B) = B
                W2[0, 5] = 1.0; W2[0, 4] = -1.0      # new A = B
                for m in range(4):                   # new B = fold(A)
                    W2[1, m] = slopes[m]
                W2[1, 4] = 1.0; W2[1, 5] = -1.0
        # readout head 0: mix self with zero slot -> (A+B) * {1/100, 1/10, 1}
        model.read_v[0, 1] = 1.0; model.read_v[0, 2] = 1.0
        for i in range(NPOS):
            model.read_q[0, i, i] = L
        model.read_q[0, 7, POS_COMMA] = L + math.log(99.0)
        model.read_q[0, 8, POS_COMMA] = L + math.log(9.0)
        # readout head 1: self, -10 * VAL (the digit just emitted)
        model.read_v[1, 0] = -10.0
        for i in range(NPOS):
            model.read_q[1, i, i] = L
        # readout head 2: pos 9 reads pos 8 (hundreds digit) * -100; else zero slot
        model.read_v[2, 0] = -100.0
        for i in range(NPOS):
            model.read_q[2, i, POS_COMMA] = L
        model.read_q[2, 9, POS_COMMA] = 0.0
        model.read_q[2, 9, 8] = L
        # head: logit_d = 2(d+.495) X - (d+.495)^2 ; constant via position code
        W = model.head.weight
        for i, c in enumerate(task.vocab):
            if c.isdigit():
                cd = int(c) + 0.495
                W[i, 0] = 2.0 * cd
                W[i, 1:] = -cd * cd
            else:
                W[i, 1:] = -1e4


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "gcd_euclid_radix4_folds_v1"
model_description = ("Parser attn (6 heads, logit 40) builds exact A,B; 12 tied rounds of 4 residual ReLU "
                     "triangle-wave folds (s=64,16,4,1; s=1 swaps) run Euclid to a (g,0)/(0,g) 2-cycle; "
                     "readout attn mixes with comma zero-slot for X=(A+B)/100,/10,/1 minus emitted digits; "
                     "head -(X-d-0.495)^2.")


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
