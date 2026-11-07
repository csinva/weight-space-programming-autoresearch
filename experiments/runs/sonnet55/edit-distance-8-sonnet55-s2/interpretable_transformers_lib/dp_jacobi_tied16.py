"""
Interpretable transformer for character-level sequence tasks.

Hand-written weights (no training) that compute Levenshtein distance of two
length-8 strings over {a,b,c,d}: prompt "s1,s2=" -> one digit.

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
# Architecture: no LayerNorm, explicit head dim, weight-tied DP block
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


D_MODEL = 224
N_DP = 16  # number of applications of the (weight-tied) DP-relaxation block


class SimpleTransformer(nn.Module):
    """Block `setup` (gather s1 chars, previous char, match bits), then block `dp`
    (one DP relaxation step) applied N_DP times with tied weights."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, D_MODEL
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.setup = Block(D_MODEL, 9, max_seq_len, 32)
        self.dp = Block(D_MODEL, 1, 72, 200)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        h = self.setup(h)
        for _ in range(N_DP):
            h = self.dp(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Weights
#
# Residual stream slots (one-hot / binary features):
#   TOK  token one-hot            POS  position one-hot       ONE constant 1
#   S1   s1[i] letter one-hot (8x4, gathered by 8 positional heads)
#   PC   previous token letter (= s2[r] for the row at position 9+r)
#   M    m_i = [s1[i] == s2[r]]
#   X    X[i][v] = [D[i][r] <= v]  thresholds of DP cell (i, r), i=0..8, v=0..7
#   P    copy of X from position p-1 (cell (i, r-1)), cleared by the MLP
# Position 9+r holds DP row r. Row 0 and column 0 are pinned via pos_emb.
# Each DP block does one Jacobi relaxation of
#   D[i][r] = min(D[i][r], D[i][r-1]+1, D[i-1][r]+1, D[i-1][r-1]+[s1_i != s2_r])
# in threshold form: new bit = [2*(left|up|diag-mismatch bits) + diag-match-AND + ... >= 2].
# The head reads D[8][8] from X[8][.] at position 17.
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    V, L, D = model.vocab_size, model.max_seq_len, model.d_model
    TOK, POS = 0, V
    ONE = POS + L
    S1 = ONE + 1
    PC = S1 + 32
    M = PC + 4
    X = M + 8
    P = X + 72
    assert P + 72 <= D
    letters = [task.stoi[c] for c in "abcd"]
    cell = lambda i, v: i * 8 + v

    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        for t in range(V):
            model.token_emb.weight[t, TOK + t] = 1.0
        pe = model.pos_emb.weight
        for p in range(L):
            pe[p, POS + p] = 1.0
            pe[p, ONE] = 1.0
        # pinned boundary: position 9 = row 0 (D[i][0] = i); position 9+r: D[0][r] = r
        for v in range(8):
            for i in range(9):
                pe[9, X + cell(i, v)] = 1.0 if i <= v else 0.0
            for r in range(1, 9):
                pe[9 + r, X + cell(0, v)] = 1.0 if r <= v else 0.0

        # ---- setup block ----
        a = model.setup.attn
        dh = a.d_head
        A = 30.0 * math.sqrt(dh)
        for h in range(9):
            for j in range(L):
                a.W_k.weight[h * dh + j, POS + j] = 1.0
            for l, tid in enumerate(letters):
                a.W_v.weight[h * dh + l, TOK + tid] = 1.0
        for h in range(8):               # head h looks at s1[h] (position h)
            a.W_q.weight[h * dh + h, ONE] = A
            for l in range(4):
                a.W_o.weight[S1 + 4 * h + l, h * dh + l] = 1.0
        for p in range(1, L):            # head 8 looks at previous token
            a.W_q.weight[8 * dh + p - 1, POS + p] = A
        for l in range(4):
            a.W_o.weight[PC + l, 8 * dh + l] = 1.0
        m = model.setup.mlp              # m_i = sum_l relu(S1[i][l] + PC[l] - 1)
        for i in range(8):
            for l in range(4):
                u = i * 4 + l
                m.fc1.weight[u, S1 + 4 * i + l] = 1.0
                m.fc1.weight[u, PC + l] = 1.0
                m.fc1.bias[u] = -1.0
                m.fc2.weight[M + i, u] = 1.0

        # ---- DP block ----
        a = model.dp.attn
        dh = a.d_head
        A = 30.0 * math.sqrt(dh)
        for j in range(L):
            a.W_k.weight[j, POS + j] = 1.0
        for p in range(1, L):
            a.W_q.weight[p - 1, POS + p] = A
        for c in range(72):
            a.W_v.weight[c, X + c] = 1.0
            a.W_o.weight[P + c, c] = 1.0
        m = model.dp.mlp
        w = m.fc1.weight
        for i in range(1, 9):
            for v in range(8):
                idx = (i - 1) * 8 + v
                u1, u2 = 2 * idx, 2 * idx + 1
                if v >= 1:
                    w[u1, P + cell(i, v - 1)] += 2        # left:  D[i][r-1] <= v-1
                    w[u1, X + cell(i - 1, v - 1)] += 2    # up:    D[i-1][r] <= v-1
                    w[u1, P + cell(i - 1, v - 1)] += 2    # diag (mismatch)
                w[u1, P + cell(i - 1, v)] += 1            # diag (match) = AND of this and M
                w[u1, M + i - 1] += 1
                w[u1, X + cell(i, v)] += -10              # already set -> no change
                w[u2] = w[u1]
                m.fc1.bias[u1], m.fc1.bias[u2] = -1.0, -2.0
                m.fc2.weight[X + cell(i, v), u1] = 1.0    # [s>=2] = relu(s-1) - relu(s-2)
                m.fc2.weight[X + cell(i, v), u2] = -1.0
        for c in range(72):                                # clear P after use
            w[128 + c, P + c] = 1.0
            m.fc2.weight[P + c, 128 + c] = -1.0

        # ---- head: answer k <=> X[8][k] - X[8][k-1] = 1 ----
        for k in range(9):
            tid = task.stoi[str(k)]
            if k <= 7:
                model.head.weight[tid, X + cell(8, k)] += 1.0
            else:
                model.head.weight[tid, ONE] += 1.0
            if k >= 1:
                model.head.weight[tid, X + cell(8, k - 1)] -= 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "dp_jacobi_tied16"
model_description = "Threshold-coded Levenshtein DP; row r at position 9+r; 1 setup block + 16 weight-tied Jacobi relaxation blocks; no LayerNorm."


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
    print(f"total_seconds:{time.time() - t0:.1f}s")
