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
    """Multi-head causal attention; head dim independent of d_model."""
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
    """Residual block without LayerNorm (keeps exact integer arithmetic). n_heads=0 -> MLP-only."""
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if n_heads > 0 else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


# Residual-stream layout
N = 8
TOK0 = 0                     # token one-hot (15)
POS0 = 15                    # position one-hot (32 slots)
CONST = POS0 + 32            # constant 1
G0 = CONST + 1               # gathered letters: 16 slots x 4
M0 = G0 + 64                 # match matrix M[i][j], i,j in 1..8
D0 = M0 + 64                 # DP table D[i][j], i,j in 0..8 (81)
Y0 = D0 + 81                 # Y[i][j] = min(D[i-1][j], D[i][j-1]) + 1
D_MODEL = Y0 + 64
D_FF = 256
N_DIAG = 2 * N - 1           # anti-diagonals i+j = 2..16


def m_idx(i, j): return M0 + (i - 1) * N + (j - 1)
def d_idx(i, j): return D0 + i * (N + 1) + j
def y_idx(i, j): return Y0 + (i - 1) * N + (j - 1)


class SimpleTransformer(nn.Module):
    """Block 0: 16-head gather attention + match MLP; then 2 MLP-only blocks per DP anti-diagonal."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        assert max_seq_len <= 32
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        blocks = [Block(D_MODEL, 16, 4, D_FF)]
        blocks += [Block(D_MODEL, 0, 1, D_FF) for _ in range(2 * N_DIAG)]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

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
    for p in model.parameters():
        p.zero_()
    stoi = task.stoi
    letters = "abcd"

    # Embeddings: token one-hot, position one-hot, constant 1.
    for t in range(task.vocab_size):
        model.token_emb.weight[t, TOK0 + t] = 1.0
    for p in range(model.max_seq_len):
        model.pos_emb.weight[p, POS0 + p] = 1.0
        model.pos_emb.weight[p, CONST] = 1.0

    # ---- Block 0 attention: head h fetches the letter at string position p_h.
    # slots 0..7 -> first string (positions 0..7), 8..15 -> second string (positions 9..16)
    att = model.blocks[0].attn
    dh = att.d_head
    BIG = 200.0 * math.sqrt(dh)
    for h in range(16):
        p_h = h if h < 8 else h + 1
        att.W_q.weight[h * dh, CONST] = 1.0
        att.W_k.weight[h * dh, POS0 + p_h] = BIG
        for c, ch in enumerate(letters):
            att.W_v.weight[h * dh + c, TOK0 + stoi[ch]] = 1.0
            att.W_o.weight[G0 + 4 * h + c, h * dh + c] = 1.0

    # ---- Block 0 MLP: M[i][j] = sum_c relu(A_i[c] + B_j[c] - 1); fc2 bias writes DP boundary.
    mlp = model.blocks[0].mlp
    u = 0
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            for c in range(4):
                mlp.fc1.weight[u, G0 + 4 * (i - 1) + c] = 1.0
                mlp.fc1.weight[u, G0 + 4 * (8 + j - 1) + c] = 1.0
                mlp.fc1.bias[u] = -1.0
                mlp.fc2.weight[m_idx(i, j), u] = 1.0
                u += 1
    for k in range(N + 1):            # D[0][k] = k, D[k][0] = k
        mlp.fc2.bias[d_idx(0, k)] = float(k)
        mlp.fc2.bias[d_idx(k, 0)] = float(k)

    # ---- DP: for anti-diagonal s = i+j, layer A computes Y, layer B computes D.
    for di, s in enumerate(range(2, 2 * N + 1)):
        cells = [(i, s - i) for i in range(1, N + 1) if 1 <= s - i <= N]
        A = model.blocks[1 + 2 * di].mlp
        Bm = model.blocks[2 + 2 * di].mlp
        u = 0
        for (i, j) in cells:
            # Y = min(up, left) + 1 = relu(up + 1) - relu(up - left)
            A.fc1.weight[u, d_idx(i - 1, j)] = 1.0
            A.fc1.bias[u] = 1.0
            A.fc2.weight[y_idx(i, j), u] = 1.0
            u += 1
            A.fc1.weight[u, d_idx(i - 1, j)] = 1.0
            A.fc1.weight[u, d_idx(i, j - 1)] = -1.0
            A.fc2.weight[y_idx(i, j), u] = -1.0
            u += 1
        u = 0
        for (i, j) in cells:
            # Z = diag + 1 - M;  D = min(Z, Y) = relu(Z) - relu(Z - Y)
            Bm.fc1.weight[u, d_idx(i - 1, j - 1)] = 1.0
            Bm.fc1.weight[u, m_idx(i, j)] = -1.0
            Bm.fc1.bias[u] = 1.0
            Bm.fc2.weight[d_idx(i, j), u] = 1.0
            u += 1
            Bm.fc1.weight[u, d_idx(i - 1, j - 1)] = 1.0
            Bm.fc1.weight[u, m_idx(i, j)] = -1.0
            Bm.fc1.weight[u, y_idx(i, j)] = -1.0
            Bm.fc1.bias[u] = 1.0
            Bm.fc2.weight[d_idx(i, j), u] = -1.0
            u += 1

    # ---- Unembedding: logit_d = 2 d D - d^2 (= -(D-d)^2 + const); non-digits very low.
    dfin = d_idx(N, N)
    for t in range(task.vocab_size):
        ch = task.itos[t]
        if ch.isdigit():
            d = float(ch)
            model.head.weight[t, dfin] = 2.0 * d
            model.head.weight[t, CONST] = -d * d
        else:
            model.head.weight[t, CONST] = -1000.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ExactDPGatherMLP"
model_description = "16-head attention gathers both strings to '=', MLP builds 8x8 match matrix, 30 LN-free MLP layers run the Levenshtein DP per anti-diagonal with exact ReLU mins; quadratic-logit readout."


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
