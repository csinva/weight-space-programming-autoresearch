"""
Hand-written transformer computing the Levenshtein distance of two 8-char strings.

Each block is one row of the DP table.  With W[l] tracking D[i][l] - l,
    W[l] <- min(W[l] + 1, min_{x<l} W[x] - match(i, l))
and the distance is 8 + min_l W[l].  The prefix min is a causal mask plus a sharp
softmax.  Tokens ',' and b hold DP columns 0..8; '=' accumulates the result.

Rules: NO training. All weights are written in closed form in write_weights().
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

# Hyperparameters of the circuit
D_MODEL, N_HEADS, N_LAYERS, D_FF = 12, 2, 9, 3
BETA = 10.0     # sharpness of the soft-min
GAMMA = 30.0    # sharpness of the letter-match decision
PEAK = 60.0     # curvature of the quadratic position peak selecting a_row
EXCLUDE = 1000.0  # score penalty on a token's own column
G = 1000.0      # base of the self-exclusion code
FAR = 100.0     # initial W of the 'a' tokens, which keeps them out of every min
CLEAR = 30.0    # offset letting one relu carry a signed linear signal


# ---------------------------------------------------------------------------
# Architecture (LayerNorms removed: registers hold raw integers)
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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS, d_ff: int = D_FF):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff)] * n_layers)  # one shared cell
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Hand-written weights
# ---------------------------------------------------------------------------

# Register layout of a token vector
ONE, X, X2 = 0, 1, 2          # constant 1, position x, x^2
GF, GI = 3, 4                 # G^column and G^-column, for self-exclusion
L1, L2 = 5, 6                 # letter on a ring: a=(1,0) b=(0,1) c=(-1,0) d=(0,-1)
COMMA = 7                     # flag for ','
W, R = 8, 9                   # DP value, (left prefix-min - match)
I, I2 = 10, 11                # row counter i and i^2, carried by every token
COMMA_POS = 8


def write_block(block: Block, dh: int) -> None:
    """The shared cell. It reads a_i; at i=8 it lands on ',' and is the read-out pass."""
    B, C = 0, dh                           # first dim of each head
    s = math.sqrt(dh)                      # cancels the score normaliser
    at = block.attn
    q, k, v, o = at.W_q.weight, at.W_k.weight, at.W_v.weight, at.W_o.weight

    # Head B: prefix min of W, my own column pushed out by -EXCLUDE * G^(col_x - col_me).
    q[B, ONE] = 1.0
    k[B, W] = -BETA * s
    q[B + 1, GI] = 1.0
    k[B + 1, GF] = -EXCLUDE * s
    v[B, W] = 1.0
    o[R, B] = 1.0

    # Head C: a_row scores GAMMA on a letter match and 0 otherwise; ',' scores GAMMA/2.
    # Score = GAMMA*(letter_x . letter_me) - PEAK*(x-i)^2 + bonus at ','.
    # (x-i)^2 = x^2 - 2xi + i^2 is bilinear in (1, i, i^2) on the query side
    # and (x^2, x, 1) on the key side; the ',' bonus cancels it at x=8.
    q[C, L1], k[C, L1] = 1.0, GAMMA * s
    q[C + 1, L2], k[C + 1, L2] = 1.0, GAMMA * s
    q[C + 2, ONE], q[C + 3, I], q[C + 4, I2] = 1.0, 1.0, 1.0
    k[C + 2, X2], k[C + 3, X], k[C + 4, ONE] = -PEAK * s, 2 * PEAK * s, -PEAK * s
    k[C + 2, COMMA] = (PEAK * COMMA_POS ** 2 + GAMMA / 2) * s
    k[C + 3, COMMA] = -2 * PEAK * COMMA_POS * s
    k[C + 4, COMMA] = PEAK * s
    v[C, ONE], v[C, COMMA] = 1.0, -1.0     # 1 from a letter, 0 from ','
    o[R, C] = -1.0

    # MLP: W <- min(W + 1, R) = W + 1 - relu(W + 1 - R), then clear R.
    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    fc1.weight[0, W], fc1.weight[0, R], fc1.bias[0] = 1.0, -1.0, 1.0
    fc2.weight[W, 0], fc2.bias[W] = -1.0, 1.0
    fc1.weight[1, R], fc1.bias[1] = 1.0, CLEAR
    fc2.weight[R, 1], fc2.bias[R] = -1.0, CLEAR
    # i <- i + 1 and i^2 <- i^2 + (2i + 1)
    fc2.bias[I] = 1.0
    fc1.weight[2, I], fc1.bias[2] = 2.0, 1.0
    fc2.weight[I2, 2] = 1.0


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    pe, te = model.pos_emb.weight, model.token_emb.weight
    for x in range(18):
        pe[x, ONE], pe[x, X], pe[x, X2] = 1.0, float(x), float(x * x)
        if x < COMMA_POS:
            pe[x, W] = FAR
        else:                              # ',' is column 0 and '=' is column 9
            pe[x, GF], pe[x, GI] = G ** (x - COMMA_POS), G ** (COMMA_POS - x)
    pe[COMMA_POS, COMMA] = 1.0
    for ch, (a, b) in zip("abcd", [(1, 0), (0, 1), (-1, 0), (0, -1)]):
        te[task.stoi[ch], L1], te[task.stoi[ch], L2] = float(a), float(b)

    write_block(model.blocks[0], D_MODEL // N_HEADS)

    # Read-out at '=': distance v = W + 8, logit_d = 2dv - d^2 (a parabola peaking at v).
    model.head.weight[:, ONE] = -EXCLUDE
    for d in range(9):
        t = task.stoi[str(d)]
        model.head.weight[t, W] = 2.0 * d
        model.head.weight[t, ONE] = 16.0 * d - d * d

    self_check(model, task)


@torch.no_grad()
def self_check(model, task, n: int = 5000) -> None:
    """Accuracy on a fresh, larger sample (evaluation only, nothing is fitted)."""
    ex = task.generate_examples(n, seed=987654)
    ids = torch.tensor([task.encode(e.prompt) for e in ex])
    pred = model(ids)[:, -1].argmax(-1).tolist()
    ok = sum(task.itos[p] == e.answer for p, e in zip(pred, ex))
    print(f"SELF_CHECK {ok}/{n}")


model_shorthand_name = "TiedCell_d12"
model_description = "All 9 blocks share one recurrent cell; a row counter (i, i^2) in every token steers the quadratic peak along a; d=12, 2 heads, ff=3."


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
