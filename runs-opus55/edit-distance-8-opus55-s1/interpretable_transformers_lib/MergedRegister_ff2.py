"""
Hand-written transformer computing the Levenshtein distance of two 8-char strings.

One block = one row of the DP table.  With E[i][j] = D[i][j] - j the recurrence is
    W[l]    = min(E_prev[l] + 1, E_prev[l-1] - match(i, l))
    E[l]    = min_{k <= l} W[k]          <- causal mask + sharp softmax = prefix min
Tokens at positions 8..16 (',' and b) are the DP columns 0..8; '=' reads the result.

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
D_MODEL, N_HEADS, N_LAYERS, D_FF = 26, 2, 9, 2
BETA = 10.0    # sharpness of the soft-min
GAMMA = 30.0   # sharpness of the letter-match decision
BIG = 1000.0    # score penalty that removes a token from an attention head
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
# Hand-written weights
# ---------------------------------------------------------------------------

# Register layout of a token vector
POS = 0                       # 18 dims: one-hot position
CH = 18                       # 4 dims: one-hot letter
W, RL = 22, 23                # candidate, (left prefix-min - match)
COMMA, EQ = 8, 17             # positions; columns are positions 8..16


def write_block(block: Block, row: int | None, dh: int) -> None:
    """Block for DP row `row` (1-based, None for the final read-out pass)."""
    B, C = 0, dh                           # first dim of each head
    s = math.sqrt(dh)                      # cancels the score normaliser
    at = block.attn
    q, k, v, o = at.W_q.weight, at.W_k.weight, at.W_v.weight, at.W_o.weight
    outside = list(range(8)) + [EQ]        # tokens that are not DP columns

    # Head B: prefix min of W over the columns strictly left of me.
    for p in range(18):
        q[B, POS + p] = 1.0
    k[B, W] = -BETA * s
    for p in outside:
        k[B, POS + p] = -BIG * s
    for c, p in enumerate(range(COMMA, EQ)):
        q[B + 1 + c, POS + p] = -1.0
        k[B + 1 + c, POS + p] = BIG * s
    v[B, W] = 1.0
    o[RL, B] = 1.0

    # Head C: does a_row equal my letter?  It chooses between a_row (scoring an
    # extra GAMMA only on a match) and ',' (scoring GAMMA/2).
    # ',' and '=' have no letter and always fall back to the ',' sentinel.
    for c in range(4):
        q[C + c, CH + c] = 1.0
        q[C + c, POS + COMMA] = 1.0
        q[C + c, POS + EQ] = 1.0
        k[C + c, CH + c] = GAMMA * s
        k[C + c, POS + COMMA] = (2 * GAMMA + GAMMA / 2) * s
        if row is not None:
            k[C + c, POS + row - 1] = 2 * GAMMA * s
    for p in range(8):
        v[C, POS + p] = 1.0
    o[RL, C] = -1.0                        # RL now holds (left min) - match

    # MLP: W <- min(W + 1, RL) = W + 1 - relu(W + 1 - RL)
    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    fc1.weight[0, W], fc1.weight[0, RL] = 1.0, -1.0
    fc1.weight[0, POS + COMMA] = -BIG      # ',' has no left column
    fc1.bias[0] = 1.0
    fc2.weight[W, 0] = -1.0
    fc2.bias[W] = 1.0
    # n1 clears RL
    fc1.weight[1, RL] = 1.0
    fc1.bias[1] = CLEAR
    fc2.weight[RL, 1] = -1.0
    fc2.bias[RL] = CLEAR


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    for p in range(model.pos_emb.num_embeddings):
        if p < 18:
            model.pos_emb.weight[p, POS + p] = 1.0
    for c, ch in enumerate("abcd"):
        model.token_emb.weight[task.stoi[ch], CH + c] = 1.0

    dh = D_MODEL // N_HEADS
    for i, block in enumerate(model.blocks):
        write_block(block, i + 1 if i < 8 else None, dh)

    # Read-out at '=': distance v = W + 8, logit_d = 2dv - d^2 (a parabola peaking at v).
    model.head.weight[:, POS + EQ] = -BIG
    for d in range(9):
        t = task.stoi[str(d)]
        model.head.weight[t, W] = 2.0 * d
        model.head.weight[t, POS + EQ] = 16.0 * d - d * d

    self_check(model, task)


@torch.no_grad()
def self_check(model, task, n: int = 5000) -> None:
    """Accuracy on a fresh, larger sample (evaluation only, nothing is fitted)."""
    ex = task.generate_examples(n, seed=987654)
    ids = torch.tensor([task.encode(e.prompt) for e in ex])
    pred = model(ids)[:, -1].argmax(-1).tolist()
    ok = sum(task.itos[p] == e.answer for p, e in zip(pred, ex))
    print(f"SELF_CHECK {ok}/{n}")


model_shorthand_name = "MergedRegister_ff2"
model_description = "W_o subtracts the match from the left-min into one register, so the MLP needs only 2 neurons; BETA=10, GAMMA=30, d=26."


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
