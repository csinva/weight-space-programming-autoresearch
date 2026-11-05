"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task `edit-distance-8`:
prompt "ddacddcd,dbdbcdaa=" (two length-8 strings over {a,b,c,d}) -> answer "5"
(the Levenshtein distance, a single digit 0..8).

Construction (no training; every weight is written in closed form):

  Layer 0, attention: 16 heads, each head attends sharply to ONE prompt position
    (head h -> position p_h), copying that character one-hot into a "slot" of the
    residual stream at the final position.
  Layer 0, MLP: computes the 64 match bits m[i][j] = [a_i == b_j] with
    relu(a_i[c] + b_j[c] - 1), summed over c. It also writes the boundary
    row/column of the DP table and sets all interior DP cells to "infinity" (8).
  Layers 1..15, MLP only: each layer is one Jacobi step of the edit-distance DP
    D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(1-m[i][j])) on all 64
    interior cells. Values are stored as thermometer bits t_k = [D >= k], k=1..8.
    In thermometer form, min is AND, +1 is a shift of k, and the diagonal term
    is a mux on m. Each cell/threshold uses 3 hidden units: two compute the AND
    cases, one passes the old bit through so the residual can be overwritten.
    The DP needs T(8,8) = 15 steps.
  Head: digit d logit = [D>=d] - [D>=d+1], which is 1 exactly for the true value.

LayerNorm is removed everywhere so that the residual arithmetic stays exact.

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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_attn: bool = True):
        super().__init__()
        self.ln1 = nn.Identity()  # LayerNorm removed: keeps residual arithmetic exact
        self.attn = CausalSelfAttention(d_model, n_heads) if use_attn else None
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """16-block causal transformer. Only block 0 has attention (the gather step)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 19,
        d_model: int = 816,
        n_heads: int = 16,
        n_layers: int = 16,
        d_ff: int = 1536,
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
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, use_attn=(i == 0)) for i in range(n_layers)]
        )
        self.final_ln = nn.Identity()
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

# Residual-stream channel layout (D = 816 >= 810 used, padded to a multiple of 16 heads)
TOK0 = 0        # token one-hot, channels 0..14
POS0 = 15       # position one-hot for positions 0..17, channels 15..32
CONST = 33      # constant 1 (set on every token)
SLOT0 = 34      # 16 gathered slots x 4 chars, channels 34..97
M0 = 98         # 64 match bits m[i][j], channels 98..161
DP0 = 162       # 81 DP cells x 8 thermometer bits, channels 162..809
FINAL = 17      # index of the '=' position (prediction position)
N_HEADS = 16
N_DP_LAYERS = 11  # Jacobi steps needed: optimal path has <= 12 interior cells (cost <= 8)
INF_INIT = 5      # initial value of interior DP cells (the 'infinity' the DP starts from)


def head_pos(h: int) -> int:
    """Head h attends to prompt position p_h (a-chars at 0..7, b-chars at 9..16)."""
    return h if h < 8 else h + 1


def dp_chan(i: int, j: int, k: int) -> int:
    """Channel of thermometer bit [D[i][j] >= k], k in 1..8."""
    return DP0 + 8 * (9 * i + j) + (k - 1)


def bit_chan(i: int, j: int, k: int) -> int:
    """Channel for [D[i][j] >= k]; k == 0 is always true, so it is the CONST channel."""
    return CONST if k == 0 else dp_chan(i, j, k)


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    stoi = task.stoi
    dh = model.d_model // model.n_heads
    beta = 30.0 * math.sqrt(dh)  # 1/sqrt(dh) scaling makes effective score 30 for the target
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings
        for t in range(model.vocab_size):
            model.token_emb.weight[t, TOK0 + t] = 1.0
        model.token_emb.weight[:, CONST] = 1.0
        for p in range(FINAL + 1):
            model.pos_emb.weight[p, POS0 + p] = 1.0

        # Block 0, attention: gather each prompt character into its slot.
        attn = model.blocks[0].attn
        for h in range(N_HEADS):
            base = h * dh
            tgt = head_pos(h)
            attn.W_q.weight[base + tgt, POS0 + FINAL] = beta      # query selects target pos
            for p in range(FINAL + 1):
                attn.W_k.weight[base + p, POS0 + p] = 1.0          # key = position one-hot
            for c, letter in enumerate("abcd"):
                attn.W_v.weight[base + c, TOK0 + stoi[letter]] = 1.0
                attn.W_o.weight[SLOT0 + 4 * h + c, base + c] = 1.0

        # Block 0, MLP: match bits, plus initial DP table (boundary = i or j, interior = INF_INIT).
        mlp0 = model.blocks[0].mlp
        for i in range(1, 9):
            for j in range(1, 9):
                ha = i - 1          # slot of a_i
                hb = 7 + j          # slot of b_j (position 8+j)
                mrow = M0 + 8 * (i - 1) + (j - 1)
                for c in range(4):
                    u = 4 * (8 * (i - 1) + (j - 1)) + c
                    mlp0.fc1.weight[u, SLOT0 + 4 * ha + c] = 1.0
                    mlp0.fc1.weight[u, SLOT0 + 4 * hb + c] = 1.0
                    mlp0.fc1.bias[u] = -1.0
                    mlp0.fc2.weight[mrow, u] = 1.0
        for i in range(9):
            for j in range(9):
                for k in range(1, 9):
                    if i == 0 or j == 0:
                        v = i + j                       # boundary D[0][j] = j, D[i][0] = i
                    else:
                        v = INF_INIT                    # interior starts at INF_INIT
                    if v >= k:
                        mlp0.fc2.bias[dp_chan(i, j, k)] = 1.0

        # Blocks 1..15: one DP (Jacobi) step each. Attention is absent.
        for b in range(1, 1 + N_DP_LAYERS):
            mlp = model.blocks[b].mlp
            u = 0
            for i in range(1, 9):
                for j in range(1, 9):
                    mrow = M0 + 8 * (i - 1) + (j - 1)
                    U = (i - 1, j)
                    Lf = (i, j - 1)
                    G = (i - 1, j - 1)
                    for k in range(1, 9):
                        u1, u2, u3 = u, u + 1, u + 2
                        u += 3
                        # h1 = relu(U[k-1] + L[k-1] + G[k] + m - 3): fires iff m=1 and all three hold
                        for ch in (bit_chan(*U, k - 1), bit_chan(*Lf, k - 1), bit_chan(*G, k)):
                            mlp.fc1.weight[u1, ch] += 1.0
                        mlp.fc1.weight[u1, mrow] += 1.0
                        mlp.fc1.bias[u1] = -3.0
                        # h2 = relu(U[k-1] + L[k-1] + G[k-1] - m - 2): fires iff m=0 and all three hold
                        for ch in (bit_chan(*U, k - 1), bit_chan(*Lf, k - 1), bit_chan(*G, k - 1)):
                            mlp.fc1.weight[u2, ch] += 1.0
                        mlp.fc1.weight[u2, mrow] -= 1.0
                        mlp.fc1.bias[u2] = -2.0
                        # u3 = relu(old bit): passes the old bit through so it can be subtracted
                        mlp.fc1.weight[u3, dp_chan(i, j, k)] = 1.0
                        d = dp_chan(i, j, k)
                        mlp.fc2.weight[d, u1] = 1.0
                        mlp.fc2.weight[d, u2] = 1.0
                        mlp.fc2.weight[d, u3] = -1.0

        # Head: digit d logit = [D>=d] - [D>=d+1]; equals 1 for the true distance, else <= 0.
        for d in range(9):
            row = stoi[str(d)]
            model.head.weight[row, bit_chan(8, 8, d)] += 1.0
            if d + 1 <= 8:
                model.head.weight[row, dp_chan(8, 8, d + 1)] -= 1.0
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ThermoDP-d11-inf5"
model_description = (
    "ThermoDP with 11 DP layers and interior cells initialised to 5. INF sweep at the exact depth."
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
    # TEMP STRESS TEST (remove before snapshot)
    _ex = task.generate_examples(20000, seed=123)
    _ids = torch.tensor([task.encode(e.prompt) for e in _ex])
    with torch.no_grad():
        _pred = model(_ids)[:, -1].argmax(-1).tolist()
    _bad = [(e.prompt, e.answer, task.itos[p]) for e, p in zip(_ex, _pred) if task.itos[p] != e.answer]
    print("STRESS n=20000 wrong:", len(_bad), "first:", _bad[:5])

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
