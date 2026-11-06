"""
Interpretable transformer for character-level sequence tasks.

Hand-written weights for `multiplication-five-digits`:
prompt "AAAAA*BBBBB=" -> answer "PPPPPPPPPP".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.
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
# LayerNorm-free transformer (so residual scalars stay exact), multi-head
# attention with configurable head count / head width, ReLU MLPs, linear
# unembedding. Runs in float64 because the product needs ~34 bits.

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        HD = n_heads * d_head
        self.W_q = nn.Linear(d_model, HD, bias=False)
        self.W_k = nn.Linear(d_model, HD, bias=False)
        self.W_v = nn.Linear(d_model, HD, bias=False)
        self.W_o = nn.Linear(HD, d_model, bias=False)

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
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer without LayerNorm."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 96,
        n_heads: int = 19,
        d_head: int = 10,
        n_layers: int = 2,
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_head, d_ff) for _ in range(n_layers)])
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
# Algorithm (answer digit k = 0..9, MSD first, predicted at position t = 11+k):
#   A  = sum_i a_i 10^(4-i)                 (5 attention heads, one per digit of A)
#   b_j one-hot                             (5 heads copy the digits of B)
#   G  = sum_{m<k} P_m 10^(9-m)             (9 heads read already generated digits;
#                                            attention sinks on '=' (value 0) if not yet generated)
#   R  = A*B - G  in [0, 10^(10-k))         (MLP1: A*B = sum_j sum_v [b_j=v] * v 10^(4-j) A,
#                                            each term an exact ReLU gate)
#   c  = #{d in 1..9 : R >= d*10^(9-k)}     (MLP2: position-gated ReLU step pairs; R is integer)
#   logit_j = 2*j*c - j^2  -> argmax = c    (linear unembedding)

# residual layout
TOK = 0          # 12 token one-hot dims (0..11)
DIGVAL = 12      # numeric value of digit token
POS = 13         # 22 position one-hot dims (13..34)
CONST = 35       # constant 1
A_DIM = 36       # value of A
BOH = 37         # 5 x 10 one-hot of B digits (37..86)
R_DIM = 87       # remainder
C_DIM = 88       # digit count

S_ATT = 200.0    # attention score gap (softmax leak ~ e^-200)
BIG = 1e11       # gating constant (> any |R|, |A*B|)


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    model.double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        stoi = task.stoi
        dh = model.blocks[0].attn.d_head
        sq = math.sqrt(dh)

        # ---- embeddings
        for ch, i in stoi.items():
            model.token_emb.weight[i, TOK + i] = 1.0
            if ch.isdigit():
                model.token_emb.weight[i, DIGVAL] = float(ch)
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS + p] = 1.0
            model.pos_emb.weight[p, CONST] = 1.0

        # ---- layer 1 attention
        at = model.blocks[0].attn
        h = 0
        # A digits: head reads position i, writes digval * 10^(4-i) into A_DIM
        for i in range(5):
            c0 = h * dh
            at.W_q.weight[c0, CONST] = S_ATT * sq
            at.W_k.weight[c0, POS + i] = 1.0
            at.W_v.weight[c0, DIGVAL] = 1.0
            at.W_o.weight[A_DIM, c0] = 10.0 ** (4 - i)
            h += 1
        # B digits: head reads position 6+j, copies digit one-hot into BOH block j
        for j in range(5):
            c0 = h * dh
            at.W_q.weight[c0, CONST] = S_ATT * sq
            at.W_k.weight[c0, POS + 6 + j] = 1.0
            for v in range(10):
                at.W_v.weight[c0 + v, TOK + stoi[str(v)]] = 1.0
                at.W_o.weight[BOH + 10 * j + v, c0 + v] = 1.0
            h += 1
        # generated digit P_m (input at position 12+m); sink at '=' (pos 11, digval 0)
        for m in range(9):
            c0 = h * dh
            at.W_q.weight[c0, CONST] = S_ATT * sq
            at.W_k.weight[c0, POS + 12 + m] = 2.0
            at.W_k.weight[c0, POS + 11] = 1.0
            at.W_v.weight[c0, DIGVAL] = 1.0
            at.W_o.weight[R_DIM, c0] = -(10.0 ** (9 - m))
            h += 1
        assert h <= at.n_heads

        # ---- layer 1 MLP: R += A*B
        m1 = model.blocks[0].mlp
        u = 0
        for j in range(5):
            for v in range(1, 10):
                m1.fc1.weight[u, A_DIM] = v * 10.0 ** (4 - j)
                m1.fc1.weight[u, BOH + 10 * j + v] = BIG
                m1.fc1.bias[u] = -BIG
                m1.fc2.weight[R_DIM, u] = 1.0
                u += 1

        # ---- layer 2 attention: unused (all zero)
        # ---- layer 2 MLP: c = #{d : R >= d*10^(9-k)} at position 11+k
        m2 = model.blocks[1].mlp
        u = 0
        for k in range(10):
            t = 11 + k
            for d in range(1, 10):
                thr = d * 10.0 ** (9 - k)
                for off, sign in ((1.0, 1.0), (0.0, -1.0)):
                    m2.fc1.weight[u, R_DIM] = 1.0
                    m2.fc1.weight[u, POS + t] = BIG
                    m2.fc1.bias[u] = -thr + off - BIG
                    m2.fc2.weight[C_DIM, u] = sign
                    u += 1
        assert u <= model.d_ff

        # ---- unembedding: logit_j = 2 j c - j^2
        for j in range(10):
            model.head.weight[stoi[str(j)], C_DIM] = 2.0 * j
            model.head.weight[stoi[str(j)], CONST] = -float(j * j)
        for ch in "*=":
            model.head.weight[stoi[ch], CONST] = -1e6


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "MulRemainderGated_v1"
model_description = ("float64 LN-free 2-layer transformer: heads gather A (scalar), B digits (one-hot) and "
                     "generated digits; MLP1 forms R=A*B-prefix via exact ReLU gating; MLP2 counts "
                     "position-gated thresholds R>=d*10^(9-k); linear unembedding picks the digit.")


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
