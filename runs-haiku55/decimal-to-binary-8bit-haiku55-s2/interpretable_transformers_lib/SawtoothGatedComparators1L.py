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
        # LayerNorm is replaced by identity: the hand-built circuit stores a
        # scalar (x) in one residual dimension, which LN would rescale.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """1-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 1024,
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

# Residual-stream layout (d_model = 64):
TOK = 0    # digit value of the current token (0..9; '=' -> 0)
X = 1      # running value x = 100*a + 10*b + c, written by attention
BIT = 2    # the output bit for the current position, written by the MLP
ONE = 3    # constant 1 (set in every position embedding)
POS = 8    # one-hot position indicator e_p lives in dims POS..POS+11

N_PROMPT = 4    # "abc="
N_ANSWER = 8    # 8 output bits, MSB first

ATTN_SHARPNESS = 160.0   # q.k before 1/sqrt(d_head); ~40 after scaling -> near-hard attention
GATE = 300.0             # position-gate strength for the bit units (x <= 255)
LOGIT_SCALE = 10.0


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Circuit (all closed form):
      1. Attention (heads 0,1,2): each head attends to one fixed prompt position
         (hundreds, tens, units digit) and writes its digit, scaled by 100/10/1,
         into residual dim X. So X = 100a + 10b + c.
      2. MLP: bit j of x (x in [0, 255]) is a sawtooth: bit_j(x) = floor(x/2^j) mod 2.
         Each jump at v = m*2^j is an exact comparator [x >= v] = relu(x-v+1) - relu(x-v)
         (valid for integer x), with sign +1 for odd m and -1 for even m.
         Units for bit j are gated by the indicator of the output position that
         emits bit j, so only one bit is nonzero at any position.
      3. Output head: logit('1') = L*BIT, logit('0') = L*(1 - BIT).
    """
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        H = model.n_heads
        dh = D // H
        blk = model.blocks[0]

        # --- embeddings -----------------------------------------------------
        for d in range(10):
            model.token_emb.weight[d, TOK] = float(d)
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, ONE] = 1.0
            if p < N_PROMPT + N_ANSWER:
                model.pos_emb.weight[p, POS + p] = 1.0

        # --- attention: gather the three digits into X ----------------------
        # head h targets prompt position h (0 = hundreds, 1 = tens, 2 = units).
        # query: constant (from ONE) pointing at target index h;
        # key: one-hot of the key's own position; value: its token digit.
        place_value = {0: 100.0, 1: 10.0, 2: 1.0}
        attn = blk.attn
        for h, w in place_value.items():
            base = h * dh
            attn.W_q.weight[base + h, ONE] = ATTN_SHARPNESS
            for i in range(N_PROMPT + N_ANSWER):
                attn.W_k.weight[base + i, POS + i] = 1.0
            attn.W_v.weight[base + 0, TOK] = 1.0
            attn.W_o.weight[X, base + 0] = w

        # --- MLP: sawtooth comparators gated by output position --------------
        mlp = blk.mlp
        W1, b1 = mlp.fc1.weight, mlp.fc1.bias
        W2 = mlp.fc2.weight
        u = 0
        for j in range(N_ANSWER):
            # output bit j (LSB = 0) is emitted from position t = N_PROMPT - 1 + (N_ANSWER - 1 - j)
            # (the MSB, j = 7, is predicted at the '=' position t = 3)
            t = N_PROMPT - 1 + (N_ANSWER - 1 - j)
            step = 2 ** j
            for m in range(1, 255 // step + 1):
                v = m * step
                sign = 1.0 if m % 2 == 1 else -1.0
                # A-unit: relu(x - v + 1 + GATE*(p-1))  -> [x >= v] when gated on
                W1[u, X] = 1.0
                W1[u, POS + t] = GATE
                b1[u] = 1.0 - v - GATE
                W2[BIT, u] = sign
                u += 1
                # B-unit: relu(x - v + GATE*(p-1))  -> subtracted from A to give a step
                W1[u, X] = 1.0
                W1[u, POS + t] = GATE
                b1[u] = -float(v) - GATE
                W2[BIT, u] = -sign
                u += 1
        assert u <= model.d_ff, f"need {u} hidden units, have {model.d_ff}"

        # --- output head ----------------------------------------------------
        model.head.weight[1, BIT] = LOGIT_SCALE            # '1'
        model.head.weight[0, ONE] = LOGIT_SCALE            # '0' = L*(1 - BIT)
        model.head.weight[0, BIT] = -LOGIT_SCALE


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SawtoothGatedComparators1L"
model_description = (
    "1-layer, no-LN: attention gathers a,b,c into x=100a+10b+c; MLP computes each output bit as a "
    "sawtooth of x via gated [x>=v] comparators (relu(x-v+1)-relu(x-v)), gated by output position; "
    "d_ff=1024, hard attention (sharpness 160)."
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
