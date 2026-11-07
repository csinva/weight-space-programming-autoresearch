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

# LayerNorm is replaced by identity: the residual stream stays an exact linear
# sum of hand-placed features, which makes closed-form weight design possible.
USE_LN = False


def _norm(d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model) if USE_LN else nn.Identity()


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
        self.ln1 = _norm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = _norm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 80,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 100,
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
        self.final_ln = _norm(d_model)
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

# Knobs (ablations / robustness sweeps). Baseline values give the 100% circuit.
CARRY_MODE = "full"     # "full": highest non-propagate column below e; "prev1": only column e-1; "none": no carry
SCORE_SCALE = 1.0       # multiplies every attention score (softmax temperature)
BIG_GATE = 10.0         # gate strength for the B-position-only pair table
K_FETCH = 50.0          # layer-1 partner-fetch match score
H1_MATCH = 80.0         # layer-2 digit-fetch match bonus
H1_PEN = 40.0           # layer-2 digit-fetch penalty on non-matching B keys
E_W = 50.0              # carry-head weight on the non-propagate flag
GE_PEN = 0.0            # carry-head penalty for columns j >= e
OUT_W = 2.0             # output weight of the lookup units (argmax-invariant)
WEIGHT_NOISE = 0.0      # std of Gaussian noise added to all weights after construction

# Residual-stream layout (d_model = 80). Each named block is one feature family.
T0 = 0       # token one-hot: dims 0-9 = digits, 10 = '+', 11 = '='
P0 = 12      # position one-hot: position p -> dim 12+p (p = 0..17)
AF0 = 30     # [layer 1] A-digit fetched from partner position (one-hot, 10)
X0 = 40      # [layer 1] (a_j + b_j) mod 10 one-hot at B-digit positions (10)
G_D = 50     # [layer 1] generate flag: a_j + b_j >= 10
E_D = 51     # [layer 1] non-propagate flag: a_j + b_j != 9
XF0 = 52     # [layer 2] fetched column digit (a_e + b_e) mod 10 one-hot (10)
CF_D = 62    # [layer 2] fetched carry into the column being written
OUT0 = 63    # output digit one-hot (10)

A_POS = list(range(0, 5))    # A digits: powers 4..0 at positions 0..4
B_POS = list(range(6, 11))   # B digits: powers 4..0 at positions 6..10
ANS_Q = list(range(11, 17))  # query positions predicting answer digits (e = 5..0)
NON_B = list(range(0, 6)) + list(range(11, 18))  # positions with no B digit


def pos_dim(p: int) -> int:
    return P0 + p


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-build a 2-layer carry-lookahead adder. No training.

    Algorithm (answer is MSB-first, so the carry into each column must be
    looked ahead from lower columns):
      Layer 1 (B-digit positions): fetch a_j from the partner position (r-6),
        then compute the pair features (a_j+b_j) mod 10, G = [a_j+b_j>=10],
        E = [a_j+b_j != 9] via a 10x10 product lookup table.
      Layer 2 (answer positions r = 11..16, column e = 16 - r):
        head 0 "carry": attend to the highest column j < e with E_j = 1 and
          read its G_j. This is exactly carry-into-e (a column with sum 9
          propagates the carry; any other column decides it).
        head 1 "digit": attend to column e and fetch (a_e+b_e) mod 10.
        MLP: output (digit + carry) mod 10 via a 20-unit lookup table.
    """
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):   # identity LayerNorm when USE_LN is on
                m.weight.fill_(1.0)
                m.bias.zero_()

        D = model.d_model
        H = model.n_heads
        dh = D // H
        sq = math.sqrt(dh) * SCORE_SCALE  # attention divides by sqrt(dh); pre-scale weights
        BIG = BIG_GATE

        # Embeddings: one-hot token, one-hot position.
        for t in range(12):
            model.token_emb.weight[t, T0 + t] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, P0 + p] = 1.0

        L1, L2 = model.blocks[0], model.blocks[1]

        # ---- Layer 1, head 0: B-position r fetches A-digit at partner r-6 ----
        for s in range(5):
            # query at position s+6 has dim s = K; key at position s has dim s = 1
            L1.attn.W_q.weight[s, pos_dim(s + 6)] = K_FETCH * sq
            L1.attn.W_k.weight[s, pos_dim(s)] = 1.0
        for d in range(10):
            L1.attn.W_v.weight[d, T0 + d] = 1.0         # value = token digit
            L1.attn.W_o.weight[AF0 + d, d] = 1.0        # write to AF block

        # ---- Layer 1 MLP: pair table (a_i, b_j) -> X, G, E at B positions ----
        for i in range(10):
            for j in range(10):
                u = 10 * i + j
                L1.mlp.fc1.weight[u, AF0 + i] = 1.0
                L1.mlp.fc1.weight[u, T0 + j] = 1.0
                for p in B_POS:
                    L1.mlp.fc1.weight[u, pos_dim(p)] = BIG   # gate: only B positions
                L1.mlp.fc1.bias[u] = -BIG - 1.5
                L1.mlp.fc2.weight[X0 + (i + j) % 10, u] = OUT_W
                if i + j >= 10:
                    L1.mlp.fc2.weight[G_D, u] = OUT_W
                if i + j != 9:
                    L1.mlp.fc2.weight[E_D, u] = OUT_W

        # ---- Layer 2, head 0: carry-in via highest non-propagate column j < e ----
        # score(key column j, query e) = 10*j - GE_PEN*[j >= e] + E_W*E_j   (CARRY_MODE "full")
        if CARRY_MODE != "none":
            for r in ANS_Q:
                e = 16 - r
                for j in range(5):
                    if CARRY_MODE == "full":
                        coef = 10.0 * j - (GE_PEN if j >= e else 0.0)
                    else:  # "prev1": only column e-1 may win
                        coef = 100.0 if j == e - 1 else -GE_PEN
                    L2.attn.W_q.weight[j, pos_dim(r)] = coef * sq
                if CARRY_MODE == "full":
                    L2.attn.W_q.weight[5, pos_dim(r)] = E_W * sq
            for j in range(5):
                L2.attn.W_k.weight[j, pos_dim(10 - j)] = 1.0   # column j lives at pos 10-j
            L2.attn.W_k.weight[5, E_D] = 1.0
            L2.attn.W_v.weight[0, G_D] = 1.0
            L2.attn.W_o.weight[CF_D, 0] = 1.0

        # ---- Layer 2, head 1: fetch (a_e+b_e) mod 10 from column e ----
        # score = H1_MATCH*[j==e] - H1_PEN*valid_key ; non-B keys score 0 with X = 0
        for r in ANS_Q:
            e = 16 - r
            for j in range(5):
                L2.attn.W_q.weight[dh + j, pos_dim(r)] = (H1_MATCH if j == e else 0.0) * sq
            L2.attn.W_q.weight[dh + 5, pos_dim(r)] = -H1_PEN * sq
        for j in range(5):
            L2.attn.W_k.weight[dh + j, pos_dim(10 - j)] = 1.0
        for p in B_POS:
            L2.attn.W_k.weight[dh + 5, pos_dim(p)] = 1.0
        for d in range(10):
            L2.attn.W_v.weight[dh + d, X0 + d] = 1.0
            L2.attn.W_o.weight[XF0 + d, dh + d] = 1.0
        # "no column" (e = 5): non-B keys win; each non-B position's position one-hot
        # contributes 1 to the digit-0 slot, so the averaged value is 1 (B keys contribute 0).
        for p in NON_B:
            L2.attn.W_v.weight[dh + 0, pos_dim(p)] = 1.0

        # ---- Layer 2 MLP: (x, c) -> (x + c) mod 10 lookup ----
        for x in range(10):
            # unit x: fires when digit x and carry 0  -> output digit x
            L2.mlp.fc1.weight[x, XF0 + x] = 1.0
            L2.mlp.fc1.weight[x, CF_D] = -1.0
            L2.mlp.fc1.bias[x] = -0.5
            L2.mlp.fc2.weight[OUT0 + x, x] = OUT_W
            # unit 10+x: fires when digit x and carry 1 -> output digit (x+1) mod 10
            L2.mlp.fc1.weight[10 + x, XF0 + x] = 1.0
            L2.mlp.fc1.weight[10 + x, CF_D] = 1.0
            L2.mlp.fc1.bias[10 + x] = -1.5
            L2.mlp.fc2.weight[OUT0 + (x + 1) % 10, 10 + x] = OUT_W

        # ---- Output head: digit k token reads OUT_k ----
        for k in range(10):
            model.head.weight[k, OUT0 + k] = 1.0

        # ---- Optional perturbation (robustness test; not training) ----
        if WEIGHT_NOISE > 0:
            gen = torch.Generator().manual_seed(0)
            for prm in model.parameters():
                prm.add_(torch.randn(prm.shape, generator=gen) * WEIGHT_NOISE)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "GePen0-2L"
model_description = (
    "Ablation: carry head has no penalty for columns j >= e (GE_PEN=0), so columns above the target "
    "digit can be selected as carry source. Tests whether the column-ordering mask is needed."
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
