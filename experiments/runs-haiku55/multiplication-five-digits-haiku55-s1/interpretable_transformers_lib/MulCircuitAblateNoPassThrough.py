"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task: multiply two 5-digit numbers.
  prompt "50494*99346=" -> answer "5016376924"  (10 digits, zero-padded, MSD first)

Circuit (no training; every weight written in closed form):
  1. Attention gathers the inputs into each answer position (one head per digit,
     sharp attention on a fixed prompt position): a_i as a scalar digit value,
     b_j as a one-hot digit.
  2. MLP "multiplier lookup": unit relu(a_i + 10*[b_j=y] - 10) = a_i*[b_j=y];
     output weight y gives a_i*b_j, summed into column S_c with c = 8-(i+j).
  3. Carry-save reduction (3 rounds): split each column into tens/units with floor()
     implemented by exact ReLU sawtooths, add the tens of the column below.
  4. Carry lookahead: columns are now in {0..10}; a column of 10 generates a carry,
     a column of 9 propagates one. Carry into column c = OR over the nearest lower
     non-9 column being 10.
  5. Digit = (column + carry_in) mod 10 as a one-hot; the answer position for step k
     (k = 0..9, MSD first) reads column 9-k through a position-gated AND.

Architecture note: LayerNorm is removed (it would rescale the exact one-hot / integer
signals). Attention is ordinary causal softmax with very sharp scores.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model.
  * `write_weights()` runs once at model construction and populates all parameters.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task multiplication-five-digits --n-samples 500
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
# Residual-stream layout (one named subspace per feature)
# ---------------------------------------------------------------------------

D_MODEL = 324
TOK = 0      # 12 dims: one-hot token id (digits 0-9, '*', '=')
POS = 12     # 22 dims: one-hot position
GA = 34      # 100 dims: gathered digits; digit slot h (0..9) at GA+10h .. GA+10h+9
             #           slots 0..4 = a0..a4, slots 5..9 = b0..b4
S = 134      # 10 dims: column sums S_c (c = power of 10, 0..9)
D1 = 144     # 10 dims: carry-save round 1: S_c - 10*floor(S_c/10) + floor(S_{c-1}/10)
D2 = 154     # 10 dims: round 2: D1_c - 10*floor(D1_c/10) + floor(D1_{c-1}/10)
D3 = 164     # 10 dims: round 3 value in [0, 10]
EQ = 174     # 10 dims: [D3_c == 10]
NQ = 184     # 10 dims: [D3_c == 9]
CIN = 194    # 10 dims: carry into column c
COUT = 204   # 10 dims: carry out of column c
DIG = 214    # 100 dims: one-hot digit per column: DIG + 10*c + d
OUT = 314    # 10 dims: one-hot answer digit for the current answer position

BETA = 60.0  # attention sharpness (pre-scale); scores are divided by sqrt(d_head)

# Per-layer spec: d_ff = hidden width; attn = whether the block has an attention sublayer.
LAYER_SPECS = [
    dict(d_ff=225, attn=True, n_heads=10, d_head=10),    # 0: gather digits + pair products
    dict(d_ff=810),                                      # 1: D1 (pass-through + floor(S/10) thresholds)
    dict(d_ff=90),                                       # 2: D2 (pass-through + floor(D1/10) thresholds)
    dict(d_ff=30),                                       # 3: D3 (pass-through + floor(D2/10) thresholds)
    dict(d_ff=60),                                       # 4: [D3==10], [D3==9]
    dict(d_ff=45),                                       # 5: carry-in lookahead
    dict(d_ff=20),                                       # 6: carry-out
    dict(d_ff=300),                                      # 7: one-hot digit per column
    dict(d_ff=100),                                      # 8: position-gated output digit
]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
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
    """Residual block without LayerNorm: x += attn(x) (optional); x += mlp(x)."""

    def __init__(self, d_model: int, d_ff: int, attn: CausalSelfAttention | None = None):
        super().__init__()
        self.attn = attn
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer with a fixed stack of blocks (see LAYER_SPECS)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 22,
        d_model: int = D_MODEL,
        layer_specs: list[dict] = LAYER_SPECS,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = len(layer_specs)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        blocks = []
        for spec in layer_specs:
            attn = None
            if spec.get("attn", False):
                attn = CausalSelfAttention(d_model, spec["n_heads"], spec["d_head"])
            blocks.append(Block(d_model, spec["d_ff"], attn))
        self.blocks = nn.ModuleList(blocks)
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

def _fill(mlp: MLP, units: list) -> None:
    """units: list of (ins, bias, outs). ins/outs are lists of (residual_dim, coef).
    Unit u computes relu(sum coef*x[dim] + bias) and writes it to outs with its coefs."""
    assert len(units) <= mlp.fc1.out_features, (len(units), mlp.fc1.out_features)
    for u, (ins, bias, outs) in enumerate(units):
        for dim, coef in ins:
            mlp.fc1.weight[u, dim] += coef
        mlp.fc1.bias[u] = bias
        for dim, coef in outs:
            mlp.fc2.weight[dim, u] += coef


def _round_units(src: int, dst: int, k_max: int, carry_next: bool = True) -> list:
    """One carry-save round in a single layer:
        dst_c = src_c - 10*floor(src_c/10) + floor(src_{c-1}/10)
    Pass-through relu(src_c) = src_c (nonneg). floor(x/10) = sum_k [x >= 10k], and
    [x >= 10k] = relu(x-10k+1) - relu(x-10k) for integer x. Each threshold unit
    feeds its own column (-10) and the next column (+1)."""
    units = []
    for c in range(10):
        # ABLATION: no pass-through of src_c (the value itself is dropped)
        for k in range(1, k_max + 1):
            up = [(dst + c + 1, 1)] if (c < 9 and carry_next) else []
            down = [(dst + c + 1, -1)] if (c < 9 and carry_next) else []
            units.append(([(src + c, 1)], -(10 * k - 1), [(dst + c, -10)] + up))
            units.append(([(src + c, 1)], -(10 * k), [(dst + c, 10)] + down))
    return units


def _indicator_units(z_ins: list, k: int, out_base: int, c: int) -> list:
    """[z == k] for integer z: relu(z-k+1) - 2 relu(z-k) + relu(z-k-1)."""
    return [
        (z_ins, -(k - 1), [(out_base + c, 1)]),
        (z_ins, -k, [(out_base + c, -2)]),
        (z_ins, -(k + 1), [(out_base + c, 1)]),
    ]


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter of `model` in closed form. No training."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings: one-hot token and position subspaces --------------------
        for v in range(model.token_emb.num_embeddings):
            model.token_emb.weight[v, TOK + v] = 1.0
        for p in range(model.pos_emb.num_embeddings):
            model.pos_emb.weight[p, POS + p] = 1.0

        # --- block 0 attention: 10 "gather" heads, one per input digit -----------
        # Head h attends sharply to prompt position target(h): a_i at position i (h=i),
        # b_j at position 6+j (h=5+j). Its value is the one-hot digit at that position.
        att = model.blocks[0].attn
        for h in range(10):
            target = h if h < 5 else h + 1
            for t in range(11, model.max_seq_len):      # answer-position queries
                att.W_q.weight[h * 10, POS + t] = 1.0
            att.W_k.weight[h * 10, POS + target] = BETA
            if h < 5:
                # a_i: carry the digit VALUE as a scalar in slot GA+10h
                for v in range(10):
                    att.W_v.weight[h * 10, TOK + v] = float(v)
                att.W_o.weight[GA + 10 * h, h * 10] = 1.0
            else:
                # b_j: carry the one-hot digit in slot GA+10h
                for v in range(10):
                    att.W_v.weight[h * 10 + v, TOK + v] = 1.0
                    att.W_o.weight[GA + 10 * h + v, h * 10 + v] = 1.0

        # --- block 0 MLP: scalar-a times one-hot-b gates -> column sums ----------
        # unit relu(a_i + 10*[b_j=y] - 10) = a_i * [b_j=y]; output weight y gives a_i*b_j.
        # a_i has weight 10^(4-i), b_j has weight 10^(4-j) => column c = 8-(i+j).
        units = []
        for i in range(5):
            for j in range(5):
                c = 8 - (i + j)
                for y in range(1, 10):
                    units.append((
                        [(GA + 10 * i, 1), (GA + 10 * (5 + j) + y, 10.0)],
                        -10.0,
                        [(S + c, float(y))],
                    ))
        _fill(model.blocks[0].mlp, units)

        # --- carry-save rounds (each one layer) ----------------------------------
        _fill(model.blocks[1].mlp, _round_units(S, D1, 40))     # D1 in [0, 49]
        _fill(model.blocks[2].mlp, _round_units(D1, D2, 4))     # D2 in [0, 13]
        _fill(model.blocks[3].mlp, _round_units(D2, D3, 1))     # D3 in [0, 10]

        # --- carry lookahead ----------------------------------------------------
        units = []
        for c in range(10):
            units += _indicator_units([(D3 + c, 1)], 10, EQ, c)
            units += _indicator_units([(D3 + c, 1)], 9, NQ, c)
        _fill(model.blocks[4].mlp, units)

        units = []
        for c in range(1, 10):
            for cp in range(c):
                m = c - cp - 1  # number of 9-columns strictly between cp and c
                ins = [(EQ + cp, 1)] + [(NQ + cc, 1) for cc in range(cp + 1, c)]
                units.append((ins, -float(m), [(CIN + c, 1)]))
        _fill(model.blocks[5].mlp, units)

        # COUT_c = EQ_c + relu(NQ_c + CIN_c - 1)
        units = []
        for c in range(10):
            units.append(([(EQ + c, 1)], 0.0, [(COUT + c, 1)]))
            units.append(([(NQ + c, 1), (CIN + c, 1)], -1.0, [(COUT + c, 1)]))
        _fill(model.blocks[6].mlp, units)

        # one-hot digit: z = D3_c + CIN_c - 10*COUT_c in [0, 9]
        units = []
        for c in range(10):
            z_ins = [(D3 + c, 1), (CIN + c, 1), (COUT + c, -10)]
            for d in range(10):
                for ins, bias, outs in _indicator_units(z_ins, d, 0, 0):
                    units.append((ins, bias, [(DIG + 10 * c + d, outs[0][1])]))
        _fill(model.blocks[7].mlp, units)

        # output: OUT_d = OR_k [position 11+k] AND [column 9-k has digit d]
        units = []
        for k in range(10):
            for d in range(10):
                units.append((
                    [(POS + 11 + k, 1), (DIG + 10 * (9 - k) + d, 1)],
                    -1.0,
                    [(OUT + d, 1)],
                ))
        _fill(model.blocks[8].mlp, units)

        # --- readout: logit for digit v is OUT_v (non-digit tokens stay at 0) ----
        for v in range(10):
            model.head.weight[v, OUT + v] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "MulCircuitAblateNoPassThrough"
model_description = (
    "Ablation of MulCircuitFusedRounds at BETA=60: the carry-save rounds drop the pass-through of "
    "the column value itself, keeping only the floor/carry terms (tests that the units digit is carried)."
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
