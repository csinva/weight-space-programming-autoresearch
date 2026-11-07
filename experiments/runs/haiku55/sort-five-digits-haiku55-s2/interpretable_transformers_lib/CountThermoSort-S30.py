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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 42,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 19,
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
        self.final_ln = nn.LayerNorm(d_model)
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built one-layer sorter: counting by attention, comparison in the MLP.

    Sequence: d0 d1 d2 d3 d4 '=' a0 a1 a2 a3 a4 (positions 0..10).
    Output a_i (predicted at position 5+i) is the smallest v with C(v) > i,
    where C(v) = #{input digits <= v}. Equivalently a_i = #{v in 0..8 : C(v) <= i}.

    Mechanism (all weights closed-form):
      * LayerNorm: every token/position embedding is a zero-sum one-hot scaled by S,
        so every scaffold vector has the same norm and mean 0; the LN scale is a known
        constant sig. Weights reading LN outputs have zero row sums, so the LN mean
        drops out exactly.
      * Attention: the query at answer positions and the keys at digit positions 0..4
        get a flag; scores differ by 20 so the 5 digits are averaged uniformly. Each
        digit is written as a thermometer thermo_v(d) = [d <= v]; the average is
        C(v)/5 minus a known constant (v+1)/11.
      * MLP: for each v, neuron N1 = relu(i - C(v) + 1), N2 = relu(i - C(v)), where
        i comes from the position embedding. U_v = N1 - N2 = [C(v) <= i].
      * Head: logit_k = U_{k-1} - U_k (with U_{-1} = 1), which is 1 exactly for k = a_i.
        '=' gets logit 0, so the argmax is always a digit.
    Residual layout: TOK 0..10, POS 11..21, THERMO 22..30, U 31..39, ONE 40, ZERO 41.
    """
    D = model.d_model
    assert model.n_layers == 1 and model.n_heads == 1 and D >= 42 and model.d_ff >= 19
    S = 30.0
    Z = 1.0 / 11.0
    sig = S * math.sqrt((20.0 / 11.0) / D)  # LN std of every scaffold vector
    blk = model.blocks[0]
    TOK, POS, THERMO, U, ONE, ZERO = 0, 11, 22, 31, 40, 41

    with torch.no_grad():
        # --- embeddings: zero-sum one-hot scaffold (never touch LayerNorm params)
        model.token_emb.weight.zero_()
        for d in range(11):
            model.token_emb.weight[d, TOK:TOK + 11] = S * (F.one_hot(torch.tensor(d), 11).float() - Z)
        model.pos_emb.weight.zero_()
        for p in range(11):
            model.pos_emb.weight[p, POS:POS + 11] = S * (F.one_hot(torch.tensor(p), 11).float() - Z)

        # --- attention (single head, d_head = D)
        A = blk.attn
        for W in (A.W_q.weight, A.W_k.weight, A.W_v.weight, A.W_o.weight):
            W.zero_()
        # flag dimension 0: queries at positions 5..10 vs keys at digit positions 0..4
        A.W_q.weight[0, POS + 5:POS + 11] = 1.0
        A.W_k.weight[0, POS:POS + 5] = 10.0
        for v in range(9):  # value dim v = thermometer [d <= v] over digit tokens
            for d in range(10):
                A.W_v.weight[v, TOK + d] = (sig / S) * (1.0 if d <= v else 0.0)
            A.W_o.weight[THERMO + v, v] = 1.0

        # --- MLP (hidden: N1_v = 0..8, N2_v = 9..17, constant = 18)
        M = blk.mlp
        M.fc1.weight.zero_(); M.fc1.bias.zero_()
        M.fc2.weight.zero_(); M.fc2.bias.zero_()
        sum_c = 15.0 / S                       # sum of c_p over p = 5..10, c_p = (p-5)/S
        a = (5.0 - sum_c) / 11.0               # zero-sum fix on TOK dims (scaffold TOK sums to 0)
        for v in range(9):
            w = torch.zeros(D)
            w[THERMO + v] = -5.0
            for p in range(5, 11):
                w[POS + p] = (p - 5) / S
            w[TOK:TOK + 11] = a
            M.fc1.weight[v] = sig * w
            M.fc1.weight[9 + v] = sig * w
            M.fc1.bias[v] = 1.0 + 15.0 / 11.0 - 5.0 * (v + 1) / 11.0
            M.fc1.bias[9 + v] = 15.0 / 11.0 - 5.0 * (v + 1) / 11.0
            M.fc2.weight[U + v, v] = 1.0
            M.fc2.weight[U + v, 9 + v] = -1.0
        M.fc1.bias[18] = 1.0                   # constant neuron (no input weights)
        M.fc2.weight[ONE, 18] = 1.0

        # --- head: logit_k = U_{k-1} - U_k (zero-sum rows; '=' row is zero)
        model.head.weight.zero_()
        model.head.weight[0, ONE] = 1.0
        model.head.weight[0, U + 0] = -1.0
        for k in range(1, 9):
            model.head.weight[k, U + k - 1] = 1.0
            model.head.weight[k, U + k] = -1.0
        model.head.weight[9, U + 8] = 1.0
        model.head.weight[9, ZERO] = -1.0      # U_9 = 0, written as a zero-valued dim


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CountThermoSort-S30"
model_description = "d42/ff19 circuit with scaffold scale S = 30; the LayerNorm scale is larger, so the feature perturbation is smaller relative to the scaffold."


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
