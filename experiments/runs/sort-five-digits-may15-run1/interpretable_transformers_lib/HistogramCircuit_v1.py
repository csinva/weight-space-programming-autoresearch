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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
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
    """Hand-designed circuit for sort-five-digits.

    Idea:
      - vocab = "0123456789=", token ids 0..9 are digits, id 10 is '='.
      - Sequence layout: positions 0..4 are input digits, position 5 is '=',
        positions 6..9 are the autoregressively-generated outputs. The model
        predicts the k-th sorted digit using logits read at position 5+k.
      - Attention head averages the 5 input positions and writes the
        histogram h[d] = #{input digits equal to d} into residual dims 11..20.
      - Pos embedding marks which output position k = 0..4 we are predicting
        for via a one-hot in residual dims 21..25.
      - MLP turns histogram + k into a one-hot of the k-th smallest digit by
        computing  I[c_d > k] - I[c_{d-1} > k]  where c_d = cumcount(d).
        Each indicator is a clamp(c_d-k, 0, 1) = relu(σ·(c_d-k)) - relu(σ·(c_d-k-1)).
      - Head reads the one-hot residual dims 26..35 as digit logits.

    Residual layout (d_model = 64):
        [0..9]   = current-token digit one-hot (token_emb)
        [10]     = +1 at input positions, -1 at output query positions (pos_emb)
        [11..20] = histogram (written by attention)
        [21..25] = k one-hot at output query positions (pos_emb)
        [26..35] = predicted-digit one-hot (written by MLP, read by head)
        [36..63] = ±3 anchor pattern (alternating; keeps post-LN std ≈ constant)

    LayerNorm scale-invariance: a fixed ±3 pattern on 28 dims has squared sum
    252, which dominates the data-dependent dims (whose squared sum is < 30).
    Thus pre-LN std ≈ sqrt(254/64) ≈ 1.99 at every position regardless of
    input, so post-LN values are simply (raw - small_mean) / ~2.0.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.zero_()

    D = model.d_model         # 64
    L = model.max_seq_len     # 16

    # ----- Token embedding: digit d -> one-hot at dim d. '=' -> zero. -----
    for d in range(10):
        model.token_emb.weight.data[d, d] = 1.0

    # ----- Position embedding -----
    # Anchor at every position: alternating ±3 in dims 36..63 (28 dims).
    anchor = torch.zeros(D)
    for i in range(36, 64):
        anchor[i] = 3.0 if (i % 2 == 0) else -3.0

    pos = model.pos_emb.weight.data
    for p in range(L):
        pos[p] += anchor
    for p in range(5):                          # input positions: marker = +1
        pos[p, 10] = 1.0
    for k in range(5):                          # output query positions 5..9
        pos[5 + k, 10] = -1.0
        pos[5 + k, 21 + k] = 1.0                # one-hot of output index k

    # Pre-LN1 squared sums at every position:
    #   anchor: 28*9 = 252
    #   data dims: 1 (dim d for token) + 1 (dim 10) + ≤1 (dim 21+k) ≈ 2 or 3
    # std ≈ sqrt(254/64) ≈ 1.992; gain σ := 1/std ≈ 0.502.
    SIGMA = 1.0 / 1.992

    blk = model.blocks[0]

    # ----- LN1: identity affine -----
    blk.ln1.weight.data.fill_(1.0)
    blk.ln1.bias.data.zero_()

    # ----- Attention (1 head, d_head = D) -----
    # Output query positions should attend uniformly to the 5 input positions.
    # Score = (Wq·r_query) · (Wk·r_key) / sqrt(d_head).
    # Put -A at column 10 of Wq and +A at column 10 of Wk so the score is
    # large positive when r_query[10] < 0 (output positions) and r_key[10] > 0
    # (input positions).
    A = 10.0
    blk.attn.W_q.weight.data[0, 10] = -A
    blk.attn.W_k.weight.data[0, 10] = +A

    # V copies the digit one-hot.
    for d in range(10):
        blk.attn.W_v.weight.data[d, d] = 1.0

    # When attention is fully on the 5 input positions, attn_out at dim d:
    #   = (1/5) * sum_i  postLN1[i, d]
    # postLN1[i, d] = (1 - mean)/std  if d == digit_at_i  else (-mean)/std
    # mean at an input position = (1 + 1 + 0)/64 = 1/32 ≈ 0.03125, std ≈ 1.992.
    # So postLN1[i, d] ≈ 0.486 if hit, else -0.0157.
    # Sum over i = 0.486*h[d] + (-0.0157)*(5 - h[d]) = 0.5017*h[d] - 0.0785.
    # Divide by 5 → 0.1003*h[d] - 0.0157.
    # Pick W_o so we write approximately h[d] to dim 11+d.
    GAIN_O = 1.0 / 0.1003
    OFFSET_O = 0.0157 * GAIN_O                  # ≈ 0.156: the unwanted offset
    for d in range(10):
        blk.attn.W_o.weight.data[11 + d, d] = GAIN_O

    # ----- LN2: identity affine -----
    blk.ln2.weight.data.fill_(1.0)
    blk.ln2.bias.data.zero_()

    # ----- MLP -----
    # fc1 produces 20 hidden units:
    #   idx d (0..9):   relu(σ·(c_d - k))         [bias 0]
    #   idx 10+d:       relu(σ·(c_d - k - 1))     [bias -σ]
    # where c_d = sum_{d' ≤ d} h[d'] and k is the output index.
    #
    # The dot product weight·postLN2 = σ·(weight·raw) - σ·mean·sum(weight)
    # because postLN2 = σ·(raw - mean). To eliminate the input-dependent mean
    # term we make sum(weight) = 0 by adding a balancing weight on a "dead"
    # residual dim (dim 35, which is always raw 0). The raw value contribution
    # from that dim is 0, so it does no harm.
    #
    # The raw value at residual dim 11+d' (after attn) is approximately
    # h[d'] - OFFSET_O. Hence weight·raw on dims 11..11+d gives
    # c_d - OFFSET_O·(d+1). We absorb that constant via fc1.bias.
    for d in range(10):
        # h_d^+  at hidden idx d
        for d_prime in range(d + 1):
            blk.mlp.fc1.weight.data[d, 11 + d_prime] = 1.0
        for j in range(5):
            blk.mlp.fc1.weight.data[d, 21 + j] = -float(j)
        weight_sum = (d + 1) + sum(-j for j in range(5))   # = d - 9
        blk.mlp.fc1.weight.data[d, 35] = -weight_sum       # makes row-sum 0
        blk.mlp.fc1.bias.data[d] = SIGMA * OFFSET_O * (d + 1)

        # h_d^-  at hidden idx 10+d  (same row, smaller bias by σ)
        for d_prime in range(d + 1):
            blk.mlp.fc1.weight.data[10 + d, 11 + d_prime] = 1.0
        for j in range(5):
            blk.mlp.fc1.weight.data[10 + d, 21 + j] = -float(j)
        blk.mlp.fc1.weight.data[10 + d, 35] = -weight_sum
        blk.mlp.fc1.bias.data[10 + d] = SIGMA * OFFSET_O * (d + 1) - SIGMA

    # fc2: write (h_d^+ - h_d^-) - (h_{d-1}^+ - h_{d-1}^-) into residual dim 26+d.
    # That value is σ for the correct digit and 0 for all others.
    OUT_GAIN = 10.0 / SIGMA                     # so the residual write ≈ +10
    for d in range(10):
        blk.mlp.fc2.weight.data[26 + d, d] = +OUT_GAIN
        blk.mlp.fc2.weight.data[26 + d, 10 + d] = -OUT_GAIN
        if d >= 1:
            blk.mlp.fc2.weight.data[26 + d, d - 1] = -OUT_GAIN
            blk.mlp.fc2.weight.data[26 + d, 10 + d - 1] = +OUT_GAIN

    # ----- Final LN + Head -----
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()

    # Vocab: 0..9 are digits, 10 is '='. Read residual dim 26+d as logit for d.
    for d in range(10):
        model.head.weight.data[d, 26 + d] = 100.0
    # '=' (vocab idx 10) gets logit 0 → never wins against the (positive) digit.


model_shorthand_name = "HistogramCircuit_v1"
model_description = (
    "1-layer transformer: attention computes input-digit histogram into residual; "
    "MLP turns (histogram, k) into a one-hot of the k-th smallest digit via two "
    "ReLUs per digit (clamp(c_d-k,0,1) - clamp(c_{d-1}-k,0,1)); head reads it."
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
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=64,
        n_heads=1,
        n_layers=1,
        d_ff=64,
    )
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
