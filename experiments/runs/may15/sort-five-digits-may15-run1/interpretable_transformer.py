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

    The model is one transformer block + a linear head.

    Residual layout (d_model = 24):
        [0..9]   digit one-hot (set by token_emb; ALSO where MLP writes the
                 predicted digit and where the head reads logits from)
        [10]     position marker: +1 at the 5 input positions, −1 at the 5
                 output-query positions (set by pos_emb)
        [11]     output index k: at output-query position 5+k this dim is k
                 (set by pos_emb)
        [12..21] input-digit histogram h[d] (written by attention)
        [22, 23] ±15 anchor (set by pos_emb), forcing LN std ≈ constant

    Three-stage circuit:

      (A) Attention head ─ computes histogram at output positions.
          Wq has −A at column 10, Wk has +A at column 10 → output-query
          positions Q-match input-position keys (and only those). The softmax
          (with A = 10) is effectively uniform over the 5 input positions.
          Wv copies the digit one-hot from dims 0..9, so the attention output
          at an output position is (digit count h[d] / 5) — up to a small
          mean-subtraction offset introduced by LN1. Wo writes GAIN_O × this
          back into dims 12..21. We cancel the residual offset later via the
          LN2 bias.

      (B) MLP ─ turns (h, k) into a one-hot of the answer digit.
          We compute, for each d = 0..9:
              clamp(c_d − k, 0, 1) = ReLU(σ·(c_d−k)) − ReLU(σ·(c_d−k−1))
          with c_d = Σ_{d'≤d} h[d']. The predicted-digit indicator is
              pred_d = clamp(c_d − k, 0, 1) − clamp(c_{d−1} − k, 0, 1)
          which is 1 iff d is the k-th smallest digit. Written into dims 0..9
          (with a large gain), it dominates over any prev-token one-hot that
          was already there from the autoregressive input.

      (C) Head ─ identity on the 10 prediction dims; vocab-10 ('=') has zero
          weight, so the model never emits '=' at output positions.

    LayerNorm handling:
      • The ±15 anchor on dims 22..23 contributes 2·225 = 450 to the per-token
        squared-sum, while data dims contribute at most ~45. So pre-LN std is
        ≈ sqrt(495/24) ≈ 4.54 with only a few-percent variation across inputs.
        We compute σ := 1/std analytically and absorb it into linear weights.
      • LN2.bias is used to cancel the constant offset OFFSET_O that Wo's
        finite gain leaves in the histogram dims. After LN2 the histogram
        dims read out cleanly as σ_LN2 · h[d].
      • Every fc1 row has zero weight-sum: input-dependent per-token means
        therefore drop out exactly (we get them to cancel by placing equal
        balancing weights on the two opposite-sign anchor dims).
    """
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.zero_()

    D = model.d_model                                # 24
    L = model.max_seq_len                            # 16

    # Layout
    DIGIT_LO                 = 0
    MARKER_DIM               = 10
    K_DIM                    = 11
    HIST_LO,  HIST_HI        = 12, 22
    ANCHOR_LO, ANCHOR_HI     = 22, 24
    N_ANCHOR                 = ANCHOR_HI - ANCHOR_LO    # 2
    ANCHOR_MAG               = 15.0
    A_attn                   = 10.0

    # ----- Token embedding ----------------------------------------------
    for d in range(10):
        model.token_emb.weight.data[d, DIGIT_LO + d] = 1.0   # '=' → 0

    # ----- Position embedding -------------------------------------------
    anchor = torch.zeros(D)
    for i, dim in enumerate(range(ANCHOR_LO, ANCHOR_HI)):
        anchor[dim] = ANCHOR_MAG if (i % 2 == 0) else -ANCHOR_MAG
    pos = model.pos_emb.weight.data
    for p in range(L):
        pos[p] += anchor
    for p in range(5):
        pos[p, MARKER_DIM] = +1.0
    for k in range(5):
        pos[5 + k, MARKER_DIM] = -1.0
        pos[5 + k, K_DIM] = float(k)

    # ----- Pre-LN1 std → σ_LN1 ------------------------------------------
    anchor_sq = N_ANCHOR * ANCHOR_MAG ** 2                       # 450
    sigma_ln1 = 1.0 / math.sqrt((anchor_sq + 3.0) / D)           # ≈ 0.220

    blk = model.blocks[0]
    blk.ln1.weight.data.fill_(1.0)
    blk.ln1.bias.data.zero_()

    # ----- Attention -----------------------------------------------------
    blk.attn.W_q.weight.data[0, MARKER_DIM] = -A_attn
    blk.attn.W_k.weight.data[0, MARKER_DIM] = +A_attn
    for d in range(10):
        blk.attn.W_v.weight.data[d, DIGIT_LO + d] = 1.0

    GAIN_O = 5.0 / sigma_ln1
    mean_input = 2.0 / D
    OFFSET_O = GAIN_O * sigma_ln1 * mean_input                   # small
    for d in range(10):
        blk.attn.W_o.weight.data[HIST_LO + d, d] = GAIN_O

    # ----- Pre-LN2 std → σ_LN2 ------------------------------------------
    h_sq_mid = 13.0
    sigma_ln2 = 1.0 / math.sqrt((anchor_sq + h_sq_mid + 3.0) / D)

    # LN2: use the bias to clean up the OFFSET_O introduced by Wo. After this,
    # postLN2[HIST_LO+d] ≈ σ_LN2 · h[d] at output positions (modulo the small
    # input-dependent mean, which the row-sum-zero trick will cancel).
    blk.ln2.weight.data.fill_(1.0)
    blk.ln2.bias.data.zero_()
    for d in range(10):
        blk.ln2.bias.data[HIST_LO + d] = sigma_ln2 * OFFSET_O

    # ----- MLP -----------------------------------------------------------
    # hidden idx d      :  h_d^+ ≈ ReLU(σ_LN2 · (c_d − k))
    # hidden idx 10 + d :  h_d^- ≈ ReLU(σ_LN2 · (c_d − k − 1))
    fc1_w = blk.mlp.fc1.weight.data
    fc1_b = blk.mlp.fc1.bias.data
    for d in range(10):
        for d_prime in range(d + 1):
            fc1_w[d,      HIST_LO + d_prime] = 1.0
            fc1_w[10 + d, HIST_LO + d_prime] = 1.0
        fc1_w[d,      K_DIM] = -1.0
        fc1_w[10 + d, K_DIM] = -1.0

        # row-sum-zero balancing on the two anchor dims (opposite signs ⇒
        # they cancel the *signal* contribution while flattening row-sum).
        raw_row_sum = (d + 1) - 1                                # = d
        bal = -raw_row_sum / 2.0
        fc1_w[d,      ANCHOR_LO + 0] = bal
        fc1_w[d,      ANCHOR_LO + 1] = bal
        fc1_w[10 + d, ANCHOR_LO + 0] = bal
        fc1_w[10 + d, ANCHOR_LO + 1] = bal

        # Only h_d^- needs a threshold shift of σ_LN2 (= the "−1" in c_d−k−1).
        fc1_b[10 + d] = -sigma_ln2

    # fc2: pred_d = (h_d^+ − h_d^-) − (h_{d−1}^+ − h_{d−1}^-), scaled up.
    OUT_GAIN = 10.0 / sigma_ln2
    fc2_w = blk.mlp.fc2.weight.data
    for d in range(10):
        fc2_w[DIGIT_LO + d, d]      = +OUT_GAIN
        fc2_w[DIGIT_LO + d, 10 + d] = -OUT_GAIN
        if d >= 1:
            fc2_w[DIGIT_LO + d, d - 1]      = -OUT_GAIN
            fc2_w[DIGIT_LO + d, 10 + d - 1] = +OUT_GAIN

    # ----- Final LN + Head ----------------------------------------------
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    for d in range(10):
        model.head.weight.data[d, DIGIT_LO + d] = 100.0


model_shorthand_name = "HistogramCircuit_v5_clean"
model_description = (
    "Same 24-dim histogram-circuit as v4, but the per-digit OFFSET_O drift "
    "introduced by Wo is now absorbed into ln2.bias rather than fc1.bias. "
    "Fc1 has only a single nonzero bias (the σ_LN2 threshold for h_d^-), "
    "making the MLP rows exceptionally clean. d_model=24, d_ff=20, 1 head, "
    "1 layer."
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
        d_model=24,
        n_heads=1,
        n_layers=1,
        d_ff=20,
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
