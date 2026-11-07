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

    The model is a single transformer block plus a linear head. The circuit
    is divided into three stages.

    Stage 1 — embeddings (write the residual stream layout):
        Residual dims (d_model = 24):
          [0..9]   digit one-hot (token_emb sets one of these to +1; '=' = 0)
          [10]     position marker: +1 at the 5 input positions, −1 at the
                   5 output-query positions 5..9 (pos_emb)
          [11]     output index k: at output-query position 5+k this dim is k
                   (pos_emb); zero everywhere else
          [12..21] will hold the input-digit histogram (set later by attn)
          [22..23] ±15 anchor (two dims), forcing pre-LN std to be ≈ constant
                   across inputs so we can analytically pre-compute LN's gain σ.

    Stage 2 — attention head (compute histogram at output positions):
        Wq has −A in column 10, Wk has +A in column 10 → output-query
        positions have Q[0] ≈ +A·σ·1, input positions have K[0] ≈ +A·σ·1,
        all other scores are negative. With A = 10 the softmax fully
        concentrates on the 5 input positions, so the attention output is
        the *mean* of the input-position values. Wv copies the digit one-hot
        from dims 0..9, so the per-dim mean is (digit count / 5) up to the
        usual post-LN scaling. Wo amplifies this and writes it into the
        histogram dims 12..21 as h[d] − OFFSET_O (per-d offset compensated by
        fc1.bias below).

    Stage 3 — MLP (turn histogram + k into a one-hot of the answer digit):
        For each digit d we build two ReLU units that approximate
            clamp(c_d − k, 0, 1) = ReLU(σ·(c_d−k)) − ReLU(σ·(c_d−k−1))
        with c_d = Σ_{d'≤d} h[d']. The predicted-digit indicator is
            pred_d = clamp(c_d − k, 0, 1) − clamp(c_{d−1} − k, 0, 1)
        which equals 1 exactly when d is the k-th smallest digit and 0
        otherwise. We write pred_d into residual dims 0..9 (overlapping the
        token-emb one-hot for the previously-output digit; the pred signal
        is ≈ 10× larger so the head still argmaxes correctly).

    LayerNorm handling:
        Anchor dominates the squared sum at every position, so pre-LN std is
        ≈ sqrt((data_sq + 450) / 24) ≈ 4.4 with small variation. To make the
        MLP independent of the (slightly input-dependent) per-token mean
        introduced by LN, every fc1 row has row-sum-zero — we add two
        balancing weights on the anchor dims (which have opposite raw signs)
        so the balance contributes 0 to the signal but cancels the mean term.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.zero_()

    D = model.d_model              # 24
    L = model.max_seq_len          # 16

    # Layout constants ---------------------------------------------------
    DIGIT_LO, DIGIT_HI       = 0, 10
    MARKER_DIM               = 10
    K_DIM                    = 11
    HIST_LO,  HIST_HI        = 12, 22
    ANCHOR_LO, ANCHOR_HI     = 22, 24
    N_ANCHOR                 = ANCHOR_HI - ANCHOR_LO        # 2
    ANCHOR_MAG               = 15.0
    A_attn                   = 10.0

    # ----- Token embedding ------------------------------------------------
    for d in range(10):
        model.token_emb.weight.data[d, DIGIT_LO + d] = 1.0  # '=' (vocab 10) → 0

    # ----- Position embedding (markers, k, anchor) -----------------------
    anchor = torch.zeros(D)
    for i, dim in enumerate(range(ANCHOR_LO, ANCHOR_HI)):
        anchor[dim] = ANCHOR_MAG if (i % 2 == 0) else -ANCHOR_MAG

    pos = model.pos_emb.weight.data
    for p in range(L):
        pos[p] += anchor
    for p in range(5):                                       # input positions
        pos[p, MARKER_DIM] = +1.0
    for k in range(5):                                       # output queries
        pos[5 + k, MARKER_DIM] = -1.0
        pos[5 + k, K_DIM] = float(k)

    # ----- Pre-LN1 std and σ_LN1 -----------------------------------------
    anchor_sq = N_ANCHOR * ANCHOR_MAG ** 2                   # = 450
    # Data dims contribute at most ~ 1 (digit) + 1 (marker) + 16 (k^2 ≤ 16).
    data_sq_typ = 3.0                                        # rough midpoint
    sigma_ln1 = 1.0 / math.sqrt((anchor_sq + data_sq_typ) / D)

    blk = model.blocks[0]
    blk.ln1.weight.data.fill_(1.0)
    blk.ln1.bias.data.zero_()

    # ----- Attention ------------------------------------------------------
    blk.attn.W_q.weight.data[0, MARKER_DIM] = -A_attn
    blk.attn.W_k.weight.data[0, MARKER_DIM] = +A_attn
    for d in range(10):
        blk.attn.W_v.weight.data[d, DIGIT_LO + d] = 1.0      # copy digit one-hot

    # When attention is fully on inputs: attn_out[d] ≈ σ_LN1 · h[d] / 5
    # (the −σ·mean offset is small; we absorb it via fc1.bias).
    GAIN_O = 5.0 / sigma_ln1
    mean_input = 2.0 / D                                     # raw sum at input pos / D
    OFFSET_O = GAIN_O * sigma_ln1 * mean_input               # constant write offset
    for d in range(10):
        blk.attn.W_o.weight.data[HIST_LO + d, d] = GAIN_O

    # ----- Pre-LN2 std and σ_LN2 -----------------------------------------
    h_sq_mid = 13.0                                          # midpoint of [3, 23]
    pre_ln2_sq = anchor_sq + h_sq_mid + 3                    # +1 for marker, k, prev-token bump
    sigma_ln2 = 1.0 / math.sqrt(pre_ln2_sq / D)

    blk.ln2.weight.data.fill_(1.0)
    blk.ln2.bias.data.zero_()

    # ----- MLP ------------------------------------------------------------
    # 20 hidden units split into two halves:
    #   idx d        :  h_d^+ ≈ ReLU(σ·(c_d − k))
    #   idx 10 + d   :  h_d^- ≈ ReLU(σ·(c_d − k − 1))
    fc1_w = blk.mlp.fc1.weight.data
    fc1_b = blk.mlp.fc1.bias.data
    for d in range(10):
        for d_prime in range(d + 1):
            fc1_w[d,      HIST_LO + d_prime] = 1.0           # accumulates c_d
            fc1_w[10 + d, HIST_LO + d_prime] = 1.0
        fc1_w[d,      K_DIM] = -1.0                          # subtracts k
        fc1_w[10 + d, K_DIM] = -1.0

        # Make the row weight sum = 0 by adding balancing weights on the two
        # anchor dims (which have opposite raw signs ⇒ no signal contribution).
        raw_row_sum = (d + 1) - 1                            # = d  (no k-one-hot any more)
        bal = -raw_row_sum / 2.0
        fc1_w[d,      ANCHOR_LO + 0] = bal
        fc1_w[d,      ANCHOR_LO + 1] = bal
        fc1_w[10 + d, ANCHOR_LO + 0] = bal
        fc1_w[10 + d, ANCHOR_LO + 1] = bal

        # Cancel the OFFSET_O·(d+1) drift and threshold h_d^- at +1.
        fc1_b[d]      = sigma_ln2 * OFFSET_O * (d + 1)
        fc1_b[10 + d] = sigma_ln2 * OFFSET_O * (d + 1) - sigma_ln2

    # fc2: pred_d = (h_d^+ − h_d^-) − (h_{d−1}^+ − h_{d−1}^-).  Scale up.
    OUT_GAIN = 10.0 / sigma_ln2
    fc2_w = blk.mlp.fc2.weight.data
    for d in range(10):
        fc2_w[DIGIT_LO + d, d]      = +OUT_GAIN
        fc2_w[DIGIT_LO + d, 10 + d] = -OUT_GAIN
        if d >= 1:
            fc2_w[DIGIT_LO + d, d - 1]      = -OUT_GAIN
            fc2_w[DIGIT_LO + d, 10 + d - 1] = +OUT_GAIN

    # ----- Final LN + Head -----------------------------------------------
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    for d in range(10):
        model.head.weight.data[d, DIGIT_LO + d] = 100.0      # '=' (vocab 10) → 0


model_shorthand_name = "HistogramCircuit_v4_d24_kscalar"
model_description = (
    "Same circuit as v3 but k is stored as a scalar in a single residual dim "
    "(not a 5-dim one-hot). Anchor reduced to 2 dims at ±15 (still dominates "
    "variance). d_model=24, d_ff=20, n_heads=1, n_layers=1."
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
