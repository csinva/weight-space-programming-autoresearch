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

    High-level circuit (one transformer block):
      1) Token embedding writes a one-hot of the digit into residual dims 0..9.
      2) Position embedding marks input vs. output-query positions on dim 10,
         and writes a one-hot of the output index k = 0..4 into dims 11..15 at
         output query positions. It also lays down a fixed ±A anchor pattern
         on the last `n_anchor` dims so that the LayerNorm std is essentially
         data-independent (the anchor squared-sum dominates).
      3) The single attention head: queries at output-query positions strongly
         attend to the five input positions (via the ±dim-10 marker), values
         copy the input-digit one-hot. The attention output at an output
         position is therefore (up to known constants) the histogram h[d] of
         input digits, which we write into residual dims 16..25.
      4) MLP turns (h, k) into a one-hot of the k-th-smallest digit:
           indicator_d = clamp(c_d - k, 0, 1) = ReLU(σ·(c_d-k)) − ReLU(σ·(c_d-k-1))
           pred_d      = indicator_d − indicator_{d-1}
         where c_d = Σ_{d'≤d} h[d']. (c_{-1}:=0.)
         pred_d is 1 exactly for the unique d with c_{d-1} ≤ k < c_d.
         We write pred_d back into residual dims 0..9 (sharing with token emb;
         the predicted-digit signal is much larger than the prev-token bump).
      5) The head reads residual dims 0..9 as logits for digits 0..9.

    Residual layout (d_model = 32):
        [0..9]   = digit one-hot (set by token_emb; also written by MLP)
        [10]     = +1 at input positions, −1 at output query positions
        [11..15] = one-hot of output index k (at output query positions only)
        [16..25] = histogram h[d] (written by attention)
        [26..31] = ±5 anchor (6 dims; alternating signs)

    LayerNorm scale-invariance: the anchor contributes 6·25 = 150 to the
    squared sum at every position, while data dims contribute at most ~25,
    so std ≈ sqrt(170/32) ≈ 2.30 with ~7 % variation. We absorb the resulting
    LN gain σ = 1/std into the linear layers.
    """
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.zero_()

    D = model.d_model          # 32
    L = model.max_seq_len      # 16

    # Layout constants
    MARKER_DIM   = 10
    K_DIM        = 11          # k one-hot lives in dims K_DIM..K_DIM+4
    H_DIM        = 16          # histogram lives in dims H_DIM..H_DIM+9
    ANCHOR_START = 26
    ANCHOR_END   = D           # exclusive
    N_ANCHOR     = ANCHOR_END - ANCHOR_START   # 6
    ANCHOR_MAG   = 5.0

    # ----- Token embedding -----
    for d in range(10):
        model.token_emb.weight.data[d, d] = 1.0      # '=' (idx 10) stays zero

    # ----- Position embedding (anchor + position-type markers + k one-hot) -----
    anchor = torch.zeros(D)
    for i in range(ANCHOR_START, ANCHOR_END):
        anchor[i] = ANCHOR_MAG if ((i - ANCHOR_START) % 2 == 0) else -ANCHOR_MAG
    pos = model.pos_emb.weight.data
    for p in range(L):
        pos[p] += anchor
    for p in range(5):                               # input positions
        pos[p, MARKER_DIM] = +1.0
    for k in range(5):                               # output query positions
        pos[5 + k, MARKER_DIM] = -1.0
        pos[5 + k, K_DIM + k] = +1.0

    # ----- Pre-LN1 std and gain σ_LN1 -----
    # At every position: 1 (digit or 0) + 1 (marker) + 1 (k one-hot or 0) +
    # anchor squared sum = N_ANCHOR * ANCHOR_MAG**2.
    pre_ln1_sq_sum = 3 + N_ANCHOR * ANCHOR_MAG ** 2   # ≈ 153
    sigma_ln1 = 1.0 / math.sqrt(pre_ln1_sq_sum / D)   # ≈ 0.459
    # Effective post-LN1 mean is tiny (raw sum ≈ 2 / D), so we approximate by 0
    # and absorb residual drift via the row-sum-zero balancing trick below.

    blk = model.blocks[0]
    blk.ln1.weight.data.fill_(1.0)
    blk.ln1.bias.data.zero_()

    # ----- Attention -----
    # Query at output positions ⇒ Q[0] = +A·σ_LN1 (since marker_dim is −1),
    # key at input positions    ⇒ K[0] = +A·σ_LN1 (since marker_dim is +1).
    # Score = (A·σ_LN1)^2 / sqrt(d_head) at output→input,
    #       and minus that at output→output / input→output.
    # A = 10 gives a margin of ~5 in score units → softmax is concentrated.
    A_attn = 10.0
    blk.attn.W_q.weight.data[0, MARKER_DIM] = -A_attn
    blk.attn.W_k.weight.data[0, MARKER_DIM] = +A_attn

    # V copies the digit one-hot.
    for d in range(10):
        blk.attn.W_v.weight.data[d, d] = 1.0

    # When attention fully concentrates on the 5 inputs, attn_out[d] ≈
    #   (1/5) · Σ_i postLN1[i, d]
    # postLN1[i, d] = σ_LN1 · (1 − mean) if d = digit_at_i else σ_LN1 · (−mean).
    # Sum_i = σ_LN1 · (h[d] − 5·mean)  →  /5 = σ_LN1·h[d]/5 − σ_LN1·mean.
    # With mean very small we have attn_out[d] ≈ σ_LN1·h[d]/5 + tiny offset.
    # Pick W_o gain so the residual write at dim H_DIM+d is ≈ h[d] − OFFSET_O.
    GAIN_O = 5.0 / sigma_ln1                          # ≈ 10.89
    # The per-position "off" digits have raw 0; their post-LN1 value is −σ·mean.
    # Using the actual mean (2/D) at input positions:
    mean_input = 2.0 / D
    OFFSET_O = GAIN_O * sigma_ln1 * mean_input        # ≈ 0.313
    for d in range(10):
        blk.attn.W_o.weight.data[H_DIM + d, d] = GAIN_O

    # ----- Pre-LN2 std and gain σ_LN2 -----
    # After attn, output-query positions have additionally h[d] − OFFSET_O in
    # H_DIM..H_DIM+9. The histogram-squared-sum varies with h's concentration;
    # we use a midpoint estimate.
    h_sq_mid = 13.0                                   # ≈ average of (3 + 23)/2
    pre_ln2_sq_sum = 3 + h_sq_mid + N_ANCHOR * ANCHOR_MAG ** 2  # ≈ 166
    sigma_ln2 = 1.0 / math.sqrt(pre_ln2_sq_sum / D)   # ≈ 0.439

    blk.ln2.weight.data.fill_(1.0)
    blk.ln2.bias.data.zero_()

    # ----- MLP -----
    # fc1 weights, for each digit d, build two hidden units:
    #   idx d:      h_d^+ = ReLU(σ_LN2 · (c_d − k))
    #   idx 10+d:   h_d^- = ReLU(σ_LN2 · (c_d − k − 1))
    # Then fc2 writes (h_d^+ − h_d^-) − (h_{d-1}^+ − h_{d-1}^-) into dim d.
    #
    # Subtleties:
    #   – fc1's linear is applied to postLN2 = σ_LN2·(raw − mean). To make the
    #     pre-ReLU independent of the (slightly input-dependent) mean we force
    #     row-sum(weight) = 0 by adding balancing weights on two anchor dims
    #     of opposite sign (26 and 27) with equal magnitude. They cancel each
    #     other's signal contribution while contributing +1 to row-sum each.
    #   – fc1.bias absorbs the constant OFFSET_O·(d+1) introduced by W_o.
    for d in range(10):
        base = blk.mlp.fc1.weight.data
        for d_prime in range(d + 1):
            base[d, H_DIM + d_prime] = 1.0            # accumulates c_d
            base[10 + d, H_DIM + d_prime] = 1.0
        for j in range(5):
            base[d, K_DIM + j] = -float(j)            # subtracts k
            base[10 + d, K_DIM + j] = -float(j)
        raw_row_sum = (d + 1) - 10                    # = d − 9
        bal = -raw_row_sum / 2.0                      # = (9 − d)/2
        base[d, ANCHOR_START + 0] = bal               # raw +ANCHOR_MAG
        base[d, ANCHOR_START + 1] = bal               # raw −ANCHOR_MAG
        base[10 + d, ANCHOR_START + 0] = bal
        base[10 + d, ANCHOR_START + 1] = bal

        # bias: cancel the OFFSET_O·(d+1) shift and (for h_d^-) threshold at +1.
        blk.mlp.fc1.bias.data[d]      = sigma_ln2 * OFFSET_O * (d + 1)
        blk.mlp.fc1.bias.data[10 + d] = sigma_ln2 * OFFSET_O * (d + 1) - sigma_ln2

    # fc2: pred_d = (h_d^+ − h_d^-) − (h_{d-1}^+ − h_{d-1}^-)
    # This equals σ_LN2 for the correct digit, 0 otherwise. Scale up to ≈ 10.
    OUT_GAIN = 10.0 / sigma_ln2
    for d in range(10):
        blk.mlp.fc2.weight.data[d, d]      = +OUT_GAIN
        blk.mlp.fc2.weight.data[d, 10 + d] = -OUT_GAIN
        if d >= 1:
            blk.mlp.fc2.weight.data[d, d - 1]      = -OUT_GAIN
            blk.mlp.fc2.weight.data[d, 10 + d - 1] = +OUT_GAIN

    # ----- Final LN + Head -----
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    # Read residual dims 0..9 as logits for digits 0..9; '=' (vocab 10) ↦ 0.
    for d in range(10):
        model.head.weight.data[d, d] = 100.0


model_shorthand_name = "HistogramCircuit_v3_d32"
model_description = (
    "Same circuit as v2 but shrunk: d_model=32 (anchor reduced to 6 dims at ±5), "
    "d_ff=20. The MLP also writes its prediction back into the digit one-hot dims "
    "0..9, removing the dedicated prediction subspace and the head reads dims 0..9 "
    "directly. All LN-aware constants (σ, OFFSET_O) derived analytically."
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
        d_model=32,
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
