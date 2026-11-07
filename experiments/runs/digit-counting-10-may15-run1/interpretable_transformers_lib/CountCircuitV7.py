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
    """Transformer block with LayerNorm replaced by identity for clean hand-built weights.

    `attn_enabled=False` makes this an MLP-only residual block (saves attention params
    when the attention layer is unused).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn_enabled: bool = True):
        super().__init__()
        self.attn_enabled = attn_enabled
        if attn_enabled:
            self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_enabled:
            x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer (no LayerNorm), vocab/seq-len configured from the task.

    `d_ff` may be a single int (shared across blocks) or a list/tuple of one int per block.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 16,
        d_model: int = 36,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff = (10, 14),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        if isinstance(d_ff, int):
            d_ff_list = [d_ff] * n_layers
        else:
            d_ff_list = list(d_ff)
            assert len(d_ff_list) == n_layers
        self.d_ff = d_ff_list

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # First block has attention; subsequent blocks are MLP-only (hand-built circuit
        # does not need attention beyond layer 0).
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff_list[i], attn_enabled=(i == 0))
            for i in range(n_layers)
        ])
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built circuit for digit-counting-10.

    Prompt: "DDDDDDDDDDQ="  (positions 0..9 = data digits, 10 = query digit, 11 = '=')
    Answer: "CC"            (positions 11→tens digit, 12→ones digit of count in 00..10)

    Residual stream layout (d_model=64):
      0-9   data digit one-hot (from token embedding)  -- also reused as OUTPUT digit slot
            (token emb sets these only at data positions and at pos 12 if first prediction
            was a digit token; L1 MLP additively writes the output digit one-hot here)
      10    '=' indicator      (from token embedding)
      11    pos in 0..9        (from position embedding — data position)
      12    pos == 10          (query position)
      13    pos == 11          (first output position; the '=' token slot)
      14    pos == 12          (second output position)
      15    count = h[query]   (written by L0 MLP)
      16-25 query digit one-hot g[d]   (written by L0 attn head 0)
      26-35 histogram h[d] = count of digit d  (written by L0 attn head 1)

    Circuit:
      L0 attn head 0: at output positions, attend hard to position 10 → broadcast query digit.
      L0 attn head 1: at output positions, attend uniformly to positions 0..9 → average data
                      digit one-hot, then scaled by 10 = full count histogram (count_0..count_9).
      L0 MLP: m_d = ReLU(h_d + ALPHA*g_d - ALPHA) selects h_query when g_d==1; sum → count.
      L1 MLP: 3-neuron bump on (count - c) gated by pos indicator → fires iff count==c AND pos==p.
              Weighted sum writes the correct output digit one-hot (tens at pos11, ones at pos12).
      Head: identity from output digit one-hot to digit token logits.
    """
    torch.manual_seed(0)

    for p in model.parameters():
        nn.init.zeros_(p)

    D = model.d_model
    H = model.n_heads
    L = model.max_seq_len
    dh = D // H

    assert D == 36 and H == 2 and dh == 18 and model.n_layers == 2

    digit_ids = [task.stoi[str(i)] for i in range(10)]
    eq_id = task.stoi["="]

    S = 10.0       # attention sharpness
    ALPHA = 20.0   # MLP gating strength

    with torch.no_grad():
        # ---- Token embedding ----
        for d, tid in enumerate(digit_ids):
            model.token_emb.weight[tid, d] = 1.0
        model.token_emb.weight[eq_id, 10] = 1.0

        # ---- Position embedding ----
        for pos in range(L):
            if pos < 10:
                model.pos_emb.weight[pos, 11] = 1.0
            elif pos == 10:
                model.pos_emb.weight[pos, 12] = 1.0
            elif pos == 11:
                model.pos_emb.weight[pos, 13] = 1.0
            elif pos == 12:
                model.pos_emb.weight[pos, 14] = 1.0
            # positions ≥ 13 are not used for prediction

        # ============================================================
        # Block 0 — attention
        # ============================================================
        b0 = model.blocks[0]

        # --- Head 0: broadcast query digit ---
        # Q[0] is large at output positions (pos==11 → dim 13, pos==12 → dim 14).
        b0.attn.W_q.weight[0, 13] = S
        b0.attn.W_q.weight[0, 14] = S
        # K[0] is large at the query position (pos==10 → dim 12).
        b0.attn.W_k.weight[0, 12] = S
        # V dims 0..9 of head 0 copy the data digit one-hot.
        for d in range(10):
            b0.attn.W_v.weight[d, d] = 1.0
        # Project head 0's V dims 0..9 to residual dims 16..25 (query digit slot).
        for d in range(10):
            b0.attn.W_o.weight[16 + d, d] = 1.0

        # --- Head 1: uniform sum over data positions → histogram ---
        # Q[dh] is large at output positions; K[dh] is large at data positions (pos<10 → dim 11).
        b0.attn.W_q.weight[dh + 0, 13] = S
        b0.attn.W_q.weight[dh + 0, 14] = S
        b0.attn.W_k.weight[dh + 0, 11] = S
        # V dims dh..dh+9 of head 1 copy the data digit one-hot.
        for d in range(10):
            b0.attn.W_v.weight[dh + d, d] = 1.0
        # 10 data positions → softmax weight ≈ 0.1 each → multiply by 10 in W_o to get counts.
        for d in range(10):
            b0.attn.W_o.weight[26 + d, dh + d] = 10.0

        # ============================================================
        # Block 0 — MLP: compute count = h[query]
        # ============================================================
        # neuron d (d=0..9): m_d = ReLU(h_d + ALPHA*g_d - ALPHA)
        #   if g_d == 1: m_d = ReLU(h_d) = h_d  (count of query digit)
        #   if g_d == 0: m_d = ReLU(h_d - ALPHA) = 0   (since h_d ≤ 10 < ALPHA)
        for d in range(10):
            b0.mlp.fc1.weight[d, 26 + d] = 1.0       # h_d
            b0.mlp.fc1.weight[d, 16 + d] = ALPHA     # g_d gate
            b0.mlp.fc1.bias[d] = -ALPHA
        # sum all m_d → residual dim 15 holds count (integer in 0..10)
        for d in range(10):
            b0.mlp.fc2.weight[15, d] = 1.0

        # ============================================================
        # Block 1 — attention is identity (all zeros, set above)
        # ============================================================
        b1 = model.blocks[1]

        # ============================================================
        # Block 1 — MLP: decode (count, pos) → output digit one-hot in dims 0..9.
        #
        # Overlap output one-hot with the data-digit-embedding dims (0..9) (safe — see V5).
        # Argmax-of-zero trick handles the "emit '0'" cases at pos 11 (count<10).
        #
        # Construction (14 neurons total):
        #   n=0          : pos==11 AND count==10 → write 4 to dim 1 (digit '1')
        #   n=1..13      : pos==12-gated SHARED step neurons: ReLU(count - k)
        #                   for k = -1, 0, 1, ..., 11  (13 neurons)
        #
        # For each digit c (c=0..9, output at pos 12), use the bump identity:
        #     indicator(count == c) = ReLU(count - (c-1)) - 2·ReLU(count - c)
        #                                + ReLU(count - (c+1))
        # Combining bumps for c=0..9 lets W2 read from shared k=-1..10 neurons.
        # For (count==10, pos 12) → digit '0', use the bump at c=10 which needs k=9..11.
        # All 13 shared k values are pos-12-gated, so they contribute 0 at pos 11.
        # ============================================================
        ALPHA_POS = 20.0

        # (1) pos 11, count==10 → '1'
        n10_p11 = 0
        b1.mlp.fc1.weight[n10_p11, 15] = 1.0
        b1.mlp.fc1.weight[n10_p11, 13] = ALPHA_POS         # pos 11 indicator
        b1.mlp.fc1.bias[n10_p11] = -9.5 - ALPHA_POS
        b1.mlp.fc2.weight[1, n10_p11] = 4.0                # ReLU output 0.5 → write 2

        # (2) Shared pos-12-gated step neurons: ReLU(count - k) for k = -1..11
        K_VALUES = list(range(-1, 12))   # 13 values
        for i, k in enumerate(K_VALUES):
            n = 1 + i                                       # neurons 1..13
            b1.mlp.fc1.weight[n, 15] = 1.0                  # count
            b1.mlp.fc1.weight[n, 14] = ALPHA_POS            # pos 12 indicator
            b1.mlp.fc1.bias[n] = -k - ALPHA_POS
            # pos12=1 → ReLU(count - k);  pos12=0 → 0.

        # Build W2 for pos-12 digit outputs using bump identities.
        # bump_c(count) = ReLU(count-(c-1)) - 2·ReLU(count-c) + ReLU(count-(c+1))
        # Scale by 2 to make indicator value = 2 (dominates the +1 token-embed contamination).
        def k_idx(k):
            return 1 + K_VALUES.index(k)

        bump_w = {-1: 2.0, 0: -4.0, 1: 2.0}   # offsets relative to c, with W2 weight
        for c in range(10):  # digit c at pos 12 when count==c
            for offset, w in bump_w.items():
                b1.mlp.fc2.weight[c, k_idx(c + offset)] += w
        # Also handle (count==10, pos 12) → digit '0' via the c=10 bump piece.
        # bump_10(count) uses ReLU(count - 9), ReLU(count - 10), ReLU(count - 11).
        for offset, w in bump_w.items():
            b1.mlp.fc2.weight[0, k_idx(10 + offset)] += w

        # ============================================================
        # Final head: dim d (0..9) → token id for digit d
        # ============================================================
        for d in range(10):
            model.head.weight[digit_ids[d], d] = 1.0
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CountCircuitV7"
model_description = (
    "V6 + per-block d_ff (10, 14) and shared step-function neurons in L1 MLP. L1 uses 13 "
    "pos-12-gated ReLU(count - k) neurons (k=-1..11), combined by W2 via bump identity to "
    "emit any digit c at pos 12 (including count=10 → '0'), plus 1 dedicated neuron for "
    "pos 11 count=10 → '1'. 14 L1 neurons total (down from 32)."
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
