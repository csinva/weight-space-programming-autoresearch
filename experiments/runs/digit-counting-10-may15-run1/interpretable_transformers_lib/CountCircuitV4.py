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
    """Transformer block with LayerNorm replaced by identity for clean hand-built weights."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer (no LayerNorm), vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 16,
        d_model: int = 48,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 31,
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
      0-9   data digit one-hot (from token embedding)
      10    '=' indicator      (from token embedding)
      11    pos in 0..9        (from position embedding — data position)
      12    pos == 10          (query position)
      13    pos == 11          (first output position; the '=' token slot)
      14    pos == 12          (second output position)
      15    count = h[query]   (written by L0 MLP)
      16-25 query digit one-hot g[d]   (written by L0 attn head 0)
      26-35 histogram h[d] = count of digit d  (written by L0 attn head 1)
      36-45 output digit one-hot  (written by L1 MLP; final head reads from here)

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

    assert D == 48 and H == 2 and dh == 24 and model.n_layers == 2

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
        # Block 1 — MLP: decode (count, pos) → output digit one-hot.
        #
        # Optimization: token '0' has id 0 in the vocab, so when ALL logits are 0
        # argmax returns '0' by default. This lets us *skip* writing any indicator
        # for the cases that emit '0':
        #   - pos==11, count in 0..9   (tens digit = '0')
        #   - pos==12, count == 10     (ones digit = '0')
        #
        # Remaining cases:
        #   - pos==11, count == 10        → emit '1'   (1 neuron: only c=10 matters)
        #   - pos==12, count == c (c=0..9) → emit 'c'   (3-neuron bumps × 10 = 30 neurons)
        # ============================================================
        ALPHA_POS = 20.0
        # --- pos 11, count == 10 → digit '1' ---
        # neuron_p11_c10 = ReLU(count - 9.5 + ALPHA_POS*pos11 - ALPHA_POS)
        #   pos11=1, count=10: ReLU(0.5) = 0.5  → multiply by 2 in W2.
        #   pos11=1, count≤9:  ReLU(≤-0.5) = 0.
        #   pos11=0:           ReLU(count - 9.5 - ALPHA_POS) = 0 (since count ≤ 10 ≪ ALPHA_POS+9.5).
        n10 = 0
        b1.mlp.fc1.weight[n10, 15] = 1.0
        b1.mlp.fc1.weight[n10, 13] = ALPHA_POS       # pos 11 indicator
        b1.mlp.fc1.bias[n10] = -9.5 - ALPHA_POS
        b1.mlp.fc2.weight[36 + 1, n10] = 2.0          # write to digit '1' slot

        # --- pos 12, count == c (c=0..9) → digit c ---
        # 3-neuron bump on count gated by pos 12. Same construction as V2 but only c=0..9.
        offsets = [1.0, 0.0, -1.0]
        weights = [1.0, -2.0, 1.0]
        for c in range(10):   # c = 0..9 only; count==10 falls through to default '0'
            for k in range(3):
                n = 1 + c * 3 + k   # neurons 1..30
                b1.mlp.fc1.weight[n, 15] = 1.0
                b1.mlp.fc1.weight[n, 14] = ALPHA_POS   # pos 12 indicator
                b1.mlp.fc1.bias[n] = -c + offsets[k] - ALPHA_POS
                b1.mlp.fc2.weight[36 + c, n] = weights[k]

        # ============================================================
        # Final head: dim (36+d) → token id for digit d
        # ============================================================
        for d in range(10):
            model.head.weight[digit_ids[d], 36 + d] = 1.0
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CountCircuitV4"
model_description = (
    "V3 with d_ff trimmed to exact minimum (31). L0 MLP uses 10 of 31 slots (count "
    "extraction); L1 MLP uses all 31 slots (1 for pos11 count=10, 30 for pos12 bumps)."
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
