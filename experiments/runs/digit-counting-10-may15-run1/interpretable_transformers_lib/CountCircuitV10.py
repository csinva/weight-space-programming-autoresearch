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
    """Causal multi-head attention with independent Q/K and V dimensions per head.

    Standard MHA constrains the per-head Q, K, V dimensions to be equal (d_head).
    Here we allow `d_qk` (per-head Q/K dim) and `d_v` (per-head V dim) to differ.
    Hand-built circuits often use a tiny `d_qk` (just enough to mark positions) but
    need a wider `d_v` (e.g., to copy a digit one-hot).  Splitting the two saves
    parameters wasted on unused Q/K columns.

    `d_attn` (legacy) is interpreted as the total V dimension across heads when
    `d_v` is not given (so d_v = d_attn / n_heads).
    """

    def __init__(self, d_model: int, n_heads: int, d_attn: int = None,
                 d_qk: int = None, d_v: int = None):
        super().__init__()
        if d_v is None:
            assert d_attn is not None and d_attn % n_heads == 0
            d_v = d_attn // n_heads
        if d_qk is None:
            d_qk = d_v
        self.n_heads = n_heads
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_model = d_model
        # Total Q/K dim = n_heads * d_qk; total V dim = n_heads * d_v.
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dqk, dv = self.n_heads, self.d_qk, self.d_v
        q = self.W_q(x).view(B, T, H, dqk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dqk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dqk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dv)
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

    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn_enabled: bool = True,
                 d_attn: int = None, d_qk: int = None, d_v: int = None):
        super().__init__()
        self.attn_enabled = attn_enabled
        if attn_enabled:
            self.attn = CausalSelfAttention(d_model, n_heads, d_attn=d_attn,
                                            d_qk=d_qk, d_v=d_v)
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
        d_model: int = 35,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff = (10, 12),
        d_attn: int = None,
        d_qk: int = 1,
        d_v: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_attn = d_attn
        self.d_qk = d_qk
        self.d_v = d_v
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
            Block(d_model, n_heads, d_ff_list[i], attn_enabled=(i == 0),
                  d_attn=d_attn, d_qk=d_qk, d_v=d_v)
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
    d_qk = model.d_qk
    d_v = model.d_v

    assert D == 35 and H == 2 and d_qk == 1 and d_v == 10 and model.n_layers == 2

    digit_ids = [task.stoi[str(i)] for i in range(10)]
    eq_id = task.stoi["="]

    S = 10.0       # attention sharpness
    ALPHA = 20.0   # MLP gating strength

    # Residual stream layout (d_model=35):
    #   0-9   data digit one-hot (token emb) — ALSO output digit slot (L1 MLP writes here)
    #   10    '=' token embed — doubles as pos-11 indicator (only at pos 11 is dim 10 = 1)
    #   11    pos in 0..9   (data position)
    #   12    pos == 10     (query position)
    #   13    pos == 12     (second output position)
    #   14    count         (L0 MLP)
    #   15-24 query digit one-hot g[d]    (L0 attn head 0)
    #   25-34 histogram h[d]              (L0 attn head 1, scaled ×10)

    POS_DATA = 11
    POS_QUERY = 12
    POS_OUT2 = 13           # pos 12 indicator (second output position)
    POS_OUT1 = 10           # pos 11 indicator (the '=' token embed dim — only '=' at pos 11)
    COUNT = 14
    G0 = 15                 # query one-hot start
    H0 = 25                 # histogram start

    with torch.no_grad():
        # ---- Token embedding ----
        for d, tid in enumerate(digit_ids):
            model.token_emb.weight[tid, d] = 1.0
        model.token_emb.weight[eq_id, POS_OUT1] = 1.0   # '=' token sets dim 10

        # ---- Position embedding ----
        for pos in range(L):
            if pos < 10:
                model.pos_emb.weight[pos, POS_DATA] = 1.0
            elif pos == 10:
                model.pos_emb.weight[pos, POS_QUERY] = 1.0
            elif pos == 12:
                model.pos_emb.weight[pos, POS_OUT2] = 1.0
            # pos 11 needs no pos-emb dim: dim 10 from token '=' already marks it.

        # ============================================================
        # Block 0 — attention
        # ============================================================
        b0 = model.blocks[0]

        # --- Head 0: broadcast query digit to output positions ---
        # Q[0] high at output positions: at pos 11 dim 10 = 1, at pos 12 dim 13 = 1.
        b0.attn.W_q.weight[0, POS_OUT1] = S
        b0.attn.W_q.weight[0, POS_OUT2] = S
        # K[0] high at the query position (pos==10 → dim 12).
        b0.attn.W_k.weight[0, POS_QUERY] = S
        # V dims 0..9 of head 0 copy the data digit one-hot from residual dims 0..9.
        for d in range(10):
            b0.attn.W_v.weight[d, d] = 1.0
        # Project head 0's V dims 0..9 into residual dims G0..G0+9 (query digit slot).
        for d in range(10):
            b0.attn.W_o.weight[G0 + d, d] = 1.0

        # --- Head 1: uniform sum over data positions → histogram ---
        b0.attn.W_q.weight[d_qk + 0, POS_OUT1] = S
        b0.attn.W_q.weight[d_qk + 0, POS_OUT2] = S
        b0.attn.W_k.weight[d_qk + 0, POS_DATA] = S
        for d in range(10):
            b0.attn.W_v.weight[d_v + d, d] = 1.0
        # 10 data positions → softmax weight ≈ 0.1 each; ×10 in W_o gives integer counts.
        for d in range(10):
            b0.attn.W_o.weight[H0 + d, d_v + d] = 10.0

        # ============================================================
        # Block 0 — MLP: count = h[query]
        # ============================================================
        # neuron d (d=0..9): m_d = ReLU(h_d + ALPHA*g_d - ALPHA)
        for d in range(10):
            b0.mlp.fc1.weight[d, H0 + d] = 1.0
            b0.mlp.fc1.weight[d, G0 + d] = ALPHA
            b0.mlp.fc1.bias[d] = -ALPHA
        for d in range(10):
            b0.mlp.fc2.weight[COUNT, d] = 1.0

        # ============================================================
        # Block 1 — attention disabled (attn_enabled=False)
        # Block 1 — MLP: (count, pos) → output digit one-hot in dims 0..9
        # 12 neurons total:
        #   n=0       : pos 11 AND count==10 → write 4 to dim 1 (digit '1')
        #   n=1..11   : pos-12-gated step neurons ReLU(count - k) for k = 0..10
        #
        # For pos 12 digit c (c=1..9), use bump identity:
        #     indicator(count==c) = step(c-1) - 2·step(c) + step(c+1)
        # Default-zero handles c=0 (residual dim 0 already = 1 from prev token '0').
        # For (count==10, pos 12) → digit '0', use bump at c=10 = step(9) - 2·step(10).
        # ============================================================
        b1 = model.blocks[1]
        ALPHA_POS = 20.0

        # (1) pos 11, count==10 → '1'.  ReLU(count - 9 + ALPHA*pos_out1 - ALPHA - 0.5)
        # At pos 11 (POS_OUT1=1): ReLU(count - 9.5). count<10 → 0; count=10 → 0.5.
        # At pos 12 (POS_OUT1=0): ReLU(count - 9.5 - ALPHA) = 0.
        n10_p11 = 0
        b1.mlp.fc1.weight[n10_p11, COUNT] = 1.0
        b1.mlp.fc1.weight[n10_p11, POS_OUT1] = ALPHA_POS
        b1.mlp.fc1.bias[n10_p11] = -9.5 - ALPHA_POS
        b1.mlp.fc2.weight[1, n10_p11] = 4.0     # 0.5 × 4 = 2 (beats residual contamination)

        # (2) Shared pos-12-gated step neurons: ReLU(count - k) for k = 0..10.
        K_VALUES = list(range(0, 11))   # 11 values
        for i, k in enumerate(K_VALUES):
            n = 1 + i                                    # neurons 1..11
            b1.mlp.fc1.weight[n, COUNT] = 1.0
            b1.mlp.fc1.weight[n, POS_OUT2] = ALPHA_POS
            b1.mlp.fc1.bias[n] = -k - ALPHA_POS

        def k_idx(k):
            return 1 + K_VALUES.index(k)

        # bump_c(count) = step(c-1) - 2*step(c) + step(c+1)
        # Skip c=0 (default-zero handles it via residual contamination from prev token '0').
        # Scale ×2 so the indicator dominates the +1 from token-embed contamination.
        bump_w = {-1: 2.0, 0: -4.0, 1: 2.0}
        for c in range(1, 10):
            for offset, w in bump_w.items():
                k = c + offset
                if k in K_VALUES:
                    b1.mlp.fc2.weight[c, k_idx(k)] += w
        # (count==10, pos 12) → digit '0': bump_10 uses step(9), step(10), step(11)=0.
        for offset, w in bump_w.items():
            k = 10 + offset
            if k in K_VALUES:
                b1.mlp.fc2.weight[0, k_idx(k)] += w

        # ============================================================
        # Final head: residual dim d (0..9) → token id for digit d
        # ============================================================
        for d in range(10):
            model.head.weight[digit_ids[d], d] = 1.0
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CountCircuitV10"
model_description = (
    "V9 + asymmetric attention: per-head d_qk=1 (we only need a 1-D score per head to mark "
    "the target position) while keeping d_v=10 (for copying a 10-dim data-digit one-hot). "
    "Saves ~1.3k attention params vs symmetric d_qk=d_v=10. Same circuit, smaller Q/K matrices."
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
