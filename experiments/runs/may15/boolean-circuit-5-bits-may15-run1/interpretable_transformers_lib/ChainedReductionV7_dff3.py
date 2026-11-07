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
        # LayerNorm replaced with Identity so the hand-built residual stream
        # passes through unchanged — no normalization to fight against.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MLPOnlyBlock(nn.Module):
    """A residual block with only an MLP and no attention. Used when an
    attention sub-layer would be set to zero anyway — drops 4*d_model^2
    params per layer."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ln = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class SimpleTransformer(nn.Module):
    """Hand-built transformer for boolean-circuit-5-bits.

    Architecture: a single attention+MLP block (layer 0) followed by
    `n_mlp_only_blocks` MLP-only residual blocks. No LayerNorm anywhere.
    The first block's attention does a one-shot "gather" of all prompt-token
    features into pos-9 slots; each block's MLP then runs one reduction step
    of the left-to-right boolean evaluation.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 40,
        n_heads: int = 10,
        n_mlp_only_blocks: int = 3,
        d_ff: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_mlp_only_blocks = n_mlp_only_blocks
        self.n_layers = 1 + n_mlp_only_blocks
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff)]
            + [MLPOnlyBlock(d_model, d_ff) for _ in range(n_mlp_only_blocks)]
        )
        self.final_ln = nn.Identity()
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
    """Compact hand-built circuit for 5-bit / 4-op boolean expressions.

    Same chained-reduction strategy as V1 but with d_model=48, d_head=4,
    d_ff=8 — every channel is used.

    Residual channel layout (d_model = 48):
      dims 0..35   — head 0..8 attention-output slots (4 dims each).
                     head h handles prompt position h; within head h:
                       dim 0 = bit_value(=is_one of token)
                       dim 1 = is_AND, dim 2 = is_OR, dim 3 = is_XOR
                     Prompt positions: 0=b0, 1=op0, 2=b1, 3=op1, 4=b2,
                                        5=op2, 6=b3, 7=op3, 8=b4.
      dims 36..39  — running accumulators acc_1..acc_4 (head 9 slot).
      dims 40..43  — token-feature embedding:
                       '1'  → dim 40 = 1
                       '&'  → dim 41 = 1
                       '|'  → dim 42 = 1
                       '^'  → dim 43 = 1
                       '0','=' → all zero
      dims 44..47  — position embedding as 4-bit ±1 encoding (so pos_emb
                     has unit cost vs 10-dim one-hot). pos i is encoded
                     as binary(i)=(b3,b2,b1,b0) → values 2*b - 1 ∈ {-1,+1}.

    Attention pattern: head h's Q at pos 9 ∝ pos_emb[h], K[pos i] = pos_emb[i].
    Dot product = 4 - 2*hamming(h,i), scaled by SCALE for sharp softmax.

    MLP recipe per step: 6 ReLU hidden units, one per (op, prev_acc, bit)
    case where the boolean result is 1 (see V1 docstring).
    """
    torch.manual_seed(0)

    D = model.d_model       # 40
    H = model.n_heads       # 10
    dh = D // H             # 4
    F = model.d_ff          # 4
    T = model.max_seq_len   # 10
    V = task.vocab_size     # 6

    assert D == 40 and H == 10 and dh == 4 and model.n_layers == 4 and F == 3
    assert V == 6
    # vocab: "01&|^=" => idx 0:'0', 1:'1', 2:'&', 3:'|', 4:'^', 5:'='

    # ---------------- Channel layout ----------------
    # Dims 0..3  : slot 0 (head 0). ALSO the token_emb feature dims
    #              (is_one, is_AND, is_OR, is_XOR). Free overlap: at pos 0 the
    #              token_emb already places b0 in dim 0 — head 0's attention
    #              gathers it again into dim 0 at pos 9 (the same channel).
    # Dims 4..7  : slot 1 (head 1) — op0 features at pos 9.
    # Dims 8..11 : slot 2 (head 2) — b1.
    # Dims 12..15: slot 3 (head 3) — op1.
    # ... (alternating bits/ops)
    # Dims 32..35: slot 8 (head 8) — b4.
    # Dims 36..39: pos_emb (4-bit ±1 encoding); head 9 is unused.
    # ACC_DIM = 0 (shared accumulator).

    # ---------------- Embeddings ----------------
    token_emb = torch.zeros(V, D)
    token_emb[1, 0] = 1.0  # '1' is_one
    token_emb[2, 1] = 1.0  # '&' is_AND
    token_emb[3, 2] = 1.0  # '|' is_OR
    token_emb[4, 3] = 1.0  # '^' is_XOR
    model.token_emb.weight.data.copy_(token_emb)

    POS_BASE = 36
    pos_emb = torch.zeros(T, D)
    for i in range(T):
        for k in range(4):
            bit = (i >> (3 - k)) & 1   # k=0 -> high bit, k=3 -> low bit
            pos_emb[i, POS_BASE + k] = 1.0 if bit == 1 else -1.0
    model.pos_emb.weight.data.copy_(pos_emb)

    # ---------------- Layer 1: gather + step 1 ----------------
    SCALE = 8.0
    block1 = model.blocks[0]

    Wq = torch.zeros(D, D)
    Wk = torch.zeros(D, D)
    Wv = torch.zeros(D, D)

    pos9 = pos_emb[9, POS_BASE:POS_BASE + 4].clone()   # ±1 pattern, shape (4,)

    for h in range(9):
        target = pos_emb[h, POS_BASE:POS_BASE + 4]
        for k in range(4):
            # Q at pos 9, head h, k = SCALE * target[k] (read off the high bit
            # of pos_emb which is +1 at pos 9 → pos9[0] == +1).
            Wq[h * dh + k, POS_BASE] = SCALE * target[k].item() / pos9[0].item()
        for k in range(4):
            Wk[h * dh + k, POS_BASE + k] = 1.0

        # V copies token features → slot's 4 dims.
        Wv[h * dh + 0, 0] = 1.0  # is_one  → slot bit_value
        Wv[h * dh + 1, 1] = 1.0  # is_AND
        Wv[h * dh + 2, 2] = 1.0  # is_OR
        Wv[h * dh + 3, 3] = 1.0  # is_XOR

    block1.attn.W_q.weight.data.copy_(Wq)
    block1.attn.W_k.weight.data.copy_(Wk)
    block1.attn.W_v.weight.data.copy_(Wv)
    block1.attn.W_o.weight.data.copy_(torch.eye(D))

    # ---------------- Layers 2..4: attention is a no-op ----------------
    # (MLPOnlyBlock layers — no attention sub-module to zero.)

    # ---------------- MLP step builder ----------------
    # 3 ReLU hidden units per layer (d_ff=3). Each adds a signed delta into
    # ACC_DIM via the residual.
    #   u0 (+1):  fires for op∈{OR,XOR} when acc=0, bit=1     (merge of two cases)
    #   u1 (-1):  fires for op=AND       when acc=1, bit=0
    #   u2 (-1):  fires for op=XOR       when acc=1, bit=1
    # Hidden formula:
    #   pre = read_op_terms + s_a*acc + s_b*bit + (1-a) + (1-b) - 2.5
    # firing produces 0.5; fc2 routes 2*sgn to ACC_DIM → delta = sgn.

    def bit_dim(prompt_pos):
        return prompt_pos * dh + 0
    def op_dims(prompt_pos):
        base = prompt_pos * dh
        return base + 1, base + 2, base + 3   # (AND, OR, XOR)

    ACC_DIM = 0

    def set_step_mlp(block, op_and_dim, op_or_dim, op_xor_dim, bit_in_dim):
        fc1_w = torch.zeros(F, D)
        fc1_b = torch.zeros(F)
        fc2_w = torch.zeros(D, F)
        fc2_b = torch.zeros(D)

        # u0: +1, op ∈ {OR, XOR}, target acc=0, bit=1
        a, b = 0, 1
        s_a = +1.0 if a == 1 else -1.0
        s_b = +1.0 if b == 1 else -1.0
        fc1_w[0, op_or_dim] = 1.0
        fc1_w[0, op_xor_dim] = 1.0
        fc1_w[0, ACC_DIM] = s_a
        fc1_w[0, bit_in_dim] = s_b
        fc1_b[0] = -2.5 + (1 - a) + (1 - b)
        fc2_w[ACC_DIM, 0] = +2.0

        # u1: -1, op = AND, target acc=1, bit=0
        a, b = 1, 0
        s_a, s_b = +1.0, -1.0
        fc1_w[1, op_and_dim] = 1.0
        fc1_w[1, ACC_DIM] = s_a
        fc1_w[1, bit_in_dim] = s_b
        fc1_b[1] = -2.5 + (1 - a) + (1 - b)
        fc2_w[ACC_DIM, 1] = -2.0

        # u2: -1, op = XOR, target acc=1, bit=1
        a, b = 1, 1
        s_a, s_b = +1.0, +1.0
        fc1_w[2, op_xor_dim] = 1.0
        fc1_w[2, ACC_DIM] = s_a
        fc1_w[2, bit_in_dim] = s_b
        fc1_b[2] = -2.5 + (1 - a) + (1 - b)
        fc2_w[ACC_DIM, 2] = -2.0

        block.mlp.fc1.weight.data.copy_(fc1_w)
        block.mlp.fc1.bias.data.copy_(fc1_b)
        block.mlp.fc2.weight.data.copy_(fc2_w)
        block.mlp.fc2.bias.data.copy_(fc2_b)

    a0, o0, x0 = op_dims(1)
    set_step_mlp(model.blocks[0], a0, o0, x0, bit_dim(2))
    a1, o1, x1 = op_dims(3)
    set_step_mlp(model.blocks[1], a1, o1, x1, bit_dim(4))
    a2, o2, x2 = op_dims(5)
    set_step_mlp(model.blocks[2], a2, o2, x2, bit_dim(6))
    a3, o3, x3 = op_dims(7)
    set_step_mlp(model.blocks[3], a3, o3, x3, bit_dim(8))

    # ---------------- Output head ----------------
    # At pos 9, dim ACC_DIM=0 holds acc_4. pos_emb[9, 39]=+1 (low bit of 9 is 1)
    # is used as the always-on baseline for token '0'.
    head_w = torch.zeros(V, D)
    head_w[0, ACC_DIM] = -10.0
    head_w[0, 39] = 0.5
    head_w[1, ACC_DIM] = 10.0
    model.head.weight.data.copy_(head_w)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ChainedReductionV7_dff3"
model_description = (
    "Same as V6 but d_ff drops 4→3 by merging the two '+1 delta' truth-table "
    "cases (OR with acc=0,bit=1 and XOR with acc=0,bit=1) into a single ReLU "
    "unit that reads (read_or + read_xor)."
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
    max_seq_len = task.seq_len
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
