"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition:
prompt "12345+67890=" -> answer "080235".
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
from src.task import get_task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (LayerNorm removed for clean hand-built weights)
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
    """LayerNorm-free transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 256,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 512,
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
# Hand-built weights: full 5-digit addition with carry propagation.
# ---------------------------------------------------------------------------
#
# Vocab "0123456789+=": digits 0..9, '+' = 10, '=' = 11.
# Prompt positions: 0..4 = A digits (MSD-first), 5 = '+', 6..10 = B digits,
# 11 = '='. Predictions live at positions 11..16 (logits[:, -1, :]).
#
# Long-addition spec (positions counted MSD-first):
#   carry_5 = 0
#   for k in 4, 3, 2, 1, 0:
#       d_k = A[k] + B[k] + carry_{k+1}
#       S_{k+1} = d_k mod 10
#       carry_k = d_k // 10
#   S_0 = carry_0
#
# Block 0 ("gather and compute candidates"):
#   - 10 attention heads each gather one input digit (A_0..A_4, B_0..B_4)
#     into a fixed 10-dim slot of the residual. Plus a "self-sink" head
#     for the '=' position so nothing leaks where queries are inactive.
#   - MLP computes, for each digit position i in 0..4:
#       g_i  = 1[A_i + B_i >= 10]              ("generate" carry)
#       p_i  = 1[A_i + B_i == 9]               ("propagate" carry)
#       S_{i+1}^0 = one-hot of (A_i + B_i) mod 10        (no incoming carry)
#       S_{i+1}^1 = one-hot of (A_i + B_i + 1) mod 10    (incoming carry = 1)
#     using one ReLU AND-unit per (i, a, b) triple = 500 units.
#
# Block 1 ("carry lookahead"):
#   - Attention zeroed (W_o = 0).
#   - MLP computes carries via the disjoint carry-lookahead expansion:
#       c_4 = g_4
#       c_k = g_k v (p_k & g_{k+1}) v (p_k & p_{k+1} & g_{k+2}) v ...
#     Each path is a ReLU(sum - count + 1) AND of binary g/p features.
#     Paths through any given digit position are mutually exclusive
#     (g_j and p_j cannot both fire), so summing them computes OR exactly.
#     Total: 1+2+3+4+5 = 15 hidden units.
#
# Block 2 ("position-select + carry multiplex"):
#   - Attention zeroed.
#   - MLP fires (pos, digit, carry) AND-units to write the final answer
#     digit one-hot to a fixed 10-dim slot of the residual:
#       pos=11: digit = c_0      (= S_0 leading carry)         2 units
#       pos=12..15: digit = S_{k}^{c_k} for k = p - 11      4 * 20 units
#       pos=16: digit = S_5^0   (no incoming carry to LSD)   10 units
#
# Head: project the final-digit slot dims to the 10 digit-token logits.
# ---------------------------------------------------------------------------

# Residual stream layout (d_model = 256):
POS_OH        = (0,   18)    # position one-hot, dims 0..17
TOK_OH        = (18,  30)    # token-digit one-hot, dims 18..29 (only 18..27 used)
SLOT_BASE     = 30           # slot s in 0..9 -> dims [30+10s, 30+10(s+1))
G_BASE        = 130          # g_0..g_4 in dims 130..134
P_BASE        = 135          # p_0..p_4 in dims 135..139
S0_BASE       = 140          # S_{i+1}^0 in dims 140 + 10*i (i=0..4)
S1_BASE       = 190          # S_{i+1}^1 in dims 190 + 10*i (i=0..3)
C_BASE        = 230          # c_0..c_4 in dims 230..234
OUT_BASE      = 235          # final output digit one-hot, dims 235..244

SCALE = 50.0


def _slot_dim(slot: int, i: int) -> int:
    """Residual dim for slot 's', value-dim 'i'.  slot in 0..9 (A0..A4,B0..B4)."""
    return SLOT_BASE + 10 * slot + i


def _prompt_pos_for_slot(slot: int) -> int:
    """Prompt position of slot s: A_0..A_4 -> 0..4, B_0..B_4 -> 6..10."""
    return slot if slot < 5 else slot + 1


def _write_block0(block: "Block") -> None:
    # ---- Attention: 10 gather heads + a self-sink ----
    # n_heads = 16, d_head = 16. Heads 0..9 = gather; head 10 = self-sink.
    dh = 16
    # Gather heads: query at output positions {11..16} matches a unique key
    # placed at the head's target prompt position.
    for h in range(10):
        p_h = _prompt_pos_for_slot(h)
        # Query: head h's intra-dim 0 = SCALE at any output position 11..16.
        for p in range(11, 17):
            block.attn.W_q.weight[h * dh + 0, p] = SCALE
        # Key: head h's intra-dim 0 = 1 at its target prompt position only.
        block.attn.W_k.weight[h * dh + 0, p_h] = 1.0
        # Value: read the digit-token one-hot (dims 18..27) into head h's
        # first 10 value dims.
        for i in range(10):
            block.attn.W_v.weight[h * dh + i, 18 + i] = 1.0
        # Output projection: head h's first 10 value dims -> residual slot h.
        for i in range(10):
            block.attn.W_o.weight[_slot_dim(h, i), h * dh + i] = 1.0
    # Self-sink head (10): at every position (including the '=') a strong
    # self-attention to the *current* prompt position with zero value, so
    # softmaxes can't blow up. This head writes nothing (W_o stays zero).
    # We give it queries/keys that match a fixed pos dim only at position 11,
    # so the gather heads never wander there themselves.
    # (Not strictly required given our W_k design, but cheap defensive code.)
    h_sink = 10
    block.attn.W_q.weight[h_sink * dh + 0, 11] = SCALE
    block.attn.W_k.weight[h_sink * dh + 0, 11] = 1.0
    # Leave W_v, W_o zero for the sink head.

    # ---- MLP: compute g_i, p_i, S^0_{i+1}, S^1_{i+1} ----
    # For each (i in 0..4, a in 0..9, b in 0..9) make one AND-unit
    #   h_{i,a,b} = ReLU(A_i_oh[a] + B_i_oh[b] - 1)
    # = 500 hidden units laid out as idx = 100*i + 10*a + b.
    for i in range(5):
        a_slot = i          # slot 0..4 holds A_0..A_4
        b_slot = i + 5      # slot 5..9 holds B_0..B_4
        for a in range(10):
            for b in range(10):
                idx = 100 * i + 10 * a + b
                block.mlp.fc1.weight[idx, _slot_dim(a_slot, a)] = 1.0
                block.mlp.fc1.weight[idx, _slot_dim(b_slot, b)] = 1.0
                block.mlp.fc1.bias[idx] = -1.0

    # fc2 routes the AND-units to g, p, S^0, S^1 dims.
    for i in range(5):
        for a in range(10):
            for b in range(10):
                idx = 100 * i + 10 * a + b
                s_no_carry = (a + b) % 10
                # S^0 (assume incoming carry = 0)
                block.mlp.fc2.weight[S0_BASE + 10 * i + s_no_carry, idx] = 1.0
                # g_i: only pairs with a + b >= 10
                if a + b >= 10:
                    block.mlp.fc2.weight[G_BASE + i, idx] = 1.0
                # p_i: only pairs with a + b == 9
                if a + b == 9:
                    block.mlp.fc2.weight[P_BASE + i, idx] = 1.0
                # S^1 (assume incoming carry = 1); skip i == 4 (S_5^1 unused)
                if i < 4:
                    s_carry = (a + b + 1) % 10
                    block.mlp.fc2.weight[S1_BASE + 10 * i + s_carry, idx] = 1.0


def _write_block1(block: "Block") -> None:
    # ---- Attention: zero everything (identity) ----
    # All four projections start at zero from torch.zero_(), nothing to do.

    # ---- MLP: carry lookahead ----
    # For each carry c_k (k in 0..4), enumerate disjoint paths
    #   "p_k & p_{k+1} & ... & p_{j-1} & g_j" for j in k..4.
    # Hidden unit for path = ReLU(sum_of_features - (j - k + 1) + 1)
    #                      = ReLU(sum_of_features - (j - k)).
    # Each path contributes +1 to c_k via fc2.
    hidden_idx = 0
    for k in range(5):
        for j in range(k, 5):
            # path features: p_k, p_{k+1}, ..., p_{j-1}, g_j
            features = []
            for q in range(k, j):
                features.append(P_BASE + q)
            features.append(G_BASE + j)
            for f in features:
                block.mlp.fc1.weight[hidden_idx, f] = 1.0
            # ReLU(sum - (len(features) - 1)) -> 1 only when ALL features fire.
            block.mlp.fc1.bias[hidden_idx] = -(len(features) - 1)
            block.mlp.fc2.weight[C_BASE + k, hidden_idx] = 1.0
            hidden_idx += 1
    assert hidden_idx == 1 + 2 + 3 + 4 + 5  # = 15


def _write_block2(block: "Block") -> None:
    # ---- Attention: zero (identity) ----
    # Already zero.

    # ---- MLP: position-select + carry multiplex ----
    # Layout of hidden units (no overlap with anything in block 1's hidden
    # layer: each block has its own MLP):
    #   - p=11: 2 units (S_0 = c_0, 1-bit output -> '0' or '1')
    #   - p=12..15: 20 units each (10 carry=0 + 10 carry=1)
    #   - p=16: 10 units (S_5 = S_5^0, no carry into LSD)
    hidden_idx = 0

    # p = 11: predict S_0 = c_0
    pos_dim = 11  # pos_emb sets dim 11 of residual at position 11
    # s = 0: fires iff pos[11] and c_0 = 0 -> ReLU(pos[11] - c_0)
    block.mlp.fc1.weight[hidden_idx, pos_dim] = 1.0
    block.mlp.fc1.weight[hidden_idx, C_BASE + 0] = -1.0
    block.mlp.fc1.bias[hidden_idx] = 0.0
    block.mlp.fc2.weight[OUT_BASE + 0, hidden_idx] = 1.0
    hidden_idx += 1
    # s = 1: fires iff pos[11] and c_0 = 1 -> ReLU(pos[11] + c_0 - 1)
    block.mlp.fc1.weight[hidden_idx, pos_dim] = 1.0
    block.mlp.fc1.weight[hidden_idx, C_BASE + 0] = 1.0
    block.mlp.fc1.bias[hidden_idx] = -1.0
    block.mlp.fc2.weight[OUT_BASE + 1, hidden_idx] = 1.0
    hidden_idx += 1

    # p = 12..15: predict S_k for k = p - 11, multiplexed by c_k.
    for p in range(12, 16):
        k = p - 11  # k in 1..4
        i_pair = k - 1  # digit-pair index for S_k (A_{k-1} + B_{k-1})
        for s in range(10):
            # carry = 0 branch: ReLU(pos[p] + S_k^0[s] - c_k - 1)
            block.mlp.fc1.weight[hidden_idx, p] = 1.0
            block.mlp.fc1.weight[hidden_idx, S0_BASE + 10 * i_pair + s] = 1.0
            block.mlp.fc1.weight[hidden_idx, C_BASE + k] = -1.0
            block.mlp.fc1.bias[hidden_idx] = -1.0
            block.mlp.fc2.weight[OUT_BASE + s, hidden_idx] = 1.0
            hidden_idx += 1
            # carry = 1 branch: ReLU(pos[p] + S_k^1[s] + c_k - 2)
            block.mlp.fc1.weight[hidden_idx, p] = 1.0
            block.mlp.fc1.weight[hidden_idx, S1_BASE + 10 * i_pair + s] = 1.0
            block.mlp.fc1.weight[hidden_idx, C_BASE + k] = 1.0
            block.mlp.fc1.bias[hidden_idx] = -2.0
            block.mlp.fc2.weight[OUT_BASE + s, hidden_idx] = 1.0
            hidden_idx += 1

    # p = 16: predict S_5 = S_5^0 (c_5 = 0 always).
    for s in range(10):
        block.mlp.fc1.weight[hidden_idx, 16] = 1.0
        block.mlp.fc1.weight[hidden_idx, S0_BASE + 10 * 4 + s] = 1.0
        block.mlp.fc1.bias[hidden_idx] = -1.0
        block.mlp.fc2.weight[OUT_BASE + s, hidden_idx] = 1.0
        hidden_idx += 1

    assert hidden_idx == 2 + 4 * 20 + 10  # = 92


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        # Zero everything first.
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        H = model.n_heads
        T = model.max_seq_len
        assert D == 256 and H == 16 and T >= 18, (D, H, T)

        # Position embedding: pos p -> one-hot at dim p (dims 0..17).
        for p in range(T):
            model.pos_emb.weight[p, p] = 1.0

        # Token embedding: digit d -> one-hot at dim 18+d (digits 0..9).
        # '+' (10) and '=' (11) get zero embeddings.
        for d in range(10):
            model.token_emb.weight[d, 18 + d] = 1.0

        _write_block0(model.blocks[0])
        _write_block1(model.blocks[1])
        _write_block2(model.blocks[2])

        # Head: project final-digit slot to digit-token logits.
        for t in range(10):
            model.head.weight[t, OUT_BASE + t] = 1.0


model_shorthand_name = "CarryLookahead_dff512"
model_description = (
    "Same circuit as CarryLookaheadRipple but with d_ff=512 (block 0 uses 500 "
    "of 512 hidden units; blocks 1-2 use 15 and 92). ~halves MLP params."
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
    parser.add_argument("--task", default="addition-five-digits", help="Task name (see src/task.py).")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "success",
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
