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
    """Causal transformer; defaults sized for hand-built 5-digit addition."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 160,
        n_heads: int = 10,
        n_layers: int = 3,
        d_ff: int = 1024,
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
# Hand-built weights for 5-digit addition with full carry propagation.
# ---------------------------------------------------------------------------
#
# Long-addition spec (positions counted MSD-first):
#   carry_5 = 0
#   for k in 4..0: d_k = A[k] + B[k] + carry_{k+1};
#                  S_{k+1} = d_k mod 10;  carry_k = d_k // 10
#   S_0 = carry_0
#
# Block 0 ("gather digits + generate/propagate"):
#   * 10 attention heads gather the 10 input digits into 10 fixed residual
#     slots. Head h queries any output position {11..16} and matches a key
#     placed only at its target prompt position.
#   * MLP computes per digit-position i in 0..4:
#       g_i = 1[A_i + B_i >= 10],   p_i = 1[A_i + B_i == 9]
#     using one ReLU AND-unit per (i, a, b) triple = 500 units.
#
# Block 1 ("carry lookahead"):
#   * Attention zeroed.
#   * MLP computes c_k via disjoint paths
#       c_k = g_k v (p_k & g_{k+1}) v (p_k & p_{k+1} & g_{k+2}) v ...
#     Paths are mutually exclusive (g_j and p_j can't co-fire), so summing
#     ReLU(sum - count + 1) AND-units yields exact OR. 15 hidden units.
#
# Block 2 ("digit emission"):
#   * Attention zeroed.
#   * MLP fires ANDs over (pos, A_{k-1}, B_{k-1}, c_k) directly. No need to
#     store carry candidates; we evaluate the sum inline:
#       p=11: 2 units mapping c_0 to one-hot of '0' or '1'
#       p=12..15: 800 units = 4 positions * 100 (a,b) * 2 (c)
#       p=16:    100 units = 100 (a,b) (c_5 = 0 always)
#     Each unit writes a 1 to the appropriate output-digit dim.
#
# Final linear head projects the output-digit one-hot slot to the 10 digit
# token logits (tokens '+' and '=' get zero logits and can never win).
# ---------------------------------------------------------------------------

# Residual stream layout (d_model = 160):
POS_BASE  = 0     # pos one-hot, dims 0..16 (17 dims; positions 0..16)
TOK_BASE  = 17    # token-digit one-hot, dims 17..26 (digits 0..9)
SLOT_BASE = 27    # 10 gathered slots * 10 dims, dims 27..126
G_BASE    = 127   # g_0..g_4 scalars, dims 127..131
P_BASE    = 132   # p_0..p_4 scalars, dims 132..136
C_BASE    = 137   # c_0..c_4 scalars, dims 137..141
OUT_BASE  = 142   # output digit one-hot, dims 142..151
# dims 152..159 free.

SCALE = 50.0  # query magnitude for sharp attention (softmax temperature)


def _slot_dim(slot: int, i: int) -> int:
    return SLOT_BASE + 10 * slot + i


def _prompt_pos_for_slot(slot: int) -> int:
    """slot 0..4 -> A position 0..4; slot 5..9 -> B position 6..10."""
    return slot if slot < 5 else slot + 1


def _write_block0(block: "Block") -> None:
    dh = 16  # d_head when d_model=160, n_heads=10
    # ---- Attention: 10 heads, one per input digit ----
    for h in range(10):
        p_h = _prompt_pos_for_slot(h)
        # Query at any output position (11..16) picks up SCALE in head h's dim 0
        for p in range(11, 17):
            block.attn.W_q.weight[h * dh + 0, POS_BASE + p] = SCALE
        # Key at the head's target prompt position p_h
        block.attn.W_k.weight[h * dh + 0, POS_BASE + p_h] = 1.0
        # Value reads digit-token one-hot (10 dims) into head h's dims 0..9
        for i in range(10):
            block.attn.W_v.weight[h * dh + i, TOK_BASE + i] = 1.0
        # Output projection: head h's dims 0..9 -> residual slot h
        for i in range(10):
            block.attn.W_o.weight[_slot_dim(h, i), h * dh + i] = 1.0

    # ---- MLP: ReLU AND-units h_{i,a,b} = ReLU(A_i[a] + B_i[b] - 1) ----
    for i in range(5):
        a_slot, b_slot = i, i + 5
        for a in range(10):
            for b in range(10):
                idx = 100 * i + 10 * a + b
                block.mlp.fc1.weight[idx, _slot_dim(a_slot, a)] = 1.0
                block.mlp.fc1.weight[idx, _slot_dim(b_slot, b)] = 1.0
                block.mlp.fc1.bias[idx] = -1.0

    # fc2 routes the AND-units to g_i and p_i.
    for i in range(5):
        for a in range(10):
            for b in range(10):
                idx = 100 * i + 10 * a + b
                if a + b >= 10:
                    block.mlp.fc2.weight[G_BASE + i, idx] = 1.0
                if a + b == 9:
                    block.mlp.fc2.weight[P_BASE + i, idx] = 1.0


def _write_block1(block: "Block") -> None:
    # Attention is identity (all weights zero).
    # MLP: 15 hidden units for carries via disjoint p*...g paths.
    hidden_idx = 0
    for k in range(5):
        for j in range(k, 5):
            # path = p_k & p_{k+1} & ... & p_{j-1} & g_j
            count = 0
            for q in range(k, j):
                block.mlp.fc1.weight[hidden_idx, P_BASE + q] = 1.0
                count += 1
            block.mlp.fc1.weight[hidden_idx, G_BASE + j] = 1.0
            count += 1
            # ReLU(sum - (count - 1)) -> 1 only when ALL features are 1
            block.mlp.fc1.bias[hidden_idx] = -(count - 1)
            block.mlp.fc2.weight[C_BASE + k, hidden_idx] = 1.0
            hidden_idx += 1
    assert hidden_idx == 15


def _write_block2(block: "Block") -> None:
    # Attention is identity.
    # MLP: for each output position p, fire AND-units that emit the digit.
    hidden_idx = 0

    # p = 11: emit c_0 -> "0" or "1"
    # ReLU(pos[11] - c_0) fires (=1) iff pos[11]=1 and c_0=0 -> writes to '0'
    block.mlp.fc1.weight[hidden_idx, POS_BASE + 11] = 1.0
    block.mlp.fc1.weight[hidden_idx, C_BASE + 0] = -1.0
    block.mlp.fc1.bias[hidden_idx] = 0.0
    block.mlp.fc2.weight[OUT_BASE + 0, hidden_idx] = 1.0
    hidden_idx += 1
    # ReLU(pos[11] + c_0 - 1) fires iff pos[11]=1 and c_0=1 -> writes to '1'
    block.mlp.fc1.weight[hidden_idx, POS_BASE + 11] = 1.0
    block.mlp.fc1.weight[hidden_idx, C_BASE + 0] = 1.0
    block.mlp.fc1.bias[hidden_idx] = -1.0
    block.mlp.fc2.weight[OUT_BASE + 1, hidden_idx] = 1.0
    hidden_idx += 1

    # p = 12..15: emit S_k = (A_{k-1} + B_{k-1} + c_k) mod 10
    for p in range(12, 16):
        k = p - 11
        i_pair = k - 1  # digit-pair index
        a_slot, b_slot = i_pair, i_pair + 5
        for a in range(10):
            for b in range(10):
                # c_k = 0 branch
                s = (a + b) % 10
                block.mlp.fc1.weight[hidden_idx, POS_BASE + p] = 1.0
                block.mlp.fc1.weight[hidden_idx, _slot_dim(a_slot, a)] = 1.0
                block.mlp.fc1.weight[hidden_idx, _slot_dim(b_slot, b)] = 1.0
                block.mlp.fc1.weight[hidden_idx, C_BASE + k] = -1.0
                block.mlp.fc1.bias[hidden_idx] = -2.0
                block.mlp.fc2.weight[OUT_BASE + s, hidden_idx] = 1.0
                hidden_idx += 1
                # c_k = 1 branch
                s = (a + b + 1) % 10
                block.mlp.fc1.weight[hidden_idx, POS_BASE + p] = 1.0
                block.mlp.fc1.weight[hidden_idx, _slot_dim(a_slot, a)] = 1.0
                block.mlp.fc1.weight[hidden_idx, _slot_dim(b_slot, b)] = 1.0
                block.mlp.fc1.weight[hidden_idx, C_BASE + k] = 1.0
                block.mlp.fc1.bias[hidden_idx] = -3.0
                block.mlp.fc2.weight[OUT_BASE + s, hidden_idx] = 1.0
                hidden_idx += 1

    # p = 16: emit S_5 = (A_4 + B_4) mod 10 (c_5 = 0 always)
    for a in range(10):
        for b in range(10):
            s = (a + b) % 10
            block.mlp.fc1.weight[hidden_idx, POS_BASE + 16] = 1.0
            block.mlp.fc1.weight[hidden_idx, _slot_dim(4, a)] = 1.0
            block.mlp.fc1.weight[hidden_idx, _slot_dim(9, b)] = 1.0
            block.mlp.fc1.bias[hidden_idx] = -2.0
            block.mlp.fc2.weight[OUT_BASE + s, hidden_idx] = 1.0
            hidden_idx += 1

    assert hidden_idx == 2 + 4 * 200 + 100  # = 902


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        H = model.n_heads
        T = model.max_seq_len
        assert D == 160 and H == 10 and T >= 18, (D, H, T)

        # Position embedding: pos p -> one-hot at dim POS_BASE + p.
        for p in range(min(T, 17)):
            model.pos_emb.weight[p, POS_BASE + p] = 1.0

        # Token embedding: digit d -> one-hot at dim TOK_BASE + d (digits 0..9).
        for d in range(10):
            model.token_emb.weight[d, TOK_BASE + d] = 1.0

        _write_block0(model.blocks[0])
        _write_block1(model.blocks[1])
        _write_block2(model.blocks[2])

        # Head: project the output-digit one-hot slot to digit-token logits.
        for t in range(10):
            model.head.weight[t, OUT_BASE + t] = 1.0


model_shorthand_name = "CompactCarryLookahead"
model_description = (
    "Compact rewrite of CarryLookaheadRipple: d_model=160 (vs 256), n_heads=10, "
    "3 blocks. Block 0 gathers 10 input digits via 10 heads and computes g_i, "
    "p_i (generate/propagate). Block 1 OR-sums disjoint p*...g carry-lookahead "
    "paths (15 hidden units). Block 2 directly emits (A_{k-1}+B_{k-1}+c_k) mod "
    "10 with 902 AND-units indexed by (pos, a, b, c_k) -- no carry-candidate "
    "storage needed in the residual stream."
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
