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


class SimpleTransformer(nn.Module):
    """Causal transformer with no LayerNorm — weights are hand-written."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 96,
        n_heads: int = 12,
        n_layers: int = 4,
        d_ff: int = 32,
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
    """Hand-build a circuit that evaluates a 5-bit / 4-op boolean expression
    left-to-right.

    Strategy (chained reduction):
      Layer 1 attention: at pos 9 ('='), use 9 attention heads — head h
        attends to prompt position h and copies that token's one-hot
        token-type features into head h's d_head slot. After layer 1's
        attention, residual at pos 9 contains all 9 prompt tokens'
        features in disjoint dim slots.
      Layer 1 MLP: compute acc_1 = b0 OP0 b1, write to dim 72.
      Layer 2 MLP: compute acc_2 = acc_1 OP1 b2, write to dim 73.
      Layer 3 MLP: compute acc_3 = acc_2 OP2 b3, write to dim 74.
      Layer 4 MLP: compute acc_4 = acc_3 OP3 b4, write to dim 75.
      Layers 2..4 attention is zeroed out (W_V=0) — pure identity.
      Output head reads dim 75 (acc_4) to pick '0' or '1' at pos 9.

    Residual stream channel layout (d_model = 96):
      dims 0..71   — head 0..8 attention-output slots (8 dims each).
                     head h handles prompt position h:
                       pos 0=b0, 1=op0, 2=b1, 3=op1, 4=b2,
                       5=op2, 6=b3, 7=op3, 8=b4
                     Within head h's slot: dim 0=bit_value, dim 1=is_AND,
                     dim 2=is_OR, dim 3=is_XOR. Other 4 dims unused.
      dims 72..75  — running accumulators acc_1, acc_2, acc_3, acc_4
                     (written by MLP layers 1..4 respectively).
      dims 76..79  — head 9 slot (unused).
      dims 80..89  — position one-hot (pos_emb[i, 80+i] = 1).
      dims 90..95  — token-type one-hot (token_emb[t, 90+t] = 1),
                     vocab = "01&|^=".

    Per-step MLP recipe: each reduction step has 6 hidden units, one per
    (op, prev_acc_val, bit_val) case where the result is 1:
      AND(1,1)=1
      OR (0,1)=1, OR(1,0)=1, OR(1,1)=1
      XOR(0,1)=1, XOR(1,0)=1
    Hidden unit h_{op,a,b} = ReLU(read_op + sign_a*read_acc + sign_b*read_bit
                                  + (1-a) + (1-b) - 2.5)
    where sign_x = +1 if x==1 else -1. With inputs in {0,1}, the unit fires
    iff (op matches, acc matches, bit matches), with value 0.5; otherwise 0.
    fc2 routes each firing hidden unit to acc_out_dim with weight +2 so the
    written delta is exactly 1 when the case matches.
    """
    torch.manual_seed(0)

    D = model.d_model       # 96
    H = model.n_heads       # 12
    dh = D // H             # 8
    F = model.d_ff          # 32
    T = model.max_seq_len   # >= 16
    V = task.vocab_size     # 6

    assert D == 96 and H == 12 and dh == 8 and model.n_layers == 4
    assert V == 6
    # vocab: "01&|^=" => idx 0:'0', 1:'1', 2:'&', 3:'|', 4:'^', 5:'='

    # ---------------- Embeddings ----------------
    token_emb = torch.zeros(V, D)
    for t in range(V):
        token_emb[t, 90 + t] = 1.0
    model.token_emb.weight.data.copy_(token_emb)

    pos_emb = torch.zeros(T, D)
    for i in range(min(T, 10)):
        pos_emb[i, 80 + i] = 1.0
    model.pos_emb.weight.data.copy_(pos_emb)

    # ---------------- Helpers ----------------
    SCALE = 10.0  # makes attention softmax effectively a hard pointer

    # Slot dim positions in residual at pos 9 (after layer 1 attention):
    #   pos h's head-h slot starts at h*dh = h*8.
    #   bit_value at dim h*8 + 0
    #   is_AND at h*8 + 1, is_OR at h*8 + 2, is_XOR at h*8 + 3
    def bit_dim(prompt_pos):   # for bit positions 0,2,4,6,8
        return prompt_pos * dh + 0
    def op_dims(prompt_pos):   # for op positions 1,3,5,7 -> (and, or, xor) dims
        base = prompt_pos * dh
        return base + 1, base + 2, base + 3

    ACC_DIMS = [72, 73, 74, 75]  # acc_1..acc_4 written by MLP layers 1..4

    # ---------------- Layer 1: gather + step 1 ----------------
    block1 = model.blocks[0]

    Wq = torch.zeros(D, D)
    Wk = torch.zeros(D, D)
    Wv = torch.zeros(D, D)

    for h in range(9):  # heads 0..8 do the gather; head 9..11 inert
        # Q at pos 9 wants to match K at pos h via "match dim" = dh-row 0.
        #   Q[pos i, head h, 0] = sum_j Wq[h*dh+0, j] * residual[pos i, j]
        #   want this to be LARGE only at pos i==9 (Q comes from pos 9).
        Wq[h * dh + 0, 80 + 9] = SCALE     # reads pos_emb dim 89 (=1 at pos 9)

        #   K[pos i, head h, 0] = LARGE only when i==h
        Wk[h * dh + 0, 80 + h] = SCALE     # reads pos_emb dim 80+h (=1 at pos h)

        # V at pos h, head h: extract 4 token features (bit_value / op-one-hot)
        #   V dim 0: is_token_'1' -> bit_value
        Wv[h * dh + 0, 91] = 1.0  # token_emb dim 91 == '1'
        Wv[h * dh + 1, 92] = 1.0  # token_emb dim 92 == '&'
        Wv[h * dh + 2, 93] = 1.0  # token_emb dim 93 == '|'
        Wv[h * dh + 3, 94] = 1.0  # token_emb dim 94 == '^'

    block1.attn.W_q.weight.data.copy_(Wq)
    block1.attn.W_k.weight.data.copy_(Wk)
    block1.attn.W_v.weight.data.copy_(Wv)
    block1.attn.W_o.weight.data.copy_(torch.eye(D))  # identity output proj

    # ---------------- Layers 2..4: attention is a no-op ----------------
    for L in range(1, model.n_layers):
        block = model.blocks[L]
        block.attn.W_q.weight.data.zero_()
        block.attn.W_k.weight.data.zero_()
        block.attn.W_v.weight.data.zero_()
        block.attn.W_o.weight.data.zero_()

    # ---------------- MLP step builder ----------------
    # Step k uses 6 hidden units, one per (op, a, b) where result=1.
    # Inputs (all 0/1 valued in residual at pos 9):
    #   read_acc  dim acc_in
    #   read_bit  dim bit_in
    #   read_op_AND, read_op_OR, read_op_XOR : 3 dims
    # Hidden unit for (op, a, b):
    #   h = ReLU(x_op + s_a * x_acc + s_b * x_bit + bias)
    #   where s_a = +1 if a==1 else -1, similarly s_b, and
    #   bias = -2.5 + (1-a) + (1-b)
    # fires with value 0.5 iff all three indicators match; fc2 routes +2 -> acc_out
    UNIT_PATTERNS = [
        ("AND", 1, 1),
        ("OR",  0, 1), ("OR", 1, 0), ("OR", 1, 1),
        ("XOR", 0, 1), ("XOR", 1, 0),
    ]
    OP_NAME_TO_OFFSET = {"AND": 0, "OR": 1, "XOR": 2}  # offsets into op_dims tuple

    def set_step_mlp(block, acc_in_dim, op_and_dim, op_or_dim, op_xor_dim,
                     bit_in_dim, acc_out_dim):
        fc1_w = torch.zeros(F, D)
        fc1_b = torch.zeros(F)
        fc2_w = torch.zeros(D, F)
        fc2_b = torch.zeros(D)

        op_dim_lookup = {"AND": op_and_dim, "OR": op_or_dim, "XOR": op_xor_dim}

        for idx, (op_name, a, b) in enumerate(UNIT_PATTERNS):
            op_dim = op_dim_lookup[op_name]
            s_a = 1.0 if a == 1 else -1.0
            s_b = 1.0 if b == 1 else -1.0
            fc1_w[idx, op_dim] = 1.0
            fc1_w[idx, acc_in_dim] = s_a
            fc1_w[idx, bit_in_dim] = s_b
            fc1_b[idx] = -2.5 + (1 - a) + (1 - b)
            # fc2: this hidden unit (fires 0.5 on match) contributes 1.0 to acc_out
            fc2_w[acc_out_dim, idx] = 2.0

        # Unused hidden units idx 6..F-1: keep zero (won't fire because their
        # bias is 0 and weights are 0 -> ReLU(0)=0).

        block.mlp.fc1.weight.data.copy_(fc1_w)
        block.mlp.fc1.bias.data.copy_(fc1_b)
        block.mlp.fc2.weight.data.copy_(fc2_w)
        block.mlp.fc2.bias.data.copy_(fc2_b)

    # ---------------- Configure the 4 MLP reduction layers ----------------
    # Step 1: acc_1 = b0 OP0 b1
    op0_and, op0_or, op0_xor = op_dims(1)
    set_step_mlp(model.blocks[0],
                 acc_in_dim=bit_dim(0),       # b0 at dim 0
                 op_and_dim=op0_and, op_or_dim=op0_or, op_xor_dim=op0_xor,
                 bit_in_dim=bit_dim(2),       # b1 at dim 16
                 acc_out_dim=ACC_DIMS[0])     # write acc_1 to dim 72

    # Step 2: acc_2 = acc_1 OP1 b2
    op1_and, op1_or, op1_xor = op_dims(3)
    set_step_mlp(model.blocks[1],
                 acc_in_dim=ACC_DIMS[0],
                 op_and_dim=op1_and, op_or_dim=op1_or, op_xor_dim=op1_xor,
                 bit_in_dim=bit_dim(4),
                 acc_out_dim=ACC_DIMS[1])

    # Step 3: acc_3 = acc_2 OP2 b3
    op2_and, op2_or, op2_xor = op_dims(5)
    set_step_mlp(model.blocks[2],
                 acc_in_dim=ACC_DIMS[1],
                 op_and_dim=op2_and, op_or_dim=op2_or, op_xor_dim=op2_xor,
                 bit_in_dim=bit_dim(6),
                 acc_out_dim=ACC_DIMS[2])

    # Step 4: acc_4 = acc_3 OP3 b4
    op3_and, op3_or, op3_xor = op_dims(7)
    set_step_mlp(model.blocks[3],
                 acc_in_dim=ACC_DIMS[2],
                 op_and_dim=op3_and, op_or_dim=op3_or, op_xor_dim=op3_xor,
                 bit_in_dim=bit_dim(8),
                 acc_out_dim=ACC_DIMS[3])

    # ---------------- Output head ----------------
    # Want at pos 9: logit('1') wins iff acc_4 == 1, else logit('0') wins.
    # Use dim 75 (acc_4) plus dim 89 (pos_emb[9]=1, always on at pos 9) for
    # the constant baseline that breaks the acc_4==0 tie.
    head_w = torch.zeros(V, D)
    head_w[0, 75] = -10.0   # token '0': logit = -10 * acc_4 + 0.5
    head_w[0, 89] = 0.5
    head_w[1, 75] = 10.0    # token '1': logit = +10 * acc_4
    # Other rows stay zero -> logit 0 -> never wins over '0' (0.5) or '1' (10).
    model.head.weight.data.copy_(head_w)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ChainedReductionV1"
model_description = (
    "No-LN transformer (d=96, 12 heads, 4 layers). Layer 1 attention uses 9 heads "
    "(one per prompt position) to gather token features into dim slots at pos 9; "
    "layers 1-4 MLPs each apply one left-to-right reduction step (acc_k = "
    "acc_{k-1} OP_{k-1} b_k) via 6 ReLU hidden units encoding the truth table; "
    "head reads acc_4 to emit '0' or '1'."
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
