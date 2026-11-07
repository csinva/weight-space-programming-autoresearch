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
        d_model: int = 48,
        n_heads: int = 12,
        n_mlp_only_blocks: int = 3,
        d_ff: int = 4,
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

    D = model.d_model       # 48
    H = model.n_heads       # 12
    dh = D // H             # 4
    F = model.d_ff          # 8
    T = model.max_seq_len   # 10
    V = task.vocab_size     # 6

    assert D == 48 and H == 12 and dh == 4 and model.n_layers == 4 and F == 4
    assert V == 6
    # vocab: "01&|^=" => idx 0:'0', 1:'1', 2:'&', 3:'|', 4:'^', 5:'='

    # ---------------- Embeddings ----------------
    # token_emb at dims 40..43 — 4 features (is_one, is_AND, is_OR, is_XOR).
    # '0' and '=' have all four = 0.
    token_emb = torch.zeros(V, D)
    token_emb[1, 40] = 1.0  # '1' is_one
    token_emb[2, 41] = 1.0  # '&' is_AND
    token_emb[3, 42] = 1.0  # '|' is_OR
    token_emb[4, 43] = 1.0  # '^' is_XOR
    model.token_emb.weight.data.copy_(token_emb)

    # pos_emb at dims 44..47 — 4-bit ±1 encoding of position.
    pos_emb = torch.zeros(T, D)
    for i in range(T):
        for k in range(4):
            bit = (i >> (3 - k)) & 1   # k=0 -> high bit (b3), k=3 -> b0
            pos_emb[i, 44 + k] = 1.0 if bit == 1 else -1.0
    model.pos_emb.weight.data.copy_(pos_emb)

    # ---------------- Layer 1: gather + step 1 ----------------
    SCALE = 8.0  # makes softmax effectively pick the matching position
    block1 = model.blocks[0]

    Wq = torch.zeros(D, D)
    Wk = torch.zeros(D, D)
    Wv = torch.zeros(D, D)

    # pos_emb[9] pattern at dims 44..47:
    pos9 = pos_emb[9, 44:48].clone()   # shape (4,)

    for h in range(9):
        target = pos_emb[h, 44:48]     # 4-vector of ±1 (== K[pos h, head h])
        # Q at pos 9 head h, dim k = (SCALE * target[k] / pos9[k]) * residual[pos 9, 44+k]
        # but residual[pos 9, 44+k] == pos9[k], so
        #   Q[pos 9, head h, k] = SCALE * target[k].
        # For pos i != 9, residual is different and Q would be different, but we
        # only use pos 9's attention output.
        for k in range(4):
            # Pick a single column to source from for simplicity: column 44 (high
            # bit of position). pos_emb[9, 44] is non-zero (specifically +1).
            # Then Q[pos 9, head h, k] = W * pos_emb[9, 44].
            Wq[h * dh + k, 44] = SCALE * target[k].item() / pos9[0].item()

        # K[pos i, head h] = pos_emb[i, 44..47]  (identity on those 4 dims)
        for k in range(4):
            Wk[h * dh + k, 44 + k] = 1.0

        # V[pos h, head h, dim k] = k-th token feature at pos h.
        Wv[h * dh + 0, 40] = 1.0  # bit_value (is_one)
        Wv[h * dh + 1, 41] = 1.0  # is_AND
        Wv[h * dh + 2, 42] = 1.0  # is_OR
        Wv[h * dh + 3, 43] = 1.0  # is_XOR

    block1.attn.W_q.weight.data.copy_(Wq)
    block1.attn.W_k.weight.data.copy_(Wk)
    block1.attn.W_v.weight.data.copy_(Wv)
    block1.attn.W_o.weight.data.copy_(torch.eye(D))  # identity output proj

    # ---------------- Layers 2..4: attention is a no-op ----------------
    # (MLPOnlyBlock layers — no attention sub-module to zero.)

    # ---------------- MLP step builder ----------------
    # Single shared accumulator dim (ACC_DIM). Each layer's MLP reads acc_old
    # from this dim and adds a SIGNED delta back into the same dim via the
    # residual connection: acc_new = acc_old + delta(op, acc_old, bit).
    # Truth-table deltas (per op):
    #   AND: nonzero only at (acc=1, bit=0) → delta=-1
    #   OR : nonzero only at (acc=0, bit=1) → delta=+1
    #   XOR: nonzero at (acc=0, bit=1) → +1, and at (acc=1, bit=1) → -1
    # Total: 4 ReLU hidden units — matches d_ff=4.
    UNIT_PATTERNS = [
        # (op_name, target_acc, target_bit, delta_sign)
        ("AND", 1, 0, -1),
        ("OR",  0, 1, +1),
        ("XOR", 0, 1, +1),
        ("XOR", 1, 1, -1),
    ]

    # Channel positions:
    def bit_dim(prompt_pos):   # bit positions 0,2,4,6,8
        return prompt_pos * dh + 0
    def op_dims(prompt_pos):   # op positions 1,3,5,7
        base = prompt_pos * dh
        return base + 1, base + 2, base + 3   # (AND, OR, XOR)

    ACC_DIM = 36   # shared across all layers

    def set_step_mlp(block, op_and_dim, op_or_dim, op_xor_dim, bit_in_dim):
        fc1_w = torch.zeros(F, D)
        fc1_b = torch.zeros(F)
        fc2_w = torch.zeros(D, F)
        fc2_b = torch.zeros(D)
        op_lookup = {"AND": op_and_dim, "OR": op_or_dim, "XOR": op_xor_dim}
        for idx, (op_name, a, b, sgn) in enumerate(UNIT_PATTERNS):
            op_dim = op_lookup[op_name]
            s_a = 1.0 if a == 1 else -1.0
            s_b = 1.0 if b == 1 else -1.0
            fc1_w[idx, op_dim] = 1.0
            fc1_w[idx, ACC_DIM] = s_a
            fc1_w[idx, bit_in_dim] = s_b
            fc1_b[idx] = -2.5 + (1 - a) + (1 - b)
            # Hidden fires with 0.5 on match → fc2 weight 2*sgn gives delta=sgn.
            fc2_w[ACC_DIM, idx] = 2.0 * sgn
        block.mlp.fc1.weight.data.copy_(fc1_w)
        block.mlp.fc1.bias.data.copy_(fc1_b)
        block.mlp.fc2.weight.data.copy_(fc2_w)
        block.mlp.fc2.bias.data.copy_(fc2_b)

    # Step 1: layer 0. Attention head 9 (set up below) gathers b0 into ACC_DIM,
    # so by the time layer 0's MLP runs, ACC_DIM = b0 — treat as acc_old. The
    # signed-delta units then transform ACC_DIM into acc_1 = b0 OP0 b1.
    a0, o0, x0 = op_dims(1)
    set_step_mlp(model.blocks[0], a0, o0, x0, bit_dim(2))

    # Set up attention head 9 (previously unused) to gather b0 into ACC_DIM
    # at pos 9. Same Q/K trick as the other 9 heads.
    Wq_curr = model.blocks[0].attn.W_q.weight.data
    Wk_curr = model.blocks[0].attn.W_k.weight.data
    Wv_curr = model.blocks[0].attn.W_v.weight.data
    target0 = pos_emb[0, 44:48]
    for k in range(4):
        Wq_curr[9 * dh + k, 44] = SCALE * target0[k].item() / pos9[0].item()
        Wk_curr[9 * dh + k, 44 + k] = 1.0
    Wv_curr[9 * dh + 0, 40] = 1.0   # V: copy is_one (= b0) into head-9 dim 0
    # Wo is identity, so head-9 dim 0 lands in residual dim 9*4+0 = 36 = ACC_DIM.

    # Step 2: acc_2 = acc_1 OP1 b2
    a1, o1, x1 = op_dims(3)
    set_step_mlp(model.blocks[1], a1, o1, x1, bit_dim(4))
    # Step 3: acc_3 = acc_2 OP2 b3
    a2, o2, x2 = op_dims(5)
    set_step_mlp(model.blocks[2], a2, o2, x2, bit_dim(6))
    # Step 4: acc_4 = acc_3 OP3 b4
    a3, o3, x3 = op_dims(7)
    set_step_mlp(model.blocks[3], a3, o3, x3, bit_dim(8))

    # ---------------- Output head ----------------
    # At pos 9 the residual has acc_4 in dim ACC_DIM (=36) and pos_emb[9, 47]=+1
    # (low bit of 9). Use dim 47 as always-on baseline for token '0'.
    head_w = torch.zeros(V, D)
    head_w[0, ACC_DIM] = -10.0
    head_w[0, 47] = 0.5
    head_w[1, ACC_DIM] = 10.0
    model.head.weight.data.copy_(head_w)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "ChainedReductionV5_sharedacc"
model_description = (
    "Single shared accumulator dim (36): each MLP layer adds a signed delta "
    "= acc_new - acc_old into ACC_DIM via residual. Truth table allows just "
    "4 ReLU units per MLP (was 6), so d_ff drops from 8 to 4. Head 9 (formerly "
    "unused) gathers b0 into ACC_DIM so layer 0 can use the same recipe."
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
