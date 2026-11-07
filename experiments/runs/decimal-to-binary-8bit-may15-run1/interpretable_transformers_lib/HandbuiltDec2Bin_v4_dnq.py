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
    """Divide-and-conquer circuit: 8 blocks, each extracts ONE bit of N.

    The residual stream's "current N" channel (ch 24) starts at N (after
    block 0's attention pools the digits) and is decremented every block:

        block b (b = 0..7): extracts bit (7-b), i.e. MSB first.
            threshold = 2 ** (7-b)
            bit       = clipped_step(curN, threshold)
            ch_bit_(7-b) += bit
            ch_curN     += -threshold * bit       (so the next block's
                                                  curN holds the remainder)

    After 8 blocks every bit-channel holds its bit value. A small final-block
    MLP gating step (folded into the head) routes the right bit to the logit
    for the current generation position.

    Channels (d_model = 35):
      0..10   token one-hot
      11..22  position one-hot
      23      is_digit_pos flag (1 at positions 0,1,2)
      24      current N (mutated by each block)
      25..32  bit_0..bit_7 of N (mutated by their respective blocks)
      33      logit-for-'0' (set by head's view of (pos, bits))
      34      attention scoring/value lane (transient)

    To keep the structure interpretable, only block 0 has a non-zero
    attention (it pools the digits into curN via log-place weighting, just
    like v3). All later blocks have zero attention.

    The position gating still has to be a multiplicative interaction. We
    handle it by reusing block 7's MLP slot: after extracting the LSB, the
    same MLP also writes the 16 position-gating AND-gate outputs into ch
    33 (logit-for-'0') and ch 34_b (folded as a second scoring lane). To
    keep things clean and within param budget, we instead add ONE extra
    block (block 8) that does only the gating, mirroring v3's "Block 1".
    """
    import math as _math
    torch.manual_seed(0)
    D = model.d_model
    V = model.vocab_size
    S = model.max_seq_len
    assert D >= 35
    assert V == 11
    assert model.n_layers == 9, "divide-and-conquer wants 8 bit-extract + 1 gate"

    for blk in model.blocks:
        blk.ln1 = nn.Identity()
        blk.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    # Per-block MLP sizing.
    for b in range(8):
        model.blocks[b].mlp = MLP(D, 2)   # one clipped-step ReLU pair
    model.blocks[8].mlp = MLP(D, 16)      # 8 bits × {is-1, is-0} position gates

    LANE = 34
    CH_N = 24
    CH_BIT0 = 25
    CH_LOGIT0 = 33
    CH_LOGIT1 = LANE  # reuse channel 34 for logit-for-'1' after attn lane goes idle

    with torch.no_grad():
        for p in model.parameters():
            p.data.zero_()

        # --- Embeddings ---
        for t in range(V):
            model.token_emb.weight.data[t, t] = 1.0
        for p in range(min(S, 12)):
            model.pos_emb.weight.data[p, 11 + p] = 1.0
        for p in (0, 1, 2):
            model.pos_emb.weight.data[p, 23] = 1.0

        # --- Block 0 attention: log-place weighted pool of digits into ch_N. ---
        attn0 = model.blocks[0].attn
        sqrtD = _math.sqrt(D)
        LARGE = 1000.0
        log_place = [_math.log(100.0), _math.log(10.0), _math.log(1.0)]
        for c in range(V):
            attn0.W_q.weight.data[LANE, c] = 1.0
        for c in range(V):
            attn0.W_k.weight.data[LANE, c] = -LARGE * sqrtD
        for p in (0, 1, 2):
            attn0.W_k.weight.data[LANE, 11 + p] = (LARGE + log_place[p]) * sqrtD
        for d in range(10):
            attn0.W_v.weight.data[LANE, d] = float(d)
        attn0.W_o.weight.data[CH_N, LANE] = 111.0

        # --- Blocks 0..7 MLP: each extracts one bit and decrements curN. ---
        for b_idx in range(8):
            bit_index = 7 - b_idx          # extract MSB first
            threshold = 1 << bit_index     # 128, 64, 32, ..., 1
            mlp = model.blocks[b_idx].mlp
            # Hidden neurons: clipped_step(curN, threshold)
            #   ia: ReLU(curN - (threshold - 1))
            #   ib: ReLU(curN - threshold)
            mlp.fc1.weight.data[0, CH_N] = 1.0
            mlp.fc1.bias.data[0] = -float(threshold - 1)
            mlp.fc1.weight.data[1, CH_N] = 1.0
            mlp.fc1.bias.data[1] = -float(threshold)
            # Output: bit = ia - ib  -> ch_bit_(bit_index)
            mlp.fc2.weight.data[CH_BIT0 + bit_index, 0] = 1.0
            mlp.fc2.weight.data[CH_BIT0 + bit_index, 1] = -1.0
            # Update: curN -= threshold * bit
            mlp.fc2.weight.data[CH_N, 0] = -float(threshold)
            mlp.fc2.weight.data[CH_N, 1] = +float(threshold)

        # --- Block 8 MLP: position gating into logit channels. ---
        # (Block 8 attention is zero, already zeroed.)
        mlp_gate = model.blocks[8].mlp
        for b in range(8):
            p = 10 - b
            i1 = 2 * b
            i0 = 2 * b + 1
            mlp_gate.fc1.weight.data[i1, 11 + p] = 1.0
            mlp_gate.fc1.weight.data[i1, CH_BIT0 + b] = 1.0
            mlp_gate.fc1.bias.data[i1] = -1.0
            mlp_gate.fc1.weight.data[i0, 11 + p] = 1.0
            mlp_gate.fc1.weight.data[i0, CH_BIT0 + b] = -1.0
            mlp_gate.fc2.weight.data[CH_LOGIT1, i1] = 1.0
            mlp_gate.fc2.weight.data[CH_LOGIT0, i0] = 1.0

        # --- Head ---
        LOGIT_SCALE = 10.0
        model.head.weight.data[0, CH_LOGIT0] = LOGIT_SCALE
        model.head.weight.data[1, CH_LOGIT1] = LOGIT_SCALE


model_shorthand_name = "HandbuiltDec2Bin_v4_dnq"
model_description = (
    "Divide-and-conquer: 8 stacked blocks, each extracting one bit by "
    "clipped_step(curN, 2^k) and subtracting 2^k * bit from curN. Block 0 "
    "also pools the digits to N via log-place attention (as in v3). A "
    "9th block runs the position-gating ANDs into the head. Each "
    "bit-extracting MLP has just 2 hidden neurons."
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
        d_model=35,
        n_heads=1,
        n_layers=9,
        d_ff=2,  # overridden per-block inside write_weights
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
