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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_attn: bool = True):
        super().__init__()
        # Identity normalizations: the hand-built circuit relies on exact
        # one-hot residual streams, so we bypass LayerNorm entirely.
        self.ln1 = nn.Identity()
        # `use_attn=False` strips out the attention module entirely so the
        # block contains only the MLP. Useful for hand-built circuits whose
        # later layers only need pointwise computation.
        self.attn = CausalSelfAttention(d_model, n_heads) if use_attn else None
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer sized for a hand-built linear-interpolation circuit."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 16,
        d_model: int = 152,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1000,
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
        # Hand-built circuit uses attention only in the first block (to gather
        # input features); subsequent blocks are MLP-only.
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, use_attn=(i == 0))
            for i in range(n_layers)
        ])
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
    """Hand-built interpretable circuit for linear-interpolation-two-points.

    Task: prompt "X1,Y1Y1;X2,Y2Y2;XQ=" → answer "YqYq" where Yq = m*xq + b
    with integer m ∈ {1..5} and b ∈ {0..9}.

    High-level circuit (faithful to the geometric definition of linear interpolation):
      * Find the unique line y = m*x + b that passes through both given points.
      * Evaluate it at xq.

    Residual stream channels (d_model = 152):
      [  0, 13)  token one-hot                                (V=vocab_size=13)
      [ 13, 29)  position one-hot                             (P=max_seq_len=16)
      [ 29, 39)  gathered x1     (digit 0-9 one-hot)
      [ 39, 49)  gathered y1_tens
      [ 49, 59)  gathered y1_ones
      [ 59, 69)  gathered x2
      [ 69, 79)  gathered y2_tens
      [ 79, 89)  gathered y2_ones
      [ 89, 99)  gathered xq
      [ 99,149)  (m, b) "passes-through-both-points" score
                 50 channels indexed by (m_idx ∈ 0..4) * 10 + (b ∈ 0..9).
                 Layer-0 MLP adds 0.5 if line (m,b) passes through pair 1,
                 and 0.5 if it passes through pair 2. The unique true line
                 ends up with score 1.0; lines through exactly one point
                 sit at 0.5; everything else is 0.
      [149,152)  unused

    Layer 0 attention: 7 heads, each gathers one digit (x1, y1_tens, y1_ones,
      x2, y2_tens, y2_ones, xq) from its source position into its slot at
      output positions 11 and 12.

    Layer 0 MLP — 1000 atomic detectors, 500 per pair:
      For each (m_i ∈ 1..5, b_i ∈ 0..9, x_i ∈ 0..9) and each pair ∈ {1, 2},
      one neuron fires only when the pair's (x, y_tens, y_ones) one-hots match
      (x_i, (m_i*x_i+b_i)//10, (m_i*x_i+b_i)%10). The neuron writes 0.5 into
      the (m_i, b_i) score channel. So the score channel for (m, b) accumulates:
        +0.5 from pair 1 if the line passes through pair 1 (5 such (m,b) per pair),
        +0.5 from pair 2 if the line passes through pair 2 (another 5 such (m,b)).
      The unique line through BOTH points ends up at exactly 1.0; lines through
      one point only sit at 0.5.

    Layer 1 attention: zeroed.

    Layer 1 MLP — 1000 (m, b, xq, pos)-indexed neurons:
      Each neuron fires only when the (m, b) score channel is 1.0 AND xq matches
      AND current position matches (∈ {11, 12}). Threshold 2.75 separates score=1.0
      (sum=3.0) from score=0.5 (sum=2.5). The firing neuron writes the correct
      output digit (yq//10 at pos 11, yq%10 at pos 12) into the token one-hot
      subspace with a large weight so head argmax picks it.

    Head: identity on the token one-hot channels.
    """
    d_model = model.d_model       # 152
    n_heads = model.n_heads       # 8
    d_head = d_model // n_heads   # 19
    V = model.vocab_size          # 13
    P = model.max_seq_len         # 16
    d_ff = model.d_ff             # 1000

    # Channel offsets
    TOK = 0
    POS = V                      # 13
    CH_X1  = V + P               # 29
    CH_Y1T = CH_X1 + 10          # 39
    CH_Y1O = CH_X1 + 20          # 49
    CH_X2  = CH_X1 + 30          # 59
    CH_Y2T = CH_X1 + 40          # 69
    CH_Y2O = CH_X1 + 50          # 79
    CH_XQ  = CH_X1 + 60          # 89
    CH_MB  = CH_X1 + 70          # 99 (50 channels: (m_idx, b))
    assert CH_MB + 50 <= d_model, "d_model too small for hand-built layout"

    def mb_chan(m_i: int, b_i: int) -> int:
        return CH_MB + (m_i - 1) * 10 + b_i

    with torch.no_grad():
        # Zero every parameter as a clean canvas.
        for p in model.parameters():
            p.zero_()

        # --- Token embedding: token c → one-hot at channel TOK+c ---
        for c in range(V):
            model.token_emb.weight[c, TOK + c] = 1.0

        # --- Position embedding: position p → one-hot at channel POS+p ---
        for p in range(P):
            model.pos_emb.weight[p, POS + p] = 1.0

        # --- Layer 0 attention: gather one digit per head into pos 11/12 ---
        # head_specs: (head_idx, source_position, dest_slot_start)
        head_specs = [
            (0, 0,  CH_X1),
            (1, 2,  CH_Y1T),
            (2, 3,  CH_Y1O),
            (3, 5,  CH_X2),
            (4, 7,  CH_Y2T),
            (5, 8,  CH_Y2O),
            (6, 10, CH_XQ),
        ]
        attn0 = model.blocks[0].attn
        # Pre-softmax dot-product score at the matched (out, src) pair will be
        # SCALE / sqrt(d_head); with SCALE=50 and d_head=19 → ~11, plenty sharp.
        SCALE = 50.0
        for h, src, slot in head_specs:
            h_start = h * d_head
            # Query at output positions 11 and 12 has 1.0 in head component 0.
            attn0.W_q.weight[h_start, POS + 11] = 1.0
            attn0.W_q.weight[h_start, POS + 12] = 1.0
            # Key at source position src has SCALE in head component 0.
            attn0.W_k.weight[h_start, POS + src] = SCALE
            # Value at any source: copy the source token's digit one-hot
            # (vocab idx 0-9) into head components [1, 11).
            for c in range(10):
                attn0.W_v.weight[h_start + 1 + c, TOK + c] = 1.0
            # Output projection: write the digit one-hot into the slot.
            for c in range(10):
                attn0.W_o.weight[slot + c, h_start + 1 + c] = 1.0

        # --- Layer 0 MLP: 1000 "(m, b, x, pair)" atomic detectors ---
        mlp0 = model.blocks[0].mlp
        neuron_idx = 0
        # Per-pair channel triples (x slot, y_tens slot, y_ones slot).
        pair_slots = [
            (CH_X1, CH_Y1T, CH_Y1O),
            (CH_X2, CH_Y2T, CH_Y2O),
        ]
        for pair_idx, (CX, CYT, CYO) in enumerate(pair_slots):
            for m_i in range(1, 6):
                for b_i in range(10):
                    for x_i in range(10):
                        y = m_i * x_i + b_i
                        mlp0.fc1.weight[neuron_idx, CX  + x_i]     = 1.0
                        mlp0.fc1.weight[neuron_idx, CYT + y // 10] = 1.0
                        mlp0.fc1.weight[neuron_idx, CYO + y % 10]  = 1.0
                        # Threshold: fires (ReLU=0.5) only when all 3 one-hots match.
                        mlp0.fc1.bias[neuron_idx] = -2.5
                        # Add 0.5 (= ReLU output * 1.0) into the (m,b) score channel.
                        mlp0.fc2.weight[mb_chan(m_i, b_i), neuron_idx] = 1.0
                        neuron_idx += 1
        assert neuron_idx <= d_ff, f"layer-0 MLP needs d_ff ≥ {neuron_idx}"

        # --- Layer 1 attention: identically zero (already zeroed above) ---

        # --- Layer 1 MLP: 1000 (m, b, xq, pos)-indexed lookups ---
        # For correct (m,b): score channel = 1.0; sum = 1.0 + 1.0 (xq) + 1.0 (pos) = 3.0
        # For other (m,b) where score channel = 0.5: sum = 0.5 + 1.0 + 1.0 = 2.5
        # Threshold 2.75 picks ONLY the true (m,b) line.
        mlp1 = model.blocks[1].mlp
        neuron_idx = 0
        for m_i in range(1, 6):
            for b_i in range(10):
                for xq_i in range(10):
                    for pos_i in (11, 12):
                        yq = m_i * xq_i + b_i
                        digit = (yq // 10) if pos_i == 11 else (yq % 10)
                        mlp1.fc1.weight[neuron_idx, mb_chan(m_i, b_i)] = 1.0
                        mlp1.fc1.weight[neuron_idx, CH_XQ + xq_i]     = 1.0
                        mlp1.fc1.weight[neuron_idx, POS + pos_i]      = 1.0
                        mlp1.fc1.bias[neuron_idx] = -2.75
                        # Big weight so the digit channel dominates the
                        # input token's own one-hot (which has magnitude 1).
                        mlp1.fc2.weight[TOK + digit, neuron_idx] = 100.0
                        neuron_idx += 1
        assert neuron_idx <= d_ff, f"layer-1 MLP needs d_ff ≥ {neuron_idx}"

        # --- Head: identity on the token one-hot channels ---
        for c in range(V):
            model.head.weight[c, TOK + c] = 1.0


model_shorthand_name = "LinInterpTwoPointAND_v4"
model_description = (
    "Same circuit as v3 but the second block has no attention module (Block "
    "constructed with use_attn=False), removing ~92K dead attention parameters "
    "from the unused second-layer attention while keeping 100% accuracy."
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
