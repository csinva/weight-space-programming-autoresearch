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
    """Hand-built circuit for decimal-to-binary-8bit.

    Circuit overview (d_model=64, 3 blocks, 1 head):
      ch 0..10  : token one-hot (from token_emb)
      ch 11..22 : position one-hot (from pos_emb, 12 positions)
      ch 23     : is_digit_pos flag (1 at positions 0,1,2)
      ch 24     : place_value contribution (digit * 100/10/1) -- MLP0
      ch 25     : N, the decoded decimal value 0..255 -- Attn1
      ch 26..33 : bit_0..bit_7 of N -- MLP1
      ch 34     : logit-for-'0' accumulator -- MLP2
      ch 35     : logit-for-'1' accumulator -- MLP2
      ch 36     : attention "score/value lane" (transient)

    Block 0: attn=0; MLP0 is a (pos, digit) lookup table producing place value.
    Block 1: attn attends to {pos 0,1,2} via is_digit_pos key flag, sums place
             values (averaged then ×3) into ch 25; MLP1 computes all 8 bits of N
             with clipped-step ReLU pairs over channel 25.
    Block 2: attn=0; MLP2 gates bits by position so that at prediction position
             p in {3..10} only bit (10-p) is routed to the '1' logit (and its
             complement to the '0' logit). Head maps ch 34 -> '0', ch 35 -> '1'.
    """
    torch.manual_seed(0)
    D = model.d_model
    V = model.vocab_size
    S = model.max_seq_len
    assert D >= 37, "need d_model >= 37 channels"
    assert V == 11, "this circuit assumes the decimal-to-binary-8bit vocab"

    # --- Strip out LayerNorm so the residual stream is exactly what we design.
    for blk in model.blocks:
        blk.ln1 = nn.Identity()
        blk.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    # --- Per-block MLP sizing (each block has different needs).
    model.blocks[0].mlp = MLP(D, 30)   # 3 digit positions × 10 digits
    model.blocks[1].mlp = MLP(D, 510)  # 255 clipped-step thresholds × 2 ReLUs
    model.blocks[2].mlp = MLP(D, 16)   # 8 output bits × {is-1, is-0} neurons

    # --- Zero everything; we'll fill in only the pieces we use.
    with torch.no_grad():
        for p in model.parameters():
            p.data.zero_()

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        # token_emb[t, c] = 1 iff c == t (channels 0..10).
        for t in range(V):
            model.token_emb.weight.data[t, t] = 1.0
        # pos_emb[p, 11+p] = 1 (channels 11..22), and is_digit_pos at ch 23.
        for p in range(min(S, 12)):
            model.pos_emb.weight.data[p, 11 + p] = 1.0
        for p in (0, 1, 2):
            model.pos_emb.weight.data[p, 23] = 1.0

        # ------------------------------------------------------------------
        # Block 0: attention is zero; MLP is a (pos, digit) lookup.
        # ------------------------------------------------------------------
        mlp0 = model.blocks[0].mlp
        place = {0: 100.0, 1: 10.0, 2: 1.0}
        for p in (0, 1, 2):
            for d in range(10):
                k = p * 10 + d
                # Hidden fires iff token == d AND position == p:
                #   ReLU([pos==p] + [tok==d] - 1)
                mlp0.fc1.weight.data[k, 11 + p] = 1.0
                mlp0.fc1.weight.data[k, d] = 1.0
                mlp0.fc1.bias.data[k] = -1.0
                # Contributes place(p) * d to channel 24.
                mlp0.fc2.weight.data[24, k] = place[p] * d

        # ------------------------------------------------------------------
        # Block 1 attention: gather place_value from positions 0,1,2 into ch 25.
        # Single head, d_head = D. Scoring lane is channel 36.
        # ------------------------------------------------------------------
        attn1 = model.blocks[1].attn
        # Q at every position has q[36] = sum of token one-hot = 1.
        for c in range(V):
            attn1.W_q.weight.data[36, c] = 1.0
        # K[j, 36] = 1000 * is_digit_pos[j]  (very sharp softmax onto 0,1,2).
        attn1.W_k.weight.data[36, 23] = 1000.0
        # V[j, 36] = place_value[j].
        attn1.W_v.weight.data[36, 24] = 1.0
        # Average over 3 active positions => attended value = N / 3, so ×3.
        attn1.W_o.weight.data[25, 36] = 3.0

        # ------------------------------------------------------------------
        # Block 1 MLP: build bit_b(N) for b=0..7 from clipped step indicators.
        # clipped_step(N, t) = ReLU(N - (t-1)) - ReLU(N - t)  in {0,1} for int N
        # and exactly = 1 iff N >= t.
        # bit_b(N) = sum_{k>=1, k*2^b <= 255} (-1)^(k+1) * clipped_step(N, k*2^b)
        # ------------------------------------------------------------------
        mlp1 = model.blocks[1].mlp
        # Each threshold t in 1..255 uses 2 hidden neurons (indices 2(t-1), 2(t-1)+1).
        for t in range(1, 256):
            idx_a = 2 * (t - 1)
            idx_b = idx_a + 1
            mlp1.fc1.weight.data[idx_a, 25] = 1.0
            mlp1.fc1.bias.data[idx_a] = -float(t - 1)
            mlp1.fc1.weight.data[idx_b, 25] = 1.0
            mlp1.fc1.bias.data[idx_b] = -float(t)
        # Accumulate bit-b contributions into channels 26..33.
        for b in range(8):
            ch_bit = 26 + b
            k = 1
            while k * (1 << b) <= 255:
                t = k * (1 << b)
                sign = 1.0 if (k % 2 == 1) else -1.0
                idx_a = 2 * (t - 1)
                idx_b = idx_a + 1
                # clipped_step = neuron_a - neuron_b
                mlp1.fc2.weight.data[ch_bit, idx_a] += sign
                mlp1.fc2.weight.data[ch_bit, idx_b] += -sign
                k += 1

        # ------------------------------------------------------------------
        # Block 2 attention: zero (already zeroed).
        # Block 2 MLP: at position p in {3..10}, route bit b = 10 - p.
        #   pos_p_bit1 = ReLU(pos_onehot[p] + bit_b - 1)      = bit_b  (when at p)
        #   pos_p_bit0 = ReLU(pos_onehot[p] - bit_b)          = 1-bit_b (when at p)
        # ------------------------------------------------------------------
        mlp2 = model.blocks[2].mlp
        for b in range(8):
            p = 10 - b  # prediction position for bit b
            i1 = 2 * b      # "bit==1" detector
            i0 = 2 * b + 1  # "bit==0" detector
            mlp2.fc1.weight.data[i1, 11 + p] = 1.0
            mlp2.fc1.weight.data[i1, 26 + b] = 1.0
            mlp2.fc1.bias.data[i1] = -1.0

            mlp2.fc1.weight.data[i0, 11 + p] = 1.0
            mlp2.fc1.weight.data[i0, 26 + b] = -1.0
            mlp2.fc1.bias.data[i0] = 0.0

            mlp2.fc2.weight.data[35, i1] = 1.0  # ch 35 = logit-for-'1' contrib
            mlp2.fc2.weight.data[34, i0] = 1.0  # ch 34 = logit-for-'0' contrib

        # ------------------------------------------------------------------
        # Head: ch 34 -> token '0' (id 0), ch 35 -> token '1' (id 1).
        # ------------------------------------------------------------------
        LOGIT_SCALE = 10.0
        model.head.weight.data[0, 34] = LOGIT_SCALE
        model.head.weight.data[1, 35] = LOGIT_SCALE


model_shorthand_name = "HandbuiltDec2Bin_v1"
model_description = (
    "Hand-built 3-block circuit: MLP0 looks up place value per digit pos; "
    "Attn1 sums to N; MLP1 computes all 8 bits via 255 clipped-step ReLU pairs; "
    "MLP2 gates bit-b to the prediction at position p=10-b; head reads bit logit. "
    "LayerNorms replaced with Identity."
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
        d_model=64,
        n_heads=1,
        n_layers=3,
        d_ff=64,  # overridden per-block inside write_weights
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
