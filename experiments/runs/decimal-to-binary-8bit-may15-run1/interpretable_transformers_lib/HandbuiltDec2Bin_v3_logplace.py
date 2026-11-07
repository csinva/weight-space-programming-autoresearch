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
    """Hand-built 2-block circuit for decimal-to-binary-8bit.

    Key simplification vs v1/v2: the place-value lookup is folded *into the
    attention weights themselves*. We set
        k_j  ∝ log(place(j))  on digit positions {0,1,2}
              ∝ -infinity      elsewhere
    so softmax gives attention weights 100/111, 10/111, 1/111 on positions
    (0, 1, 2). The value is the raw digit at j, so the attended scalar at
    every query position p >= 2 equals N / 111. We multiply by 111 in W_o
    to land N in channel 25.  No "place value" residual channel is needed,
    and we drop a whole block (Block 0).

    Channels (d_model = 36):
      0..10   token one-hot   (from token_emb)
      11..22  position one-hot (from pos_emb, 12 positions)
      23      is_digit_pos flag (1 at positions 0,1,2)
      24      N, the decoded decimal value
      25..32  bit_0..bit_7 of N
      33      logit-for-'0' contribution
      34      logit-for-'1' contribution
      35      attention scoring / value lane (transient)

    Block 0:
      Attn0 — log-place weighted attention pools digit positions into ch 24.
      MLP0  — all 8 bits of N via 255 clipped-step ReLU pairs (510 hidden).
    Block 1:
      Attn1 — zero.
      MLP1  — position-gated bit selection into ch 33/34 (16 hidden).
    Head    — ch 33 -> token '0' (id 0); ch 34 -> token '1' (id 1).
    """
    import math as _math
    torch.manual_seed(0)
    D = model.d_model
    V = model.vocab_size
    S = model.max_seq_len
    assert D >= 36
    assert V == 11

    # Strip LayerNorms so the residual stream is what we design.
    for blk in model.blocks:
        blk.ln1 = nn.Identity()
        blk.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    # Per-block MLP sizing.
    model.blocks[0].mlp = MLP(D, 510)
    model.blocks[1].mlp = MLP(D, 16)

    LANE = 35      # attention scoring/value lane
    CH_N = 24
    CH_BIT0 = 25   # ch_bit_b lives at CH_BIT0 + b
    CH_LOGIT0 = 33
    CH_LOGIT1 = 34

    with torch.no_grad():
        for p in model.parameters():
            p.data.zero_()

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        for t in range(V):
            model.token_emb.weight.data[t, t] = 1.0
        for p in range(min(S, 12)):
            model.pos_emb.weight.data[p, 11 + p] = 1.0
        for p in (0, 1, 2):
            model.pos_emb.weight.data[p, 23] = 1.0

        # ------------------------------------------------------------------
        # Block 0 attention: pool place_value * digit into ch_N directly.
        # Score is (q · k) / sqrt(d_head). With one head, d_head = D.
        # Pre-scale our log-place biases by sqrt(d_head) so the *effective*
        # logits after division are exactly log(place(j)).
        # ------------------------------------------------------------------
        attn0 = model.blocks[0].attn
        sqrtD = _math.sqrt(D)
        LARGE = 1000.0
        log_place = [_math.log(100.0), _math.log(10.0), _math.log(1.0)]  # for p=0,1,2

        # q at every position has q[LANE] = sum of token one-hot = 1.
        for c in range(V):
            attn0.W_q.weight.data[LANE, c] = 1.0

        # k_j[LANE] should equal sqrtD * (log_place(j) on digit pos, else -LARGE).
        # We build it as: -LARGE * "1 channel" + (LARGE + log_place) * digit_pos_p
        # where "1 channel" is implemented as the sum of token one-hot (always 1).
        for c in range(V):
            attn0.W_k.weight.data[LANE, c] = -LARGE * sqrtD
        # On the digit-pos one-hots, add (LARGE + log_place[p]) so net is log_place.
        for p in (0, 1, 2):
            attn0.W_k.weight.data[LANE, 11 + p] = (LARGE + log_place[p]) * sqrtD

        # v_j[LANE] = digit value of token at j (= sum c * onehot[j, c]).
        for d in range(10):
            attn0.W_v.weight.data[LANE, d] = float(d)

        # Attended scalar = (100*d0 + 10*d1 + d2)/111 = N/111; scale up to N.
        attn0.W_o.weight.data[CH_N, LANE] = 111.0

        # ------------------------------------------------------------------
        # Block 0 MLP: 8 bits of N via clipped-step ReLU pairs on ch_N.
        # clipped_step(N, t) = ReLU(N - (t-1)) - ReLU(N - t)  ∈ {0,1} for int N.
        # bit_b(N) = Σ_{k≥1, k·2^b ≤ 255} (-1)^(k+1) · clipped_step(N, k·2^b).
        # ------------------------------------------------------------------
        mlp0 = model.blocks[0].mlp
        for t in range(1, 256):
            ia = 2 * (t - 1)
            ib = ia + 1
            mlp0.fc1.weight.data[ia, CH_N] = 1.0
            mlp0.fc1.bias.data[ia] = -float(t - 1)
            mlp0.fc1.weight.data[ib, CH_N] = 1.0
            mlp0.fc1.bias.data[ib] = -float(t)
        for b in range(8):
            ch_bit = CH_BIT0 + b
            k = 1
            while k * (1 << b) <= 255:
                t = k * (1 << b)
                sign = 1.0 if (k % 2 == 1) else -1.0
                ia = 2 * (t - 1)
                ib = ia + 1
                mlp0.fc2.weight.data[ch_bit, ia] += sign
                mlp0.fc2.weight.data[ch_bit, ib] += -sign
                k += 1

        # ------------------------------------------------------------------
        # Block 1 attention: zero (already zeroed).
        # Block 1 MLP: at query position p ∈ {3..10} pick bit b = 10 - p.
        #   pos_p_bit1 = ReLU(pos_onehot[p] + bit_b - 1)  (= bit_b when at p)
        #   pos_p_bit0 = ReLU(pos_onehot[p] - bit_b)      (= 1-bit_b when at p)
        # ------------------------------------------------------------------
        mlp1 = model.blocks[1].mlp
        for b in range(8):
            p = 10 - b
            i1 = 2 * b
            i0 = 2 * b + 1
            mlp1.fc1.weight.data[i1, 11 + p] = 1.0
            mlp1.fc1.weight.data[i1, CH_BIT0 + b] = 1.0
            mlp1.fc1.bias.data[i1] = -1.0
            mlp1.fc1.weight.data[i0, 11 + p] = 1.0
            mlp1.fc1.weight.data[i0, CH_BIT0 + b] = -1.0
            mlp1.fc2.weight.data[CH_LOGIT1, i1] = 1.0
            mlp1.fc2.weight.data[CH_LOGIT0, i0] = 1.0

        # ------------------------------------------------------------------
        # Head: ch_LOGIT0 -> token '0' (id 0); ch_LOGIT1 -> token '1' (id 1).
        # ------------------------------------------------------------------
        LOGIT_SCALE = 10.0
        model.head.weight.data[0, CH_LOGIT0] = LOGIT_SCALE
        model.head.weight.data[1, CH_LOGIT1] = LOGIT_SCALE


model_shorthand_name = "HandbuiltDec2Bin_v3_logplace"
model_description = (
    "2-block circuit. Block 0 attention's softmax IS the place-value weighting "
    "(k_j = log(place(j)) on digit positions, -inf elsewhere); attended * 111 = N. "
    "Block 0 MLP: 255 clipped-step ReLU pairs over N produce all 8 bits. "
    "Block 1 MLP: 16 AND-gates select bit_(10-p) at prediction position p. "
    "LayerNorms replaced with Identity. No place-value channel, no MLP0 lookup."
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
        d_model=36,
        n_heads=1,
        n_layers=2,
        d_ff=36,  # overridden per-block inside write_weights
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
