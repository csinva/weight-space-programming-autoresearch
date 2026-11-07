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
        # Identity normalizations: the hand-built circuit relies on exact
        # one-hot residual streams, so we bypass LayerNorm entirely.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer sized for a hand-built linear-interpolation circuit."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 16,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 5000,
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
    """Hand-built interpretable circuit for linear-interpolation-two-points.

    Task: prompt "X1,Y1Y1;X2,Y2Y2;XQ=" → answer "YqYq" where Yq = m*xq + b,
    m ∈ {1..5} and b ∈ {0..9} are integers, x's are single digits.

    The circuit uses a clearly-partitioned residual stream and 2 transformer
    layers with attention identities and lookup-table MLPs.

    Residual stream channels (d_model = 192):
      [ 0, 13)  token one-hot           (V=vocab_size=13)
      [13, 29)  position one-hot        (P=max_seq_len=16)
      [29, 39)  gathered x1     (digit 0-9 one-hot)
      [39, 49)  gathered y1_tens
      [49, 59)  gathered y1_ones
      [59, 69)  gathered x2
      [69, 79)  gathered y2_tens
      [79, 89)  gathered y2_ones
      [89, 99)  gathered xq
      [99,104)  m one-hot              (m ∈ {1..5}, 5 channels)
      [104,114) b one-hot              (b ∈ {0..9}, 10 channels)
      [114,192) unused

    Layer 0 attention (7 heads, one per gathered feature):
      For each useful head h with source position p_h, at output positions
      11 (the '=') and 12 (the just-generated yq_tens slot), softmax peaks
      on src=p_h and writes the source token's digit one-hot into the
      head's dedicated slot in the residual stream.

    Layer 0 MLP — lookup (m, b) from (x1, y1, x2, y2):
      One neuron per valid (m, b, x1, x2) tuple (4500 in total). The neuron
      fires (after ReLU) only when the six gathered one-hots all match
      x1=x1_i, y1=m_i*x1_i+b_i, x2=x2_i, y2=m_i*x2_i+b_i. It then writes
      m_i and b_i one-hots into their residual slots.

    Layer 1 attention: zeroed (no-op).

    Layer 1 MLP — compute yq=m*xq+b and emit digit:
      One neuron per (m, b, xq, pos∈{11,12}) tuple (1000 in total). The
      neuron fires only when m, b, xq match AND the current position equals
      pos. It writes the correct digit (yq // 10 if pos==11, yq % 10 if
      pos==12) into the token one-hot subspace with a large weight so the
      head argmax picks it.

    Head: identity over the token one-hot channels.
    """
    d_model = model.d_model       # 192
    n_heads = model.n_heads       # 8
    d_head = d_model // n_heads   # 24
    V = model.vocab_size          # 13
    P = model.max_seq_len         # 16
    d_ff = model.d_ff             # 5000

    # Channel offsets
    TOK = 0
    POS = V                    # 13
    CH_X1  = V + P             # 29
    CH_Y1T = CH_X1 + 10        # 39
    CH_Y1O = CH_X1 + 20        # 49
    CH_X2  = CH_X1 + 30        # 59
    CH_Y2T = CH_X1 + 40        # 69
    CH_Y2O = CH_X1 + 50        # 79
    CH_XQ  = CH_X1 + 60        # 89
    CH_M   = CH_X1 + 70        # 99 (5 channels: m=1..5)
    CH_B   = CH_M + 5          # 104 (10 channels: b=0..9)
    assert CH_B + 10 <= d_model, "d_model too small for hand-built layout"

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

        # --- Layer 0 attention: gather one input feature per head ---
        # Head h attends from output positions {11, 12} to its source p_h.
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
        # SCALE makes softmax peak sharply on src=p_h.
        # Pre-softmax score at matched (out∈{11,12}, src=p_h):
        #   q·k / sqrt(d_head) = SCALE / sqrt(24) ≈ SCALE / 4.9
        SCALE = 50.0
        for h, src, slot in head_specs:
            h_start = h * d_head
            # W_q: query at out positions 11 and 12 has 1.0 in component h_start
            attn0.W_q.weight[h_start, POS + 11] = 1.0
            attn0.W_q.weight[h_start, POS + 12] = 1.0
            # W_k: key at src position p_h has SCALE in component h_start
            attn0.W_k.weight[h_start, POS + src] = SCALE
            # W_v: at any source, copy the token's digit one-hot (vocab idx 0-9)
            # into components [h_start+1, h_start+11). For non-digit tokens
            # these components stay 0 — harmless since we only attend to digit positions.
            for c in range(10):
                attn0.W_v.weight[h_start + 1 + c, TOK + c] = 1.0
            # W_o: write the digit one-hot from head h into its residual slot
            for c in range(10):
                attn0.W_o.weight[slot + c, h_start + 1 + c] = 1.0

        # --- Layer 0 MLP: detect (m, b, x1, x2) → write (m, b) one-hots ---
        # 4500 neurons total (m=1..5, b=0..9, x1=0..9, x2=0..9, x2≠x1).
        # Each neuron checks 6 one-hot indicators sum to 6, ReLU(sum - 5.5) = 0.5.
        mlp0 = model.blocks[0].mlp
        neuron_idx = 0
        for m_i in range(1, 6):
            for b_i in range(10):
                for x1_i in range(10):
                    for x2_i in range(10):
                        if x2_i == x1_i:
                            continue
                        y1 = m_i * x1_i + b_i
                        y2 = m_i * x2_i + b_i
                        mlp0.fc1.weight[neuron_idx, CH_X1  + x1_i]   = 1.0
                        mlp0.fc1.weight[neuron_idx, CH_Y1T + y1 // 10] = 1.0
                        mlp0.fc1.weight[neuron_idx, CH_Y1O + y1 % 10]  = 1.0
                        mlp0.fc1.weight[neuron_idx, CH_X2  + x2_i]   = 1.0
                        mlp0.fc1.weight[neuron_idx, CH_Y2T + y2 // 10] = 1.0
                        mlp0.fc1.weight[neuron_idx, CH_Y2O + y2 % 10]  = 1.0
                        mlp0.fc1.bias[neuron_idx] = -5.5
                        mlp0.fc2.weight[CH_M + (m_i - 1), neuron_idx] = 1.0
                        mlp0.fc2.weight[CH_B + b_i,       neuron_idx] = 1.0
                        neuron_idx += 1
        assert neuron_idx <= d_ff, f"layer-0 MLP needs d_ff ≥ {neuron_idx}"

        # --- Layer 1 attention: identically zero (already zeroed above) ---

        # --- Layer 1 MLP: detect (m, b, xq, pos) → write digit into vocab one-hot ---
        # 1000 neurons total (m=1..5, b=0..9, xq=0..9, pos ∈ {11, 12}).
        # m and b channels carry value 0.5 each (from layer-0 MLP firing exactly
        # one neuron with ReLU(0.5)). xq and pos channels carry value 1.0.
        # Sum on match = 0.5 + 0.5 + 1.0 + 1.0 = 3.0; ReLU(sum - 2.5) = 0.5.
        # Output channel TOK+digit gets weight 100, contributing 50, which
        # dwarfs the input token's one-hot value of 1 at the head step.
        mlp1 = model.blocks[1].mlp
        neuron_idx = 0
        for m_i in range(1, 6):
            for b_i in range(10):
                for xq_i in range(10):
                    for pos_i in (11, 12):
                        yq = m_i * xq_i + b_i
                        digit = (yq // 10) if pos_i == 11 else (yq % 10)
                        mlp1.fc1.weight[neuron_idx, CH_M + (m_i - 1)] = 1.0
                        mlp1.fc1.weight[neuron_idx, CH_B + b_i]       = 1.0
                        mlp1.fc1.weight[neuron_idx, CH_XQ + xq_i]     = 1.0
                        mlp1.fc1.weight[neuron_idx, POS + pos_i]      = 1.0
                        mlp1.fc1.bias[neuron_idx] = -2.5
                        mlp1.fc2.weight[TOK + digit, neuron_idx] = 100.0
                        neuron_idx += 1
        assert neuron_idx <= d_ff, f"layer-1 MLP needs d_ff ≥ {neuron_idx}"

        # --- Head: identity over the token one-hot channels ---
        for c in range(V):
            model.head.weight[c, TOK + c] = 1.0


model_shorthand_name = "LinInterpLookup_v1"
model_description = (
    "2-layer transformer with LN bypassed. Layer-0 attention gathers x1,y1,x2,y2,xq "
    "one-hots into dedicated residual slots; layer-0 MLP is a (m,b,x1,x2)-indexed "
    "lookup writing m,b one-hots; layer-1 MLP is an (m,b,xq,pos)-indexed lookup "
    "writing yq tens digit at pos 11 and ones digit at pos 12."
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
