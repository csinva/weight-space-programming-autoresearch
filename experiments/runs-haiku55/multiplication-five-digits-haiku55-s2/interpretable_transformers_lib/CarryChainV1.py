"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. Task: multiply two 5-digit numbers (`multiplication-five-digits`):
prompt "12345*67890=" -> answer "0838102050" (10 zero-padded digits, MSD first).

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task multiplication-five-digits --n-samples 500
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
    """Pre-residual block. No LayerNorm: residual values stay exact integers/one-hots."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer with no LayerNorm, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 280,
        n_heads: int = 10,
        n_layers: int = 11,
        d_ff: int = 2500,
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
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

# Residual-stream layout (d_model = 280; dims 265..279 unused).
#   0..11    token one-hot (0-9 digits, 10 '*', 11 '=')
#   12..33   position one-hot (positions 0..21)
#   34       constant 1 (bias channel, present at every position)
#   35..134  gathered digits: slot h (0-4 = a_h, 5-9 = b_{h-5}) x digit d
#   135..143 column sums c_s = sum_{i+j=s} a_i*b_j, s = 0..8
#   144..152 carries into column s (scalar), s = 0..8
#   153..252 output digit one-hot: place p (0 = 10^0 ... 9 = 10^9) x digit e
#   253..264 output logits (vocab order)
TOK0, POS0, BIAS = 0, 12, 34
GATH0, C0, CARRY0, DIG0, OUT0 = 35, 135, 144, 153, 253
N_HEADS, D_HEAD = 10, 28
K_MAX = 461          # column value v = c_s + carry_s is at most 405 + 45 = 450
PROMPT_ANS_POS = 11  # index of '=' ; answer digit t is predicted at position 11 + t


def gath(h: int, d: int) -> int:
    return GATH0 + 10 * h + d


def dig(p: int, d: int) -> int:
    return DIG0 + 10 * p + d


def attended_pos(h: int) -> int:
    """Prompt position whose digit head h copies: a_h at pos h, b_j at pos 6+j."""
    return h if h < 5 else h + 1


def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-built multiplier: gather digits, AND-gate pair products into column sums,
    propagate carries column by column with ReLU threshold indicators, read out by position.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: token one-hot + constant bias channel; position one-hot.
        for t in range(12):
            model.token_emb.weight[t, TOK0 + t] = 1.0
            model.token_emb.weight[t, BIAS] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS0 + p] = 1.0

        # Block 0 attention: head h copies the digit at a fixed prompt position.
        # Query is a constant (driven by the bias channel) that peaks on key position p_h.
        att = model.blocks[0].attn
        beta = 30.0 * math.sqrt(D_HEAD)  # scores = beta / sqrt(d_head) = 30 at the target
        for h in range(N_HEADS):
            base = h * D_HEAD
            p_h = attended_pos(h)
            att.W_q.weight[base + p_h, BIAS] = beta
            for p in range(model.max_seq_len):
                att.W_k.weight[base + p, POS0 + p] = 1.0
            for d in range(10):
                att.W_v.weight[base + d, TOK0 + d] = 1.0
                att.W_o.weight[gath(h, d), base + d] = 1.0

        # Block 0 MLP: AND gate for every (a_i = d) & (b_j = e) pair; sum d*e into column 8-i-j.
        mlp = model.blocks[0].mlp
        u = 0
        for i in range(5):
            for j in range(5):
                for d in range(10):
                    for e in range(10):
                        mlp.fc1.weight[u, gath(i, d)] = 1.0
                        mlp.fc1.weight[u, gath(5 + j, e)] = 1.0
                        mlp.fc1.bias[u] = -1.0          # relu(x + y - 1) = AND(x, y)
                        # prompt is MSD-first: a_i has place value 10^(4-i), b_j 10^(4-j)
                        mlp.fc2.weight[C0 + (8 - i - j), u] = float(d * e)
                        u += 1

        # Blocks 1..9: column s. v = c_s + carry_s; digit_s = v mod 10; carry_{s+1} = v // 10.
        # Indicator 1[v = k] = relu(v-k+1) - 2 relu(v-k) + relu(v-k-1) (exact on integers).
        for s in range(9):
            mlp = model.blocks[s + 1].mlp
            u = 0
            for k in range(K_MAX):
                for sign, bias in ((1.0, -k + 1.0), (-2.0, -k * 1.0), (1.0, -k - 1.0)):
                    mlp.fc1.weight[u, C0 + s] = 1.0
                    if s > 0:
                        mlp.fc1.weight[u, CARRY0 + s] = 1.0
                    mlp.fc1.bias[u] = bias
                    mlp.fc2.weight[dig(s, k % 10), u] += sign
                    if s < 8:
                        mlp.fc2.weight[CARRY0 + s + 1, u] += sign * (k // 10)
                    elif k // 10 < 10:
                        mlp.fc2.weight[dig(9, k // 10), u] += sign   # top digit = carry out
                    u += 1

        # Block 10: output read-out. Position 11 + t emits digit of place 9 - t.
        mlp = model.blocks[10].mlp
        u = 0
        for t in range(10):
            for e in range(10):
                mlp.fc1.weight[u, POS0 + PROMPT_ANS_POS + t] = 1.0
                mlp.fc1.weight[u, dig(9 - t, e)] = 1.0
                mlp.fc1.bias[u] = -1.0
                mlp.fc2.weight[OUT0 + e, u] = 1.0
                u += 1
        for e in range(10):
            model.head.weight[e, OUT0 + e] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CarryChainV1"
model_description = (
    "Hand-built circuit, no LayerNorm: block 0 attention gathers the 10 digits to each position, "
    "block 0 MLP ANDs every (a_i,b_j) digit pair and sums d*e into column 8-i-j, blocks 1-9 propagate "
    "carries one column each via ReLU threshold indicators on v=c_s+carry_s, block 10 reads out digit "
    "by position."
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
