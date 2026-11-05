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
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
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
        d_model: int = 33,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 0,
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

#
# Task: "w1_w2_w3=" -> "w3_w2_w1" (three 3-letter words, reverse the word order).
# Prompt occupies positions 0..11 (w1 = 0..2, '_' = 3, w2 = 4..6, '_' = 7,
# w3 = 8..10, '=' = 11). The answer token j is produced at position 11+j and
# must equal the prompt token at SRC_POS[j]: a fixed position permutation.
#
# Residual layout (d_model = 33):
#   dims   0..4    token code: 5-bit +/-1 code   (token_emb)
#   dims   5..27   position one-hot              (pos_emb)
#   dims  28..32   copied token code             (written by attention W_o)
#
# One attention head does the whole job: the query at position 11+j has its
# position one-hot routed to key-space dim SRC_POS[j]; keys are their own
# position one-hot, so the score is large only at the source position.
# The value carries the token code, which W_o writes into the copy block.
# The unembedding is a dot product with the codes, so the copied token wins.
# MLPs are zeroed (inert).

TOK_BITS = 5
TOK_OFF = 0
POS_OFF = 5
COPY_OFF = 28
ATTN_GAIN = 30.0
SRC_POS = [8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2]  # answer index j -> prompt position


def _bit_code(t: int) -> list[float]:
    """Token id -> 5-bit code in {-1, +1}. Hamming-distance-based dot products
    then have a unique maximum at the matching token."""
    return [1.0 if (t >> k) & 1 else -1.0 for k in range(TOK_BITS)]


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    prompt_len = task.prompt_len
    V = task.vocab_size
    P = model.max_seq_len
    blk = model.blocks[0]
    with torch.no_grad():
        # Zero everything, then set LayerNorm to identity (weight 1, bias 0).
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.zero_()
                if getattr(m, "bias", None) is not None:
                    m.bias.zero_()

        # Embeddings: each token is a 5-bit +/-1 code; position is one-hot.
        for t in range(V):
            for k in range(TOK_BITS):
                model.token_emb.weight[t, TOK_OFF + k] = _bit_code(t)[k]
        for p in range(P):
            model.pos_emb.weight[p, POS_OFF + p] = 1.0

        # Attention: query at answer step j routes to its source position.
        for j in range(task.answer_len):
            q = prompt_len - 1 + j
            blk.attn.W_q.weight[POS_OFF + SRC_POS[j], POS_OFF + q] = ATTN_GAIN
        for p in range(P):
            blk.attn.W_k.weight[POS_OFF + p, POS_OFF + p] = 1.0
        for k in range(TOK_BITS):
            blk.attn.W_v.weight[TOK_OFF + k, TOK_OFF + k] = 1.0
            blk.attn.W_o.weight[COPY_OFF + k, TOK_OFF + k] = 1.0

        # Unembedding: logit(t) = <code(t), copy bits>; unique max at the copied token.
        for t in range(V):
            for k in range(TOK_BITS):
                model.head.weight[t, COPY_OFF + k] = _bit_code(t)[k]


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "copy-bits5-d33-noLN-g30"
model_description = "No-LayerNorm bits5 circuit (d_ff=0) at attention gain 30: checks the upper end of the no-LN gain range that works."


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
