"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
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
    """Pre-LN removed: residual stream stays interpretable for hand-built weights."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 128,
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
# Hand-built weights: per-digit add WITHOUT carry propagation
# ---------------------------------------------------------------------------
#
# Vocab: "0123456789+="  (ids 0..9 = digits, 10='+', 11='=')
# Prompt: "A0 A1 A2 A3 A4 + B0 B1 B2 B3 B4 ="  (positions 0..11, MSD-first)
# Predict S_i at position 11+i for i=0..5 (autoregressive, using logits[:,-1,:]).
#
# This circuit:
#   - At output position p in {12..16}, attention gathers A_{p-12} into residual
#     dims 28..37 and B_{p-12} into 38..47 (each as a 10-d one-hot).
#   - MLP computes (A_{p-12} + B_{p-12}) mod 10 via 100 (a,b)-pair detectors.
#   - At p=11, attention output is forced to zero via a self-sink, and a
#     "p=11 fallback" MLP unit forces output token "0" (leading digit).
#   - No carry propagation, so accuracy ~ probability of no carry in any digit
#     pair ~= 0.55^5 ~= 5%.
#
# Residual stream layout (d_model = 64):
#   dims  0..17 : position one-hot (set by pos_emb)
#   dims 18..27 : token-digit one-hot (set by token_emb for digit tokens)
#   dims 28..37 : gathered A digit one-hot (set by attention head 0)
#   dims 38..47 : gathered B digit one-hot (set by attention head 1)
#   dims 48..57 : output digit logits (set by MLP)
#   dims 58..63 : unused
#
# Attention heads (n_heads=4, d_head=16):
#   head 0 retrieves A_{p-12}; head 1 retrieves B_{p-12}; heads 2, 3 unused.
#   Within each used head, dims 0..4 carry the position-matching score and
#   dim 5 carries a "self-sink" that catches the p=11 query (giving zero value).

def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        # Zero everything first.
        for p in model.parameters():
            p.zero_()

        D = model.d_model
        H = model.n_heads
        dh = D // H
        T = model.max_seq_len
        assert D == 64 and H == 4 and dh == 16 and T >= 18, (D, H, dh, T)

        # ---- Embeddings ----
        # Position embedding: pos p -> one-hot at dim p.
        for p in range(T):
            model.pos_emb.weight[p, p] = 1.0
        # Token embedding: digit d -> one-hot at dim 18+d. '+' and '=' left as zero.
        for d in range(10):
            model.token_emb.weight[d, 18 + d] = 1.0

        block = model.blocks[0]
        SCALE = 50.0  # sharpens the attention softmax to near-hard selection

        # ---- Attention: W_q ----
        # At output position p in {12..16}, head 0 and head 1 both query for
        # prompt slot (p - 12). Encode as one-hot in dim (p-12) within head.
        for i in range(5):
            block.attn.W_q.weight[i,        12 + i] = SCALE  # head 0
            block.attn.W_q.weight[dh + i,   12 + i] = SCALE  # head 1
        # At position 11 (the '='), query a self-sink dim (dim 5 of head).
        block.attn.W_q.weight[5,      11] = SCALE
        block.attn.W_q.weight[dh + 5, 11] = SCALE

        # ---- Attention: W_k ----
        # head 0's keys: at prompt A position q in {0..4}, key = one-hot dim q.
        for q in range(5):
            block.attn.W_k.weight[q, q] = 1.0
        # head 1's keys: at prompt B position 6+q for q in {0..4}, key = one-hot dim q.
        for q in range(5):
            block.attn.W_k.weight[dh + q, 6 + q] = 1.0
        # Both heads' self-sink: at position 11, key = one-hot in dim 5.
        block.attn.W_k.weight[5,      11] = 1.0
        block.attn.W_k.weight[dh + 5, 11] = 1.0

        # ---- Attention: W_v ----
        # Value reads the digit one-hot from dims 18..27 of the residual.
        for d in range(10):
            block.attn.W_v.weight[d,      18 + d] = 1.0  # head 0
            block.attn.W_v.weight[dh + d, 18 + d] = 1.0  # head 1

        # ---- Attention: W_o ----
        # Head 0's first 10 dims -> residual dims 28..37 (gathered A digit).
        # Head 1's first 10 dims -> residual dims 38..47 (gathered B digit).
        for d in range(10):
            block.attn.W_o.weight[28 + d, d] = 1.0
            block.attn.W_o.weight[38 + d, dh + d] = 1.0

        # ---- MLP fc1 ----
        # Hidden units 0..99: h_{a,b} = ReLU(A_oh[a] + B_oh[b] - 1).
        for a in range(10):
            for b in range(10):
                idx = a * 10 + b
                block.mlp.fc1.weight[idx, 28 + a] = 1.0
                block.mlp.fc1.weight[idx, 38 + b] = 1.0
                block.mlp.fc1.bias[idx] = -1.0
        # Hidden unit 100: p=11 fallback = ReLU(pos_dim_11 - 0.5).
        block.mlp.fc1.weight[100, 11] = 1.0
        block.mlp.fc1.bias[100] = -0.5

        # ---- MLP fc2 ----
        # For each (a,b) hidden unit, add 1 to dim 48 + ((a+b) mod 10).
        for a in range(10):
            for b in range(10):
                s = (a + b) % 10
                idx = a * 10 + b
                block.mlp.fc2.weight[48 + s, idx] = 1.0
        # p=11 fallback: writes +100 to dim 48 (digit '0' logit) and
        # -100 to dims 49..57 to dominate any spurious activity.
        block.mlp.fc2.weight[48, 100] = 200.0
        for s in range(1, 10):
            block.mlp.fc2.weight[48 + s, 100] = -200.0

        # ---- Head ----
        # Project residual dim (48 + t) -> token-t logit, for t in {0..9}.
        # '+' (t=10) and '=' (t=11) get zero logits and never win.
        for t in range(10):
            model.head.weight[t, 48 + t] = 1.0


model_shorthand_name = "PerDigitNoCarry"
model_description = (
    "1 layer, 4 heads. Attention gathers A_i and B_i to residual slots; MLP "
    "computes (A_i+B_i) mod 10 via 100 one-hot AND-units. No carry handling; "
    "p=11 fallback forces leading digit '0'."
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
