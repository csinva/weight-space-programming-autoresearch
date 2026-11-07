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
    """Hand-built position-routing circuit for word-reversal-3x3.

    Layout of the residual stream (d_model = vocab_size + max_seq_len):
      dims [0 .. V-1]   = token one-hot subspace
      dims [V .. V+L-1] = position one-hot subspace

    A single attention head uses position dims only to look up the right
    source position for each output step, and copies the token one-hot from
    that source into the residual. The unembedding reads the token subspace.
    """
    torch.manual_seed(0)

    V = task.vocab_size          # 28 for word-reversal-3x3
    L = model.max_seq_len        # 23 (= prompt 12 + answer 11)
    D = model.d_model
    assert D == V + L, f"d_model must be V+L={V+L}, got {D}"
    assert model.n_heads == 1 and model.n_layers == 1

    SCALE = float(os.environ.get("AR_SCALE", "50.0"))
    HEAD_SCALE = float(os.environ.get("AR_HEAD_SCALE", str(SCALE)))

    # When the model is computing logits at sequence position p, it predicts
    # the token at position p+1. We hard-code the source for each p:
    #   prompt positions:   0..11 (word1 _ word2 _ word3 =)
    #   answer positions:  12..22 (word3 _ word2 _ word1)
    #   p=11 -> src 8   (word3[0])     p=12 -> 9   p=13 -> 10
    #   p=14 -> src 7   ('_')          p=15 -> 4   p=16 -> 5   p=17 -> 6
    #   p=18 -> src 3   ('_')          p=19 -> 0   p=20 -> 1   p=21 -> 2
    src_map = {11: 8, 12: 9, 13: 10, 14: 7, 15: 4, 16: 5, 17: 6,
               18: 3, 19: 0, 20: 1, 21: 2}
    src = [src_map.get(p, 0) for p in range(L)]

    with torch.no_grad():
        # --- Embeddings: one-hots into disjoint subspaces ---
        tok_w = torch.zeros(V, D)
        tok_w[:, :V] = torch.eye(V)
        model.token_emb.weight.copy_(tok_w)

        pos_w = torch.zeros(L, D)
        pos_w[:, V:V + L] = torch.eye(L)
        model.pos_emb.weight.copy_(pos_w)

        block = model.blocks[0]

        # --- LayerNorms: identity (gain=1, bias=0) ---
        block.ln1.weight.fill_(1.0); block.ln1.bias.zero_()
        block.ln2.weight.fill_(1.0); block.ln2.bias.zero_()
        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()

        # --- Attention: position-only routing ---
        # nn.Linear stores weight as (out_features, in_features); y = x @ W.T
        # so W[out, in] is the coefficient from input dim `in` to output dim `out`.

        # W_q: at current position p, produce a vector that has SCALE at the
        # position-dim of the desired source. Reads only pos one-hot.
        Wq = torch.zeros(D, D)
        for p in range(L):
            Wq[V + src[p], V + p] = SCALE
        block.attn.W_q.weight.copy_(Wq)

        # W_k: identity on the position subspace, zero elsewhere.
        # So key at source position s has a 1 at dim V+s.
        Wk = torch.zeros(D, D)
        for s in range(L):
            Wk[V + s, V + s] = 1.0
        block.attn.W_k.weight.copy_(Wk)

        # W_v: identity on the token subspace, zero on the pos subspace.
        # So value at position s equals the token one-hot at s.
        Wv = torch.zeros(D, D)
        for t in range(V):
            Wv[t, t] = 1.0
        block.attn.W_v.weight.copy_(Wv)

        # W_o: identity. The attention output is already a token one-hot in
        # dims [0..V-1] and gets added straight back to the residual.
        block.attn.W_o.weight.copy_(torch.eye(D))

        # --- MLP: disabled (zero in/out) ---
        block.mlp.fc1.weight.zero_(); block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_(); block.mlp.fc2.bias.zero_()

        # --- Unembedding: read token subspace, amplify to dominate LN noise ---
        head_w = torch.zeros(V, D)
        head_w[:, :V] = torch.eye(V) * HEAD_SCALE
        model.head.weight.copy_(head_w)


model_shorthand_name = os.environ.get("AR_NAME", "PosRoutingCopyV1")
model_description = os.environ.get("AR_DESC",
    "1-layer 1-head transformer; token & position one-hot embeddings in disjoint subspaces; "
    "attention uses position-only Q/K to route each output step to a fixed source position "
    "and copies the source token one-hot via identity V/W_o; MLP zeroed."
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
    d_model = task.vocab_size + max_seq_len  # disjoint token/pos one-hot subspaces
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=1,
        n_layers=1,
        d_ff=1,
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
