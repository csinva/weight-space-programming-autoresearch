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
    """Single head with narrow q/k (d_qk=7: word one-hot 3 + char one-hot 4) and value width d_v (= vocab)."""

    D_QK = 7

    def __init__(self, d_model: int, n_heads: int, d_v: int = 28):
        super().__init__()
        self.W_q = nn.Linear(d_model, self.D_QK, bias=False)
        self.W_k = nn.Linear(d_model, self.D_QK, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.W_o = nn.Linear(d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        scores = (self.W_q(x) @ self.W_k(x).transpose(-2, -1)) / math.sqrt(self.D_QK)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return self.W_o(attn @ self.W_v(x))


class Block(nn.Module):
    """Attention-only residual block: this task is pure routing, so no MLP and no LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


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
        self.final_ln = nn.Identity()  # argmax over token dims does not need normalisation
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tied unembedding: copied token one-hot -> same token's logit

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
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Keep deterministic so reruns of the same code produce the same model.
    V, P = task.vocab_size, task.prompt_len  # 28 tokens, prompt length 12
    T = task.seq_len                          # 23
    # residual layout: [0,V) token one-hot | [V,V+T) position one-hot | [V+T,2V+T) copied token
    TOK, POS, OUT = 0, V, 0
    # answer index j (output predicted at position P-1+j) copies from prompt position SRC[j]
    SRC = [8, 9, 10, 7, 4, 5, 6, 7, 0, 1, 2]  # both underscores read from the second separator
    QG, OG = 10.0, 2.0  # query gain (attention sharpness), output gain; no LN: own token = 1, attended token = OG
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for i in range(V):
            model.token_emb.weight[i, TOK + i] = 1.0
        # Factorised position code: (word w in 0..2 one-hot | char c in 0..3 one-hot).
        # key code dims POS+0..6 on prompt positions 0..10; query code dims POS+7..13 on answer steps,
        # holding the (word, char) of the source position to read.
        # Key code (dims 0..6): word one-hot (0..2) | letter-index one-hot (3..5) | separator flag (6).
        # Query code (dims 7..13) is the answer position's OWN (word, letter) or separator flag;
        # W_q applies the word reversal as an anti-identity on the 3 word dims.
        NK = len(SRC)
        for t in range(NK):
            if t % 4 == 3:
                model.pos_emb.weight[t, POS + 6] = 1.0
            else:
                model.pos_emb.weight[t, POS + t // 4] = 1.0
                model.pos_emb.weight[t, POS + 3 + t % 4] = 1.0
        for j in range(NK):
            if j % 4 == 3:
                model.pos_emb.weight[P - 1 + j, POS + 7 + 6] = 1.0
            else:
                model.pos_emb.weight[P - 1 + j, POS + 7 + j // 4] = 1.0
                model.pos_emb.weight[P - 1 + j, POS + 7 + 3 + j % 4] = 1.0
        blk = model.blocks[0]
        for d in range(7):
            blk.attn.W_k.weight[d, POS + d] = 1.0          # key: my (word, letter) / separator flag
        for w in range(3):
            blk.attn.W_q.weight[2 - w, POS + 7 + w] = QG   # word reversal: output word w reads input word 2-w
        for d in range(3, 7):
            blk.attn.W_q.weight[d, POS + 7 + d] = QG       # letter index / separator flag copied through
        for i in range(V):
            blk.attn.W_v.weight[i, TOK + i] = 1.0          # value: token identity
            blk.attn.W_o.weight[OUT + i, i] = OG           # write into token dims, outweighing own token
            model.head.weight[i, OUT + i] = 1.0                  # unembed copied token


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "AntiIdentityCanonical"
model_description = "Canonical final config: AntiIdentityReversal, QG=10, OG=2 (W_q anti-identity on word dims implements the word reversal; separators matched by a flag dim)."


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
    max_seq_len = task.seq_len - 1  # the final answer token is never fed as input
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len,
                              d_model=task.vocab_size + 14, n_heads=1, n_layers=1, d_ff=1)
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
