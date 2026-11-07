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
import itertools
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

N_STEPS = 4  # one Block with shared weights is applied this many times


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
        self.ln1 = nn.Identity()  # no LayerNorm: registers hold literal 0/1 values
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
        d_model: int = 30,
        n_heads: int = 3,
        n_layers: int = 1,
        d_ff: int = 13,
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
        for _ in range(N_STEPS):          # the same block is clocked N_STEPS times
            for block in self.blocks:
                h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """A tiny CPU living at the '=' token: a single block is clocked 4 times.
    A program counter register steers the heads to (op_L, bit_L); the MLP is
    the transition table state <- state op bit and also advances the counter."""
    D, H = model.d_model, model.n_heads
    dh = D // H
    TOK, POS, S, OP, BIT, PC = 0, 6, 16, 17, 20, 22  # named registers in x
    AND, OR, XOR = OP, OP + 1, OP + 2
    B0, B1 = BIT, BIT + 1
    EQ = TOK + 5
    BETA = 30.0 * math.sqrt(dh)                  # hard (argmax-like) attention

    for p in model.parameters():
        p.data.zero_()
    for t in range(6):
        model.token_emb.weight.data[t, TOK + t] = 1.0
    for p in range(10):
        model.pos_emb.weight.data[p, POS + p] = 1.0
    model.pos_emb.weight.data[9, PC] = 1.0       # the counter boots at step 1 on '='

    a, m = model.blocks[0].attn, model.blocks[0].mlp
    for h in range(H):                           # every head keys on position
        for p in range(10):
            a.W_k.weight.data[h * dh + p, POS + p] = 1.0
    for L in range(1, 5):
        pc = PC + L - 1
        a.W_q.weight.data[0 * dh + 2 * L - 1, pc] = BETA   # head 0: op_L
        a.W_q.weight.data[1 * dh + 2 * L, pc] = BETA       # head 1: bit_L
        # head 2: load b0 at step 1, afterwards read '=' (whose value is zero)
        a.W_q.weight.data[2 * dh + (0 if L == 1 else 9), pc] = BETA
    for j in range(3):
        a.W_v.weight.data[j, TOK + 2 + j] = 1.0
        a.W_o.weight.data[OP + j, j] = 1.0
    for j in range(2):
        a.W_v.weight.data[dh + j, TOK + j] = 1.0
        a.W_o.weight.data[BIT + j, dh + j] = 1.0
    a.W_v.weight.data[2 * dh, TOK + 1] = 1.0
    a.W_o.weight.data[S, 2 * dh] = 1.0

    # Transition table: only the four rules that change the state.
    rules = [(1, AND, B0, -1.0), (0, OR, B1, 1.0),
             (1, XOR, B1, -1.0), (0, XOR, B1, 1.0)]
    for u, (s, op, b, delta) in enumerate(rules):
        m.fc1.weight.data[u, S] = 1.0 if s else -1.0
        m.fc1.weight.data[u, op] = 1.0
        m.fc1.weight.data[u, b] = 1.0
        m.fc1.bias.data[u] = -2.0 if s else -1.0
        m.fc2.weight.data[S, u] = delta
    for j, r in enumerate([AND, OR, XOR, B0, B1]):  # clear the read registers
        m.fc1.weight.data[4 + j, r] = 1.0
        m.fc2.weight.data[r, 4 + j] = -1.0
    for L in range(4):                           # PC <- PC + 1
        m.fc1.weight.data[9 + L, PC + L] = 1.0
        m.fc2.weight.data[PC + L, 9 + L] = -1.0
        if L < 3:
            m.fc2.weight.data[PC + L + 1, 9 + L] = 1.0

    w = model.head.weight.data                   # S -> logit, EQ supplies the bias
    w[0, S], w[0, EQ], w[1, S], w[1, EQ] = -1.0, 0.5, 1.0, -0.5
    verify(model, task)


@torch.no_grad()
def verify(model, task) -> None:
    """Exhaustive check on all 2592 prompts (test only; nothing here touches weights)."""
    def fold(s):
        v = int(s[0])
        for i in range(1, 9, 2):
            b = int(s[i + 1])
            v = v & b if s[i] == "&" else v | b if s[i] == "|" else v ^ b
        return str(v)
    ex = task.generate_examples(5000, seed=123)
    agree = sum(fold(e.prompt) == e.answer for e in ex) / len(ex)
    prompts = ["".join(c) + "=" for c in itertools.product(
        "01", "&|^", "01", "&|^", "01", "&|^", "01", "&|^", "01")]
    ids = torch.tensor([task.encode(p) for p in prompts])
    pred = model(ids)[:, -1].argmax(-1).tolist()
    ok = sum(task.itos[i] == fold(p) for i, p in zip(pred, prompts))
    print(f"VERIFY fold_vs_task={agree:.4f} exhaustive={ok}/{len(prompts)}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "E02_tied_block_pc"
model_description = "One block with shared weights clocked 4 times; a program-counter register steers attention to (op_L, bit_L) and the MLP advances it."


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
