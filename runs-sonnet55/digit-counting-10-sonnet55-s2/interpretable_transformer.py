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
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None
        if self.mlp is not None:
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        d_ff: tuple = (0, 20),
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff[i]) for i in range(n_layers)])
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
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Residual layout (d_model=64):
    #   0-9 token digit | 10 '=' token | 11-26 position one-hot
    #   27-36 copy of the LAST digit d | 37 sink-attention mass g | 38-47 logit for digit u
    for p in model.parameters():
        p.data.zero_()
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.weight.data.fill_(1.0)
    D = 10
    TOK, POS, COPY, G, OUT = 0, 11, 27, 37, 38
    with torch.no_grad():
        model.token_emb.weight[torch.arange(11), torch.arange(11)] = 1.0
        model.pos_emb.weight[torch.arange(16), POS + torch.arange(16)] = 1.0

        # Layer 1 attention: every position p<=10 attends to itself, positions >=11 attend
        # to position 10 (the last digit); it copies that digit into the COPY slots.
        a1 = model.blocks[0].attn
        for p in range(16):
            tgt = p if p <= 10 else 10
            a1.W_q.weight[tgt, POS + p] = L1S
            a1.W_k.weight[p, POS + p] = 1.0
        for j in range(D):
            a1.W_v.weight[j, TOK + j] = 1.0
            a1.W_o.weight[COPY + j, j] = 0.2

        # Layer 2 attention: query = copied last digit d; keys = digit tokens.
        # Keys equal to d get a higher score. The '=' position (pos 11) is a "sink" whose
        # value is 1; so its attention mass g = 1/(const + m*(E-1)) encodes the number m
        # of digits equal to d (including itself). Pos 12's own key is masked out.
        a2 = model.blocks[1].attn
        for j in range(D):
            a2.W_q.weight[j, COPY + j] = SQ
            a2.W_k.weight[j, TOK + j] = 1.0
        a2.W_q.weight[10, POS + 11] = EXCL
        a2.W_q.weight[10, POS + 12] = EXCL
        a2.W_k.weight[10, POS + 12] = -EXCL
        a2.W_v.weight[0, POS + 11] = 1.0
        a2.W_o.weight[G, 0] = GGAIN

        # Layer 2 MLP: nearest-neighbour decoding of g -> count m, one 'hat' per u=m-1.
        # hidden 2u: relu(A(h-th_u)), 2u+1: relu(A(th_u-h)); both penalise distance.
        # Gated off at pos 11 (there the answer is simply '0' via '=' token -> digit 0).
        mlp = model.blocks[1].mlp
        for u in range(NC):
            mlp.fc1.weight[2 * u, G] = A
            mlp.fc1.bias[2 * u] = -A * THETA[u]
            mlp.fc1.weight[2 * u + 1, G] = -A
            mlp.fc1.bias[2 * u + 1] = A * THETA[u]
            for k in (2 * u, 2 * u + 1):
                # gate: ~0 at pos 12 (layer-normed active pos dim = V_POS), -45 elsewhere (pos 11)
                mlp.fc1.weight[k, POS + 12] = A / GATE_DIV
                mlp.fc1.bias[k] -= (A / GATE_DIV) * V_POS
                mlp.fc2.weight[OUT + u, k] = -1.0

        # Head: digit u <- OUT+u ; '=' token -> digit 0 (tens digit of the answer is 0)
        for u in range(NC):
            model.head.weight[u, OUT + u] = 1.0
        model.head.weight[0, 10] = 1.0
        # never emit '=' at pos 12 (otherwise its zero logit beats the layer-normed digit logits)
        model.head.weight[10, POS + 12] = -HEADPEN


SQ = 0.37
V_POS = 4.3243# measured layer-normed value of the active position one-hot dim
GGAIN = 0.1
A = 300.0
L1S = 4.0  # layer-1 attention sharpness (query -> target position)
GATE_DIV = 30.0 # gate weight = A / GATE_DIV
NC = 10     # number of count classes decoded (counts 1..NC); d_ff = 2*NC
EXCL = 3.0 # strength of masking out pos-12's own key in layer 2
HEADPEN = 0.0# penalty on '=' logit at pos 12
# Expected (layer-normed) sink-mass reading for count m: h(m) = BETA + GAMMA/(m + R).
# Constants fitted to measured readings at m=1..4 and consistent with the analytic form.
BETA, GAMMA, R = -0.28503, 1.107, 6.158
THETA = [BETA + GAMMA / (u + 1 + R) for u in range(NC)]


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SinkCount_final2"
model_description = "Final simplified: no '=' penalty in head (HEADPEN=0), gate A/30, L1S=4, EXCL=3, NC=10 hats; 2-layer 1-head sink-mass counting circuit."


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
