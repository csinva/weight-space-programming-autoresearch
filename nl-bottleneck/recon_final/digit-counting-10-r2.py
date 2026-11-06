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


class Block(nn.Module):
    """Attention-only block: no MLP, no LayerNorm."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 14,
        d_model: int = 3,
        n_heads: int = 1,
        n_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Channels: 0,1 = digit on the unit circle (angle 2*pi*d/10); 2 = signed flag.

    Flag: +1 at query (pos 10), -1 at '=' (pos 11) and tens digit (pos 12), 0 on inputs.
    L1: circle dot-product attention, scaled so a match beats the nearest non-match by
        40 logits; value = flag -> flag. Query flag -> 1 + 1/(k+1). '=' and the tens
        position sit off the unit circle (radius > 1) and attend only to themselves -> -2.
    L2: score = -60 f_q f_k; negative-flag positions attend to the query and add its
        flag -> rho - 1, rho = 1/(k+1).
    Readout: logit_u = 2 rho/(u+1) - 1/(u+1)^2 (nearest prototype on rho). The flag
        coefficient is 2/(u+1); the constants come from the circle channels, which also
        suppress digits > 1 at '=', put the tens threshold between k=9 and k=10, and
        force units '0' after tens '1'.
    """
    torch.manual_seed(0)
    V = task.vocab_size
    eq = task.stoi['=']
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        ang = lambda d: (math.cos(2 * math.pi * d / 10), math.sin(2 * math.pi * d / 10))
        # geometry of the readout positions (circle channels)
        p12 = (0.0, 2.0)                                     # circle offset at tens position
        x0 = (ang(0)[0] + p12[0], ang(0)[1] + p12[1])        # tens position, tens digit '0'
        x1 = (ang(1)[0] + p12[0], ang(1)[1] + p12[1])        # tens position, tens digit '1'
        beta = 0.5
        xe = (beta * x1[0], beta * x1[1])                    # '=' embedding (off circle)
        # embeddings
        for d in range(10):
            model.token_emb.weight[task.stoi[str(d)], 0] = ang(d)[0]
            model.token_emb.weight[task.stoi[str(d)], 1] = ang(d)[1]
        model.token_emb.weight[eq, 0] = xe[0]
        model.token_emb.weight[eq, 1] = xe[1]
        model.pos_emb.weight[10, 2] = 1.0
        model.pos_emb.weight[11, 2] = -1.0
        model.pos_emb.weight[12, 0] = p12[0]
        model.pos_emb.weight[12, 1] = p12[1]
        model.pos_emb.weight[12, 2] = -1.0
        # layer 1: same-digit matching, counts by softmax
        dh = model.d_model // model.n_heads
        scale = 40.0 / (1 - math.cos(2 * math.pi / 10))
        g = math.sqrt(scale * math.sqrt(dh))
        a1 = model.blocks[0].attn
        a1.W_q.weight[0, 0] = g; a1.W_q.weight[1, 1] = g
        a1.W_k.weight[0, 0] = g; a1.W_k.weight[1, 1] = g
        a1.W_v.weight[2, 2] = 1.0
        a1.W_o.weight[2, 2] = 1.0
        # layer 2: negative flags fetch the query flag
        g2 = math.sqrt(60.0 * math.sqrt(dh))
        a2 = model.blocks[1].attn
        a2.W_q.weight[0, 2] = g2
        a2.W_k.weight[0, 2] = -g2
        a2.W_v.weight[2, 2] = 1.0
        a2.W_o.weight[2, 2] = 1.0
        # readout: solve the 2x2 system [x0; x1] (a, b) = (K, B) in closed form
        M = 10.0
        rho_star = 0.5 * (1 / 10 + 1 / 11)
        det = x0[0] * x1[1] - x0[1] * x1[0]
        for v in range(V):
            if v == eq:
                K, D, fc = -M, -M, 0.0
            else:
                u = int(task.itos[v])
                fc = 2.0 / (u + 1)
                K = 2.0 / (u + 1) - 1.0 / (u + 1) ** 2      # after tens '0'
                D = {0: 1 - rho_star, 1: 0.0}.get(u, -M)    # at '=' (tens readout)
            Bc = D / beta                                   # after tens '1' (since xe = beta*x1)
            a = (K * x1[1] - Bc * x0[1]) / det
            b = (Bc * x0[0] - K * x1[0]) / det
            model.head.weight[v, 0] = a
            model.head.weight[v, 1] = b
            model.head.weight[v, 2] = fc


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CircleFlagSoftmaxCount"
model_description = "2-layer attn-only width-3: circle digit channels + signed flag; L1 softmax counts matches into query flag 1+1/(k+1); L2 copies it to '='/tens giving rho-1; linear nearest-prototype readout on rho with circle-channel constants."


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
