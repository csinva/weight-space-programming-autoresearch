"""
Interpretable transformer for decimal-to-binary-8bit: "197=" -> "11000101".

The model is never trained. write_weights() sets every parameter in closed form.
Algorithm: greedy binary expansion, MSB first. At output position p (bit k=10-p)
    r = n - sum(emitted bits * 2^j),   emit 1 iff r >= 2^k.
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
    """Attention only: no LayerNorm and no MLP, the task is linear in attended tokens."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 22,
                 n_heads: int = 11, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
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

@torch.no_grad()
def check_all_inputs(model, task) -> None:
    """Greedy-decode all 256 inputs; a test only, the model never sees the targets."""
    good, margin = 0, float("inf")
    for n in range(256):
        ids = task.encode(f"{n:03d}=")
        for _ in range(task.answer_len):
            logits = model(torch.tensor([ids]))[0, -1]
            top2 = logits.topk(2).values
            margin = min(margin, float(top2[0] - top2[1]))
            ids = ids + [int(logits.argmax())]
        good += task.decode(ids[task.prompt_len:]) == f"{n:08b}"
    print(f"exhaustive: {good}/256, min logit margin {margin:.4f}")


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    H = 11
    dh = model.d_model // H
    VAL, ONE, THR = 11, 12, 13   # feature dims; dims 0..10 are the one-hot position
    attn = model.blocks[0].attn
    for d in range(10):
        model.token_emb.weight[task.stoi[str(d)], VAL] = d
    for p in range(model.max_seq_len):
        model.pos_emb.weight[p, ONE] = 1.0
        if p < H:
            model.pos_emb.weight[p, p] = 1.0
        if 3 <= p <= 10:
            model.pos_emb.weight[p, THR] = 0.5 - 2 ** (10 - p)
    # place value contributed by the token read by head j
    place = [100, 10, 1, 0, -128, -64, -32, -16, -8, -4, -2]
    for j in range(H):
        attn.W_q.weight[j * dh, ONE] = math.sqrt(dh)        # constant query
        attn.W_k.weight[j * dh, j] = 40.0                   # look at position j...
        attn.W_k.weight[j * dh, 3] += 20.0                  # ...else the zero-valued '='
        attn.W_v.weight[j * dh, VAL] = 1.0
        attn.W_o.weight[THR, j * dh] = place[j]
    # logit('1') = r - 2^k + 1/2, logit('0') = 0, every other token far below.
    model.head.weight[task.stoi["1"], THR] = 1.0
    for t, s in enumerate(task.vocab):
        if s not in "01":
            model.head.weight[t, ONE] = -1e4
    # robustness probes: round every weight, then add seeded Gaussian noise to every weight
    if DECIMALS is not None:
        for p in model.parameters():
            p.copy_(torch.round(p, decimals=DECIMALS))
    if NOISE > 0:
        g = torch.Generator().manual_seed(0)
        for p in model.parameters():
            p.add_(NOISE * torch.randn(p.shape, generator=g))
    check_all_inputs(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
M = 6.0                      # how strongly a head prefers its own region
DECIMALS = None              # round every weight to this many decimals (None = exact)
NOISE = 1e-3                 # std of seeded Gaussian noise added to every weight
model_shorthand_name = "e28_hard_attn_noise_1e-3"
model_description = "e03 hard one-position-per-head attention under the same all-weights noise as e25: does saturated attention trade parameters for robustness?"


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
