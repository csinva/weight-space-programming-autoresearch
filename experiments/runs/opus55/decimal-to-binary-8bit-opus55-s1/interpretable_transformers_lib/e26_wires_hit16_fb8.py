"""
Interpretable transformer for decimal-to-binary-8bit: "197=" -> "11000101" (MSB first).

Rules of the game: hand-written weights only, never trained.
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
    """Attention only: no LayerNorm and no MLP, so the arithmetic stays exactly linear."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 13,
                 n_heads: int = 13, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(min(max_seq_len, 11), d_model)
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
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

# Channels 0..10 are a one-hot of the position. Then OUT (starts at 1) and VAL (digit value).
OUT, VAL = 11, 12
EQ_POS = 3       # position of '=', a zero-valued fallback while a wire's target is in the future
HIT, FALLBACK = 16.0, 8.0   # attention scores of a wire's target and of the fallback
NOISE = 0.0      # relative Gaussian noise on the weights chosen by NOISE_TARGETS
NOISE_TARGETS = lambda m: [m.blocks[0].attn.W_q.weight, m.blocks[0].attn.W_k.weight]
# Wiring table: head h reads position t and adds 2 * place * value to OUT.
WIRES = [(0, 100), (1, 10), (2, 1)] + [(t, -(2 ** (11 - t))) for t in range(4, 11)]


def verify_all_inputs(model, task) -> None:
    """Sanity check only: greedy-decode every n in 0..255 and count exact matches."""
    ok = 0
    with torch.no_grad():
        for n in range(256):
            ids = task.encode(f"{n:03d}=")
            for _ in range(task.answer_len):
                ids = ids + [int(model(torch.tensor([ids]))[0, -1].argmax())]
            ok += task.decode(ids[task.prompt_len:]) == format(n, "08b")
    seen = [int(e.prompt[:-1]) for e in task.generate_examples(5000, 1)]
    print(f"VERIFY exhaustive {ok}/256; sampled n range {min(seen)}..{max(seen)}")


def write_weights(model: SimpleTransformer, task) -> None:
    """Integer-only trial subtraction: each head is a wire from one position to OUT.

    OUT = 1 + 2 * (100a + 10b + c - 128 b7 - 64 b6 - ...), so r = N - S is doubled and made odd.
    The head subtracts 2 * 2^k at the position testing bit k, so logit('1') = 2(r - 2^k) + 1,
    which is odd and therefore never ties. logit('0') is its negative. All weights are integers.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        attn = model.blocks[0].attn

        # Embeddings: digit value and a position one-hot with a constant OUT = 1.
        for d in range(10):
            model.token_emb.weight[task.stoi[str(d)], VAL] = float(d)
        for t in range(model.pos_emb.num_embeddings):
            model.pos_emb.weight[t, t] = 1.0
            model.pos_emb.weight[t, OUT] = 1.0

        # Each wire: a constant query hits its target position, or '=' before the target exists.
        for h, (t, place) in enumerate(WIRES):
            attn.W_q.weight[h, OUT] = 1.0
            attn.W_k.weight[h, t] = HIT
            attn.W_k.weight[h, EQ_POS] = FALLBACK
            attn.W_v.weight[h, VAL] = 1.0
            attn.W_o.weight[OUT, h] = 2.0 * place

        # Output: the threshold table lives in the head, indexed by the position one-hot.
        for p in range(3, 11):
            model.head.weight[task.stoi["1"], p] = -2.0 * 2 ** (10 - p)
        model.head.weight[task.stoi["1"], OUT] = 1.0
        model.head.weight[task.stoi["0"]] = -model.head.weight[task.stoi["1"]]

        # Robustness probe: multiplicative noise on every weight.
        if NOISE:
            g = torch.Generator().manual_seed(0)
            for p in NOISE_TARGETS(model):
                p.mul_(1 + NOISE * torch.randn(p.shape, generator=g))
    verify_all_inputs(model, task)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e26_wires_hit16_fb8"
model_description = "Score sweep on e17: HIT=16 and FALLBACK=8."


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
