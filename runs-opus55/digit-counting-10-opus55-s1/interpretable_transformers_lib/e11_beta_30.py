"""Hand-written transformer for digit-counting-10. No training anywhere."""

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
        self.register_ch = 2                      # the one channel attention moves

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = x.view(B, T, H, dh).transpose(1, 2)   # keys are the raw hidden state
        v = torch.zeros_like(x)
        v[..., self.register_ch] = x[..., self.register_ch]
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).contiguous().view(B, T, D)


class Block(nn.Module):
    """Attention-only block: no LayerNorm, no MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    """2-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 4,
        n_heads: int = 1,
        n_layers: int = 2,
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
        self.pos_emb = nn.Embedding(3, d_model)   # only pos 10, 11, 12 are special
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        h = self.token_emb(ids)
        if T > 10:
            pos = torch.arange(10, T, device=ids.device) - 10
            h = torch.cat([h[:, :10], h[:, 10:] + self.pos_emb(pos)[None, :, :]], dim=1)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Count c = #{i < 10 : digit[i] == digit[10]} by softmax normalisation.

    Channels: COS, SIN = digit phase code; P_Q = register; ANS = answer position.
    Layer 1, at the query (pos 10): q = k = phase code, so the query attends
    equally to itself and each match; the value is "am I the query", which
    adds x = attn[self] = 1 / (1 + c) to the register (it starts at 1).
    Layer 2, at pos 11 and 12: attend to pos 10 and copy the register P = 1 + x.
    Head: logit_k = 2 u_k P - u_k^2 with u_k = 1 + 1/(1+k), which is -(P - u_k)^2
    up to a term shared by all k, i.e. nearest-template decoding.
    """
    COS, SIN, P_Q, ANS = range(4)
    BETA = 30.0                               # attention sharpness
    scale = BETA * math.sqrt(model.d_model)   # cancels the sqrt(dh) in attention
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: digits as points on the unit circle (phase code), plus
        # markers for the 3 special positions.
        for d in range(10):
            model.token_emb.weight[d, COS] = math.cos(2 * math.pi * d / 10)
            model.token_emb.weight[d, SIN] = math.sin(2 * math.pi * d / 10)
        model.pos_emb.weight[0, P_Q] = 1.0        # pos 10: the query
        model.pos_emb.weight[1, ANS] = 1.0        # pos 11: tens digit
        model.pos_emb.weight[2, ANS] = 1.0        # pos 12: units digit

        # Layer 1: match on digit identity; values mark the query position.
        a = model.blocks[0].attn
        for i in (COS, SIN):
            a.W_q.weight[i, i] = scale
        # Pos 11/12 look at themselves instead, so they leave P_Q at zero.
        a.W_q.weight[ANS, ANS] = scale            # P_Q at pos 10 is now 1 + x

        # Layer 2: pos 11/12 look up pos 10 and copy P_Q = 1 + x onto themselves.
        a = model.blocks[1].attn
        a.W_q.weight[P_Q, ANS] = scale

        # Head. Pos 11 and 12 both have ANS = 1; only pos 12 has COS = 1 (its
        # token is '0'), so ANS-only terms set the tens-digit rule and COS
        # cancels them to leave the units-digit parabola.
        BIG = 100.0
        THETA = 1 + (1 / 10 + 1 / 11) / 2         # '1' at pos 11 iff P_Q < THETA
        model.head.weight[1, ANS] = THETA
        model.head.weight[2:, ANS] = -BIG
        for k in range(10):
            u = 1.0 + 1.0 / (1 + k)
            model.head.weight[k, P_Q] = 2 * u
            model.head.weight[k, COS] = -u * u - model.head.weight[k, ANS]
        # Previous output '1' (SIN > 0; '0' has SIN = 0) means c = 10, so the
        # units digit must be 0.
        model.head.weight[0, SIN] = BIG
    stress_test(model, task)


def stress_test(model, task, n_per_case: int = 20) -> None:
    """Verification only: greedy-decode crafted prompts for every (query, count)."""
    import random
    rng = random.Random(123)
    wrong, total = 0, 0
    with torch.no_grad():
        for q in range(10):
            for c in range(11):
                for _ in range(n_per_case):
                    others = [d for d in range(10) if d != q]
                    digits = [q] * c + [rng.choice(others) for _ in range(10 - c)]
                    rng.shuffle(digits)
                    ids = task.encode("".join(map(str, digits)) + f"{q}=")
                    for _ in range(2):
                        logits = model(torch.tensor([ids]))
                        ids.append(int(logits[0, -1].argmax()))
                    total += 1
                    if task.decode(ids[-2:]) != f"{c:02d}":
                        wrong += 1
                        if wrong <= 5:
                            print("STRESS FAIL", digits, q, c, task.decode(ids[-2:]))
    print(f"STRESS {total - wrong}/{total} correct")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e11_beta_30"
model_description = "Sharpness sweep: lower BETA from 100 to 30 to find where non-matching digits start leaking into the count."


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
