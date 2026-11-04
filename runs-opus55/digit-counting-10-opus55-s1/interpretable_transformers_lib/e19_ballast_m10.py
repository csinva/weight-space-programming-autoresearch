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
    """Count c = #{i < 10 : digit[i] == digit[10]} on the stock architecture.

    The circuit is the same as the slim one-hot version: the query's attention
    to itself gives x = w / (w + c), layer 2 copies 1 + x to pos 11/12, and the
    head is a nearest-template decoder.

    LayerNorm is neutralised rather than removed. Every signal is written as a
    zero-sum pair (+s in one channel, -s in its partner), so each token's mean
    is exactly 0. Two ballast channels hold +M and -M, so each token's standard
    deviation is M * sqrt(2 / 64) to high precision. LayerNorm therefore just
    divides by that constant, and its gain STD multiplies it back. MLPs and the
    third attention layer are written as exact no-ops, and heads 1-3 are unused.
    """
    NEG, P, P_NEG, ANS, ANS_NEG, BAL, BAL_NEG = 10, 11, 12, 13, 14, 15, 16
    BETA, W = 10.0, 10.0                      # attention sharpness, anchor weight
    M = 10.0                                  # ballast size
    STD = M * math.sqrt(2 / model.d_model)    # LayerNorm's nearly constant divisor
    scale = BETA * math.sqrt(model.d_model // model.n_heads)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # LayerNorm gain cancels the constant divisor; biases stay 0.
        norms = [model.final_ln] + [n for b in model.blocks for n in (b.ln1, b.ln2)]
        for norm in norms:
            norm.weight.fill_(STD)

        # Embeddings: zero-sum one-hots, position pairs, and ballast everywhere.
        for d in range(10):
            model.token_emb.weight[d, d] = 1.0
            model.token_emb.weight[d, NEG] = -1.0
        model.pos_emb.weight[:, BAL] = M
        model.pos_emb.weight[:, BAL_NEG] = -M
        model.pos_emb.weight[10, P] = 1.0
        model.pos_emb.weight[10, P_NEG] = -1.0
        for pos in (11, 12):
            model.pos_emb.weight[pos, ANS] = 1.0
            model.pos_emb.weight[pos, ANS_NEG] = -1.0

        # Layer 1: head 0 matches digits, with the anchor log W on the query
        # itself; pos 11/12 look at themselves and so leave the register at 0.
        a = model.blocks[0].attn
        for i in range(10):
            a.W_q.weight[i, i] = scale
            a.W_k.weight[i, i] = 1.0
        a.W_q.weight[ANS, ANS] = scale
        a.W_k.weight[ANS, ANS] = 1.0
        a.W_q.weight[P, P] = math.log(W) * math.sqrt(model.d_model // model.n_heads)
        a.W_k.weight[P, P] = 1.0

        # Layer 2: pos 11/12 look up pos 10.
        a2 = model.blocks[1].attn
        a2.W_q.weight[P, ANS] = scale
        a2.W_k.weight[P, P] = 1.0

        # Both layers move the register, written back as a zero-sum pair.
        for attn in (a, a2):
            attn.W_v.weight[P, P] = 1.0
            attn.W_o.weight[P, P] = 1.0
            attn.W_o.weight[P_NEG, P] = -1.0

        # Head: same nearest-template decoder as the slim circuit.
        BIG = 100.0
        THETA = 1 + (W / (W + 9) + W / (W + 10)) / 2
        model.head.weight[1, ANS] = 2 * (2.0 - (1.0 + W / (W + 1))) * THETA
        model.head.weight[2:, ANS] = -BIG
        for k in range(10):
            u = 1.0 + W / (W + k)
            model.head.weight[k, P] = 2 * u
            model.head.weight[k, 0] = -u * u - model.head.weight[k, ANS]
        model.head.weight[0, 1] = BIG
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
model_shorthand_name = "e19_ballast_m10"
model_description = "Ballast sweep on the stock architecture: M=10 instead of 1000, so LayerNorm's divisor now varies slightly with the signal itself."


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
