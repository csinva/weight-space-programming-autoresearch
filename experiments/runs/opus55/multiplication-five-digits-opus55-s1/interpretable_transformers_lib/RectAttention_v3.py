"""
Interpretable transformer for 5-digit multiplication: "50494*99346=" -> 10 digits, MSB first.
The model is never trained: write_weights() sets every parameter in closed form.
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
    """Rectangular attention: d_key-dim addresses and a scalar value per head."""

    def __init__(self, d_model: int, n_heads: int, d_key: int):
        super().__init__()
        self.n_heads, self.d_key = n_heads, d_key
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_key
        q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, 1).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
        return self.W_o(out)


class Block(nn.Module):
    """Attention + residual only: no LayerNorm (registers are real numbers), no MLP."""

    def __init__(self, d_model: int, n_heads: int, d_key: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 13):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, 1, 3), Block(d_model, 2, 2)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

# Hidden-state channels ("registers")
DIG, ONE, PP, X, PLACE = range(5)                       # values
QD, QS, ISA, KA, SINK, KP, KV, NV = range(5, 13)        # attention address codes

LN10 = math.log(10.0)
OFF = -1000.0    # log-weight of "never attend here"
SA = 1e5         # block 1 attention weights sum to SA: b*11111 + sink
SV = 2e9         # block 2 head 1 weights sum to SV: emitted place values + sink


def write_weights(model: SimpleTransformer, task) -> None:
    model.double()  # the registers hold integers up to 1e10
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Tokens. QD/QS are the digit's query: log b on multiplicand, log(SA-11111b) on sink.
        te = model.token_emb.weight
        te[:, QD] = OFF
        te[:, QS] = math.log(SA)        # '*', '=' behave as digit 0
        for d in range(10):
            te[d, DIG] = d
            te[d, QD] = math.log(d) if d else OFF
            te[d, QS] = math.log(SA - 11111.0 * d)

        # Positions: addresses that turn softmax attention into exact weighted sums.
        pe = model.pos_emb.weight
        pe[:, ONE] = 1.0
        pe[:, X] = 0.5                  # tie-break, so that the decoder floors
        for t in range(pe.shape[0]):
            pe[t, ISA] = float(t < 5)
            pe[t, KA] = (4 - t) * LN10 if t < 5 else (0.0 if t == 5 else OFF)
            pe[t, SINK] = float(t == 5)  # '*' is the zero-valued sink
            pe[t, KP] = (10 - t) * LN10 if 6 <= t <= 10 else OFF
            pe[t, KV] = (21 - t) * LN10 if t >= 12 else (0.0 if t == 5 else OFF)
            if 11 <= t <= 20:           # position t predicts the digit of place 10^(20-t)
                pe[t, PLACE] = 10.0 ** (20 - t)
                pe[t, NV] = math.log(SV - sum(10.0 ** (21 - i) for i in range(12, t + 1)))

        # Block 1, head 0: multiplication as attention. A token with digit b attends to
        # the multiplicand digits with weight b*10^(4-i)/SA, so it reads b*A/SA.
        at = model.blocks[0].attn
        at.W_q.weight[0, QD] = 1.0;   at.W_k.weight[0, ISA] = 1.0
        at.W_q.weight[1, ONE] = 1.0;  at.W_k.weight[1, KA] = 1.0
        at.W_q.weight[2, QS] = 1.0;   at.W_k.weight[2, SINK] = 1.0
        at.W_v.weight[0, DIG] = 1.0
        at.W_o.weight[PP, 0] = SA

        # Block 2, head 0: P = sum_j 10^(4-j) A b_j. Head 1: emitted prefix V.
        # X = P - V + 0.5 is the remainder of a long division.
        at = model.blocks[1].attn
        at.W_q.weight[0, ONE] = 1.0;  at.W_k.weight[0, KP] = 1.0
        at.W_v.weight[0, PP] = 1.0
        at.W_o.weight[X, 0] = 11111.0
        at.W_q.weight[2, ONE] = 1.0;  at.W_k.weight[2, KV] = 1.0
        at.W_q.weight[3, NV] = 1.0;   at.W_k.weight[3, SINK] = 1.0
        at.W_v.weight[1, DIG] = 1.0
        at.W_o.weight[X, 1] = -SV

        # Tropical decoder: argmax_k (k X - PLACE k(k+1)/2) = floor(X / PLACE).
        for k in range(10):
            model.head.weight[k, X] = k
            model.head.weight[k, PLACE] = -k * (k + 1) / 2
        model.head.weight[10, ONE] = model.head.weight[11, ONE] = -1e15

    self_test(model, task)


@torch.no_grad()
def self_test(model, task, n: int = 20000) -> None:
    """Sanity check beyond the 200-sample eval; prints only, changes nothing."""
    edge = ["00000*00000=", "99999*99999=", "99999*00001=", "00001*99999=", "10000*10000=", "99999*00000=",
            "00000*99999=", "50000*20000=", "33333*30000=", "11111*90000="]
    prompts = [e.prompt for e in task.generate_examples(n, seed=12345)] + edge
    ids = torch.tensor([task.encode(p) for p in prompts])
    want = [f"{int(p[:5]) * int(p[6:11]):010d}" for p in prompts]
    for _ in range(task.answer_len):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    got = ["".join(task.decode(r[12:].tolist())) for r in ids]
    bad = [i for i in range(len(got)) if got[i] != want[i]]
    print(f"SELF_TEST {len(got) - len(bad)}/{len(got)} (edge fails: {sum(i >= n for i in bad)})",
          [(prompts[i], got[i], want[i]) for i in bad[:3]])


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "RectAttention_v3"
model_description = ("PureAttention_v2 with rectangular attention: 13 registers, per-block heads/key sizes "
                     "(1 head 3-dim keys, then 2 heads 2-dim keys), scalar values, no sqrt(d) factor.")


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
