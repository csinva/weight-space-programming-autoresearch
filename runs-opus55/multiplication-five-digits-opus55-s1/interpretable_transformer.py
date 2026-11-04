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
        self.W_q = nn.Linear(d_model, n_heads * d_key, bias=True)   # bias = position-free query
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
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, 2)    # token writes registers DIG, QL
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, 1, 2), Block(d_model, 2, 1)])
        self.head = nn.Linear(2, vocab_size, bias=False)  # head reads registers X, PLACE

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.pos_emb(pos)[None, :, :].repeat(ids.shape[0], 1, 1)
        h[:, :, :2] += self.token_emb(ids)
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, :, 2:4])


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

# Hidden-state channels ("registers")
DIG, QL, X, PLACE, SINK, KA, KP, KV = range(8)

LN10 = math.log(10.0)
OFF = -200.0     # log-weight of "never attend here"
SA = 1e17        # block 1 reads b*A/SA: a sink this large makes softmax act like plain exp
SV = 1e22        # block 2 head 1 reads V/SV, likewise
TIE = 0.5        # added to the remainder so that the decoder floors


def write_weights(model: SimpleTransformer, task) -> None:
    model.double()  # the registers hold integers up to 1e10
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Tokens. QL = log(b/SA) is the digit's log-weight on the multiplicand vs the sink.
        te = model.token_emb.weight
        te[:, QL] = OFF                 # '*', '=' behave as digit 0
        for d in range(10):
            te[d, DIG] = d
            te[d, QL] = math.log(d / SA) if d else OFF

        # Positions: addresses that turn softmax attention into weighted sums.
        # '*' (t=5) is a zero-valued sink with log-weight 0 in KA and log(SV) in KV.
        pe = model.pos_emb.weight
        for t in range(pe.shape[0]):
            if t >= 11:
                pe[t, QL] = OFF         # outputs stay silent in block 1, leaving X free for them
                pe[t, X] = TIE          # tie-break, so that the decoder floors
            pe[t, SINK] = -float(t == 5)  # a query q scales the sink by exp(-q) vs everything else
            pe[t, KA] = (4 - t) * LN10 if t < 5 else (0.0 if t == 5 else OFF)
            pe[t, KP] = (10 - t) * LN10 if 6 <= t <= 10 else OFF
            pe[t, KV] = (21 - t) * LN10 if t >= 12 else (math.log(SV) if t == 5 else OFF)
            if 11 <= t <= 20:           # position t predicts the digit of place 10^(20-t)
                pe[t, PLACE] = 10.0 ** (20 - t)

        # Block 1: multiplication as attention. A token with digit b attends to multiplicand
        # digit i with weight b*10^(4-i)/SA and to the sink otherwise, so it reads b*A/SA.
        at = model.blocks[0].attn
        at.W_q.weight[0, QL] = 1.0;   at.W_k.weight[0, SINK] = 1.0
        at.W_q.bias[1] = 1.0;         at.W_k.weight[1, KA] = 1.0
        at.W_v.weight[0, DIG] = 1.0
        at.W_o.weight[X, 0] = SA

        # Block 2, head 0: P = sum_j 10^(4-j) A b_j. Head 1: emitted prefix V.
        # X = P - V + 0.5 is the remainder of a long division.
        at = model.blocks[1].attn
        at.W_q.bias[0] = 1.0;         at.W_k.weight[0, KP] = 1.0
        at.W_v.weight[0, X] = 1.0
        at.W_o.weight[X, 0] = 11111.0
        at.W_q.bias[1] = 1.0;         at.W_k.weight[1, KV] = 1.0
        at.W_v.weight[1, DIG] = 1.0
        at.W_o.weight[X, 1] = -SV

        # Tropical decoder: argmax_k (k X - PLACE k(k+1)/2) = floor(X / PLACE).
        # Tokens 10 and 11 are just the same line continued: they win only if X >= 10 PLACE.
        for k in range(12):
            model.head.weight[k, 0] = k
            model.head.weight[k, 1] = -k * (k + 1) / 2

    self_test(model, task)


@torch.no_grad()
def self_test(model, task, n: int = 200000) -> None:
    """Sanity check beyond the 200-sample eval; prints only, changes nothing."""
    edge = ["00000*00000=", "99999*99999=", "99999*00001=", "00001*99999=", "10000*10000=", "99999*00000=",
            "00000*99999=", "50000*20000=", "33333*30000=", "11111*90000="]
    prompts = [e.prompt for e in task.generate_examples(n, seed=2026)] + edge
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
model_shorthand_name = "LinearSinkFinal_v12"
model_description = ("Final circuit, LinearSink_v10 with knobs removed. Two attention-only blocks: digit queries compute b*A, then "
                     "partial products minus emitted digits give the long-division remainder; a tropical linear head floors it. No MLP/ReLU/LayerNorm; needs float64.")


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
