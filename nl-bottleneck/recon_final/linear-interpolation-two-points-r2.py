"""
Interpretable transformer for linear-interpolation-two-points.

Attention-only 2-layer transformer (no MLP, no LayerNorm), 9 named residual
channels, all weights written in closed form (no training).

Usage:
    python interpretable_transformer.py --task linear-interpolation-two-points
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
    """Multi-head causal attention with per-head q/k width d_qk and value width d_v.
    No 1/sqrt(d) scaling: logits are exactly q.k as written in write_weights."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int = 2, d_v: int = 1):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v)
        return self.W_o(out)


class SimpleTransformer(nn.Module):
    """Attention-only 2-layer transformer (no MLP, no LayerNorm), 9 residual channels.
    Layer 1: 5 'read' heads; layer 2: 4 'fit' heads."""

    def __init__(self, vocab_size: int, max_seq_len: int = 16, d_model: int = 9,
                 heads_per_layer=(5, 4)):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([CausalSelfAttention(d_model, h) for h in heads_per_layer])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for layer in self.layers:
            h = h + layer(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

# residual channels
ONE, EQ, PA, PB, PC, OUT, DY, DX, DQ = range(9)


def write_weights(model: SimpleTransformer, task) -> None:
    stoi = task.stoi
    G = 20.0                     # layer-1 attention logit gap
    L10, L9, L99 = math.log(10), math.log(9), math.log(99)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings ---
        te, pe = model.token_emb.weight, model.pos_emb.weight
        for d in range(10):
            te[stoi[str(d)], OUT] = -10.0 * d        # digit value stored as -10*d in "out"
        te[stoi["="], EQ] = 1.0
        for p in range(11, model.max_seq_len):
            pe[p, ONE] = 1.0                          # answer-position constant
        # positional tags read by layer-1 heads (two heads share a channel via sign)
        pe[0, PA] = G                                 # x1
        pe[2, PA] = -(G + L10); pe[3, PA] = -G        # y1 tens:units = 10:1
        pe[5, PB] = G                                 # x2
        pe[7, PB] = -(G + L10); pe[8, PB] = -G        # y2
        pe[10, PC] = G                                # xq

        # --- layer 1: read heads (x1, y1, x2, y2, xq) ---
        # At prompt positions ONE=EQ=0 -> all heads attend uniformly -> identical
        # reads -> dy=dx=dq=0 there (needed so layer-2 keys are inert).
        a1 = model.layers[0]
        heads = [  # (pos channel, sign, '=' self bonus, value scale, writes)
            (PA, +1, G + L9, 1.0, {DX: -1, DQ: -1, OUT: -3}),
            (PA, -1, G + L99, 11.0, {DY: -1, OUT: +1}),
            (PB, +1, G + L9, 1.0, {DX: +1}),
            (PB, -1, G + L99, 11.0, {DY: +1}),
            (PC, +1, G + L9, 1.0, {DQ: +1, OUT: +3}),
        ]
        for h, (ch, s, eqb, scale, writes) in enumerate(heads):
            a1.W_q.weight[2 * h, ONE] = 1.0
            a1.W_k.weight[2 * h, ch] = s
            a1.W_q.weight[2 * h + 1, EQ] = 1.0
            a1.W_k.weight[2 * h + 1, EQ] = eqb       # '=' gets 9x the number's weight
            a1.W_v.weight[h, OUT] = -0.1 * scale      # value = digit (x11 for y heads)
            for c, w in writes.items():
                a1.W_o.weight[c, h] = w

        # --- layer 2: fit heads for slopes 1,2,4,5 (3 is the default) ---
        a2 = model.layers[1]
        P, STAY = 2000.0, 10.0
        for h, k in enumerate([1, 2, 4, 5]):
            # u = dy - k*dx ; logit = -P*u_q*u_k  (= -P*u^2 on self)
            a2.W_q.weight[2 * h, DY] = -P
            a2.W_q.weight[2 * h, DX] = P * k
            a2.W_k.weight[2 * h, DY] = 1.0
            a2.W_k.weight[2 * h, DX] = -k
            # stay bonus (one_q - 2 eq_q)(10 one_k - 20 eq_k): +10 on self, -10 on '=' from
            # the units step, 0 on prompt tokens (zero-valued fallback)
            a2.W_q.weight[2 * h + 1, ONE] = 1.0
            a2.W_q.weight[2 * h + 1, EQ] = -2.0
            a2.W_k.weight[2 * h + 1, ONE] = STAY
            a2.W_k.weight[2 * h + 1, EQ] = -2 * STAY
            a2.W_v.weight[h, DQ] = float(k - 3)
            a2.W_o.weight[OUT, h] = 1.0

        # --- readout: logit_d = 2d*out - d^2 - 0.9d*eq  (out shifted -0.45 at '=') ---
        W = model.head.weight
        for d in range(10):
            i = stoi[str(d)]
            W[i, OUT] = 2.0 * d
            W[i, ONE] = -float(d * d)
            W[i, EQ] = -0.9 * d
        for c in ",;=":
            W[stoi[c], ONE] = -100.0


model_shorthand_name = "AttnOnly2L_Read5_Fit4"
model_description = ("Attention-only 2-layer, 9 channels. L1: 5 read heads (x1,y1,x2,y2,xq; 10:1 tens:units, "
                     "'=' self-attn gives 1/10 scale at tens step) write dy,dx,dq and default y1+3dq. "
                     "L2: slope heads 1,2,4,5 score -2000(dy-k dx)^2 + stay bonus, add (k-3)dq. "
                     "Readout 2d*out-d^2 with -0.45 shift at '='; tens digit emb subtracts 10*tens.")


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
