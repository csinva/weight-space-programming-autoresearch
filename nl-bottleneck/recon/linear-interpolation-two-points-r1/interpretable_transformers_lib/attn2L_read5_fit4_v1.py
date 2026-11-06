"""
Interpretable transformer for `linear-interpolation-two-points`.

Hand-written, attention-only 2-layer transformer (no MLP, no LayerNorm) with
9 named residual channels. No training: every weight is set in closed form.

Residual channels:
  VAL  digit value of the token (punct = 0)
  OUT  answer accumulator (read by the unembedding)
  DY   y2 - y1        DX  x2 - x1        DQ  xq - x1
  EQ   +1 on '=' token, -1 on ',' token
  ANS  1 at answer positions (11..), 0 in the prompt   (positional)
  PC, PS  cos/sin of 2*pi*p/16                          (positional)
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

VAL, OUT, DY, DX, DQ, EQ, ANS, PC, PS = range(9)
D_MODEL = 9


class MultiHeadAttn(nn.Module):
    """Causal attention with n_heads heads, QK dim d_qk, scalar value/out per head.

    No 1/sqrt(d) scaling: the bilinear scores are written directly.
    """

    def __init__(self, d_model: int, n_heads: int, d_qk: int):
        super().__init__()
        self.n_heads, self.d_qk = n_heads, d_qk
        self.W_q = nn.Parameter(torch.zeros(n_heads, d_model, d_qk))
        self.W_k = nn.Parameter(torch.zeros(n_heads, d_model, d_qk))
        self.W_v = nn.Parameter(torch.zeros(n_heads, d_model))      # scalar value per head
        self.W_o = nn.Parameter(torch.zeros(n_heads, d_model))      # written direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = torch.einsum("btd,hdk->bhtk", x, self.W_q)
        k = torch.einsum("btd,hdk->bhtk", x, self.W_k)
        v = torch.einsum("btd,hd->bht", x, self.W_v)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        z = torch.einsum("bhts,bhs->bht", attn, v)
        return torch.einsum("bht,hd->btd", z, self.W_o)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.read = MultiHeadAttn(D_MODEL, n_heads=5, d_qk=3)   # layer 1
        self.fit = MultiHeadAttn(D_MODEL, n_heads=4, d_qk=3)    # layer 2
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        h = h + self.read(h)
        h = h + self.fit(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    s = task.stoi
    TH = 2 * math.pi / 16
    GAP = 20.0          # logit gap to nearest competitor in layer 1
    ANS_START = 11      # position of '=' (first answer-predicting position)

    # ---- embeddings ----
    E = model.token_emb.weight
    for d in range(10):
        E[s[str(d)], VAL] = d
        E[s[str(d)], OUT] = -10.0 * d     # subtract printed tens digit at units step
    E[s["="], EQ] = 1.0
    E[s["="], OUT] = -0.45                # floor-rounding shift at the tens step
    E[s[","], EQ] = -1.0                  # comma marker (fallback target of layer 2)

    P = model.pos_emb.weight
    for p in range(model.max_seq_len):
        P[p, PC] = math.cos(TH * p)
        P[p, PS] = math.sin(TH * p)
        P[p, ANS] = 1.0 if p >= ANS_START else 0.0

    def kvec(p):
        return torch.tensor([math.cos(TH * p), math.sin(TH * p)])

    visible = range(0, 13)   # positions an answer position can see

    # ---- layer 1: read heads ----
    # (target positions, weights, value->channel writes)
    heads = [
        ([0],    {DX: -1.0, DQ: -1.0, OUT: -3.0}),   # x1
        ([2, 3], {DY: -11.0, OUT: 11.0}),            # y1
        ([5],    {DX: 1.0}),                         # x2
        ([7, 8], {DY: 11.0}),                        # y2
        ([10],   {DQ: 1.0, OUT: 3.0}),               # xq
    ]
    L1 = model.read
    for h, (tgt, writes) in enumerate(heads):
        if len(tgt) == 1:
            u = kvec(tgt[0])
            # scale so nearest competitor is GAP below
            A = GAP / (1 - math.cos(TH))
            q = A * u
            eq_ratio = 9.0                         # '=' weight = 9 x number's weight
        else:
            t0, t1 = tgt
            kt, ku = kvec(t0), kvec(t1)
            delta = kt - ku
            Bc = math.log(10.0) / float(delta @ delta)   # tens:units = 10:1
            base = Bc * delta
            mid = kt + ku
            # A such that the best competitor is GAP below the units score
            others = [p for p in visible if p not in tgt]
            # score(p) = A*mid.k_p + base.k_p ; need su - max_other = GAP
            # find A in closed form for each competitor, take the max needed
            need = []
            for p in others:
                kp = kvec(p)
                dm = float(mid @ (ku - kp))
                db = float(base @ (ku - kp))
                need.append((GAP - db) / dm)
            A = max(need)
            q = A * mid + base
            eq_ratio = 9.9                         # '=' : tens = 99 : 10
        s_t = float(q @ kvec(tgt[0]))
        s_eq_pos = float(q @ kvec(ANS_START))
        bonus = s_t + math.log(eq_ratio) - s_eq_pos
        L1.W_q[h, ANS, 0] = q[0]
        L1.W_q[h, ANS, 1] = q[1]
        L1.W_q[h, EQ, 2] = bonus
        L1.W_k[h, PC, 0] = 1.0
        L1.W_k[h, PS, 1] = 1.0
        L1.W_k[h, EQ, 2] = 1.0
        L1.W_v[h, VAL] = 1.0
        for ch, c in writes.items():
            L1.W_o[h, ch] = c

    # ---- layer 2: slope-fit heads (k = 1, 2, 4, 5; slope 3 is the default) ----
    S, a, b = 30.0, -20.0, 10.0
    L2 = model.fit
    for h, kslope in enumerate([1, 2, 4, 5]):
        # e = dy - k*dx ; score_self = -2000 e^2 + stay bonus
        L2.W_q[h, DY, 0] = -2000.0
        L2.W_q[h, DX, 0] = 2000.0 * kslope
        L2.W_k[h, DY, 0] = 1.0
        L2.W_k[h, DX, 0] = -1.0 * kslope
        L2.W_q[h, ANS, 1] = 1.0
        L2.W_k[h, ANS, 1] = S
        L2.W_k[h, EQ, 1] = a          # comma (+20) favoured, '=' penalised
        L2.W_q[h, EQ, 2] = 1.0
        L2.W_k[h, EQ, 2] = b
        L2.W_v[h, DQ] = 1.0
        L2.W_o[h, OUT] = kslope - 3.0

    # ---- readout: logit_d = 2 d out - d^2 ----
    W = model.head.weight
    for d in range(10):
        W[s[str(d)], OUT] = 2.0 * d
        W[s[str(d)], ANS] = -float(d * d)
    for c in [",", ";", "="]:
        W[s[c], ANS] = -1000.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "attn2L_read5_fit4_v1"
model_description = ("Attention-only 2-layer, 9 channels. L1: 5 positional read heads (x1,y1,x2,y2,xq; y tens:units 10:1, "
                     "'=' 9x weight -> 1/10 scale) write dy,dx,dq and out=y1+3dq. L2: slope heads 1,2,4,5 score -2000(dy-k dx)^2 + stay bonus, "
                     "fall back to comma. Readout 2d*out-d^2 with -0.45 floor shift and -10*tens subtraction.")


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
