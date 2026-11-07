"""
Interpretable transformer for character-level sequence tasks (task: nli-snli).

Prompt layout: premise (60 chars, pos 0..59) '|' (pos 60) hypothesis (pos 61..120) '=' (pos 121)
Answer: one char, 0=entailment, 1=neutral, 2=contradiction.

Usage:
    uv run interpretable_transformer.py --task nli-snli
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
# Architecture (no LayerNorm: identity, to keep the circuit readable)
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
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=144, n_heads=2, n_layers=3, d_ff=4):
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
# Residual stream layout (d_model=144, 2 heads of 72):
#   0..33  TOK one-hot token          34 isHyp (pos 61..120)   35 isPrem (pos 0..59)   36 const 1
#   37 is60 (the '|' position)        38 p (position index)    39 p^2
#   40..73 PREV one-hot of previous token       74 FOUND   75 NOVFLAG   76 REAL   77 NOVMEAN   78 REALMEAN
#
# Layer 1 (previous-token head): query at i wants key j = i-1 via score = -(j-(i-1))^2
#          (built from p and p^2).  Position 60 ('|') attends to itself, so PREV[60]='|'.
# Layer 2 (induction-style bigram lookup): query (TOK_i, PREV_i) vs key (TOK_j, PREV_j) for premise keys j.
#          Full bigram match scores 2, partial <=1, the sink key at '|' scores 1.5, non-premise keys -10.
#          Attention mass on premise (FOUND ~ 1) therefore means "this bigram occurs in the premise".
#          MLP: NOVFLAG = relu(1 - [tok=='_'] - FOUND)  (real hypothesis char whose bigram is new),  REAL = 1 - [tok=='_'].
# Layer 3 (hypothesis mean): last position averages NOVFLAG and REAL over hypothesis positions.
# Unembed: score_k = 60*(NOV[k]*NOVMEAN + LEN[k]*REALMEAN) + BIAS[k] for answers '0','1','2'.



def _design(task):
    """Design-time calibration of NOV/LEN/BIAS by coarse grid search (not part of the model)."""
    import itertools
    import numpy as np
    ex = list({e.prompt: e for e in task.generate_examples(6000, seed=1)}.values())
    rows = []
    for e in ex:
        p = e.prompt[:60]; h = e.prompt[61:121]
        pb = {(p[max(j - 1, 0)] if j > 0 else p[0], p[j]) for j in range(60)}
        nov = sum((h[i - 1] if i > 0 else "|", h[i]) not in pb for i in range(60) if h[i] != "_")
        rows.append((nov, sum(c != "_" for c in h), int(e.answer)))
    R = np.array(rows, float)
    grid = np.linspace(-1, 1, 9)
    best, bestacc = None, 0
    for wn1, wl1, wn2, wl2 in itertools.product(grid, grid, grid, grid):
        for b1 in range(-12, 13, 3):
            for b2 in range(-12, 13, 3):
                s = np.stack([np.zeros(len(R)), wn1 * R[:, 0] + wl1 * R[:, 1] / 4 + b1, wn2 * R[:, 0] + wl2 * R[:, 1] / 4 + b2], 1)
                acc = (s.argmax(1) == R[:, 2]).mean()
                if acc > bestacc:
                    bestacc, best = acc, (wn1, wl1 / 4, b1, wn2, wl2 / 4, b2)
    print("design best", bestacc, best)


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    stoi = task.stoi
    D, V = model.d_model, model.vocab_size
    DH = D // model.n_heads
    SL = 60
    PAD = stoi["_"]
    TOK, ISHYP, ISPREM, CONST, IS60, P, P2 = 0, 34, 35, 36, 37, 38, 39
    PREV, FOUND, NOVF, REAL, NOVM, REALM = 40, 74, 75, 76, 77, 78
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for i in range(V):
            model.token_emb.weight[i, TOK + i] = 1.0
        pe = model.pos_emb.weight
        for pos in range(model.max_seq_len):
            pe[pos, CONST] = 1.0
            pe[pos, P] = float(pos)
            pe[pos, P2] = float(pos * pos)
            if pos < SL:
                pe[pos, ISPREM] = 1.0
            elif pos == SL:
                pe[pos, IS60] = 1.0
            elif pos <= 2 * SL:
                pe[pos, ISHYP] = 1.0

        # ---- Layer 1: previous-token head (head 0) ----
        a = model.blocks[0].attn
        G1 = 160.0  # /sqrt(72)... (divided by 8.49) -> neighbouring-position logit gap ~ 19
        a.W_q.weight[0, P] = G1; a.W_q.weight[0, CONST] = -G1; a.W_q.weight[0, IS60] = G1
        a.W_q.weight[1, CONST] = G1
        a.W_k.weight[0, P] = 2.0; a.W_k.weight[1, P2] = -1.0
        for i in range(V):
            a.W_v.weight[i, TOK + i] = 1.0
            a.W_o.weight[PREV + i, i] = 1.0

        # ---- Layer 2: bigram-in-premise lookup (head 0) + per-position novelty flags ----
        a = model.blocks[1].attn
        G2 = 320.0
        for i in range(V):
            a.W_q.weight[i, TOK + i] = G2
            a.W_q.weight[V + i, PREV + i] = G2
            a.W_k.weight[i, TOK + i] = 1.0
            a.W_k.weight[V + i, PREV + i] = 1.0
        a.W_q.weight[2 * V, CONST] = G2
        a.W_k.weight[2 * V, CONST] = -10.0   # hypothesis / '=' keys: -10
        a.W_k.weight[2 * V, ISPREM] = 10.0   # premise keys: 0
        a.W_k.weight[2 * V, IS60] = 10.0 + SINK  # sink key score (full bigram match = 2, partial <= 1)
        a.W_v.weight[0, ISPREM] = 1.0
        a.W_o.weight[FOUND, 0] = 1.0
        m = model.blocks[1].mlp
        m.fc1.weight[0, TOK + PAD] = -1.0; m.fc1.weight[0, FOUND] = -1.0; m.fc1.bias[0] = 1.0
        m.fc2.weight[NOVF, 0] = 1.0
        m.fc1.weight[1, TOK + PAD] = -1.0; m.fc1.bias[1] = 1.0
        m.fc2.weight[REAL, 1] = 1.0

        # ---- Layer 3: average over hypothesis positions ----
        a = model.blocks[2].attn
        a.W_q.weight[0, CONST] = 160.0
        a.W_k.weight[0, ISHYP] = 1.0
        a.W_v.weight[0, NOVF] = 1.0; a.W_v.weight[1, REAL] = 1.0
        a.W_o.weight[NOVM, 0] = 1.0; a.W_o.weight[REALM, 1] = 1.0

        # ---- Unembed ----
        for k in range(3):
            t = stoi[str(k)]
            model.head.weight[t, NOVM] = NOV[k] * SL
            model.head.weight[t, REALM] = LEN[k] * SL
            model.head.weight[t, CONST] = BIAS[k] + 100.0  # +100 keeps answer tokens above all other (zero) logits


# score_k = 60*(NOV[k]*NOVMEAN + LEN[k]*REALMEAN) + BIAS[k]
model_shorthand_name = "Final_BigramNovelty_Len_v50"
SINK = 1.5
NOV, LEN, BIAS = [0, .25, .55], [0, .28, -.1875], [0, -14, 0]
model_description = f"Simplified (no unigram head): prev-token head, bigram-in-premise induction head, hyp mean; NOV={NOV} LEN={LEN} BIAS={BIAS}"


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
