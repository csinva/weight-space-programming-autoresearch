"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) model for `paraphrase-mrpc`:
predict "1" iff some 8-character window near the start of sentence 2 also
appears (under a coarse 2-bit character hash) in sentence 1.

Usage:
    uv run interpretable_transformer.py --task paraphrase-mrpc
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
# Attention-only, no LayerNorm, no MLP, no 1/sqrt(d) scaling (scales are
# folded into the hand-written weights). Each layer has its own head count
# and head width.

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        hd = n_heads * d_head
        self.W_q = nn.Linear(d_model, hd, bias=False)
        self.W_k = nn.Linear(d_model, hd, bias=False)
        self.W_v = nn.Linear(d_model, hd, bias=False)
        self.W_o = nn.Linear(hd, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class SimpleTransformer(nn.Module):
    LAYERS = [(8, 7), (1, 17), (1, 1)]   # (n_heads, d_head) per layer

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 88):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([CausalSelfAttention(d_model, h, dh) for h, dh in self.LAYERS])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Hand-written weights
# ---------------------------------------------------------------------------
# Residual layout
CONST = 0            # constant 1
CODE = 1             # 2 dims: char hash code (+-1), zero for _ | = 0 1
PBITS = 3            # 7 dims: binary code of own position (+-1)
QBITS = 10           # 8 x 7 dims: binary code of the position head k should fetch
WIN = 66             # 8 x 2 dims: window code (chars t-0 .. t-7)
IS_S1 = 82           # 1 on sentence-1 positions
BLOCK = 83           # 1 on positions that may not be lookup targets
SENT = 84            # 1 on '|' token (sentinel key)
FLAG = 85            # match flag written by layer 2
INRNG = 86           # 1 on the first 40 sentence-2 positions
AVG = 87             # mean of flags (layer 3)
NBITS, WINLEN, NCHECK = 7, 8, 40


def _bits(p):
    return [1.0 if (p >> i) & 1 else -1.0 for i in range(NBITS)]


def write_weights(model: SimpleTransformer, task) -> None:
    torch.manual_seed(0)
    T = model.max_seq_len
    L = task.SENT_LEN                      # 60
    BAR = L                                 # position of '|'
    S2 = L + 1                              # first sentence-2 position
    zero_code = set("_|=01")

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # ---- token embedding: 2-bit +-1 hash code from low bits of (index+1)
        E = model.token_emb.weight
        for ch, i in task.stoi.items():
            if ch not in zero_code:
                v = (i + 1) & 3
                E[i, CODE + 0] = 1.0 if v & 1 else -1.0
                E[i, CODE + 1] = 1.0 if v & 2 else -1.0
        E[task.stoi["|"], SENT] = 1.0

        # ---- positional embedding
        P = model.pos_emb.weight
        for p in range(T):
            P[p, CONST] = 1.0
            P[p, PBITS:PBITS + NBITS] = torch.tensor(_bits(p))
            for k in range(WINLEN):
                tgt = p - k
                if p == BAR:
                    tgt = p                 # '|' fetches only itself -> all-zero window
                if tgt >= 0:
                    P[p, QBITS + NBITS * k: QBITS + NBITS * (k + 1)] = torch.tensor(_bits(tgt))
            if p < L:
                P[p, IS_S1] = 1.0
            if p < WINLEN - 1 or p > BAR:   # incomplete s1 windows and everything after '|'
                P[p, BLOCK] = 1.0
            if S2 <= p < S2 + NCHECK:
                P[p, INRNG] = 1.0

        # ---- layer 1: 8 heads, head k copies the code of the char at t-k
        a = model.blocks[0]
        B1 = 10.0
        for k in range(WINLEN):
            for b in range(NBITS):
                a.W_q.weight[k * 7 + b, QBITS + NBITS * k + b] = B1
                a.W_k.weight[k * 7 + b, PBITS + b] = 1.0
            for c in range(2):
                a.W_v.weight[k * 7 + c, CODE + c] = 1.0
                a.W_o.weight[WIN + 2 * k + c, k * 7 + c] = 1.0

        # ---- layer 2: one head, each window looks for an identical code in s1
        a = model.blocks[1]
        B2 = 40.0
        for i in range(2 * WINLEN):
            a.W_q.weight[i, WIN + i] = B2
            a.W_k.weight[i, WIN + i] = 1.0
        a.W_q.weight[16, CONST] = B2
        a.W_k.weight[16, SENT] = 2 * WINLEN - 0.5     # sentinel: just below a full match (16)
        a.W_k.weight[16, BLOCK] = -100.0
        a.W_v.weight[0, IS_S1] = 1.0                   # flag = lookup landed in s1
        a.W_o.weight[FLAG, 0] = 1.0

        # ---- layer 3: '=' averages flags over first 40 sentence-2 positions
        a = model.blocks[2]
        a.W_q.weight[0, CONST] = 1.0
        a.W_k.weight[0, INRNG] = 40.0
        a.W_v.weight[0, FLAG] = 1.0
        a.W_o.weight[AVG, 0] = 1.0

        # ---- unembedding: logit('1') - logit('0') = 50 * (count - 0.5)
        H = model.head.weight
        i1, i0 = task.stoi["1"], task.stoi["0"]
        H[i1, AVG] = 25.0 * NCHECK
        H[i1, CONST] = -12.5
        H[i0, AVG] = -25.0 * NCHECK
        H[i0, CONST] = 12.5


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "hash8_window_match_v1"
model_description = ("2-bit char hash; L1 8 binary-position heads build 8-char window codes; L2 one head looks up "
                     "each s2 window in s1 with '|' sentinel at 15.5/16; L3 '=' averages flags over first 40 s2 "
                     "positions; output 1 iff >=1 match (logit 50*(count-0.5)).")


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
