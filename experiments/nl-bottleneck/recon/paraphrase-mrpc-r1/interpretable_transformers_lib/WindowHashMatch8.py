"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 3-layer transformer for `paraphrase-mrpc`:
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

class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with explicit per-head q/k and v widths and no
    1/sqrt(d) scaling (the scale is folded into the hand-written weights)."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int):
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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """LayerNorms removed so the hand-written residual stream is read exactly."""

    def __init__(self, d_model: int, n_heads: int, d_qk: int, d_v: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_qk, d_v)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


# Residual-stream layout
NB = 7                      # position bits
W = 8                       # window length (= number of layer-1 heads)
C_CODE = 0                  # 2 dims: char hash code (+-1, or 0 for special tokens)
C_PBIT = 2                  # 7 dims: binary code of own position (keys of layer 1)
C_QBIT = 9                  # 8*7 dims: binary code of max(t-k,0), k=0..7 (queries of layer 1)
C_WIN = C_QBIT + W * NB     # 16 dims: 8-char window code (layer 1 output)
C_S1 = C_WIN + 2 * W        # in sentence 1 (positions 0..SENT_LEN-1)
C_BAR = C_S1 + 1            # token is '|'
C_S2 = C_BAR + 1            # position >= SENT_LEN+1 (sentence 2 and after)
C_CHK = C_S2 + 1            # one of the first 40 sentence-2 positions
C_ONE = C_CHK + 1           # constant 1
C_MATCH = C_ONE + 1         # match flag (layer 2 output)
C_COUNT = C_MATCH + 1       # number of matching windows (layer 3 output)
D_MODEL = C_COUNT + 1
N_CHECK = 40


class SimpleTransformer(nn.Module):
    """3-layer causal transformer: L1 8 window heads, L2 one lookup head, L3 one counting head."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = D_MODEL):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads=W, d_qk=NB, d_v=2, d_ff=4 * W),
            Block(d_model, n_heads=1, d_qk=2 * W + 2, d_v=1, d_ff=0),
            Block(d_model, n_heads=1, d_qk=1, d_v=1, d_ff=0),
        ])
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

def _bits(n: int) -> list[float]:
    return [1.0 if (n >> b) & 1 else -1.0 for b in range(NB)]


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    L = task.SENT_LEN
    s2_start = L + 1

    # --- token embedding: 2-bit hash from low two bits of (index+1); specials get 0
    special = set("_|=01")
    te = model.token_emb.weight
    for c, i in task.stoi.items():
        if c not in special:
            v = i + 1
            te[i, C_CODE + 0] = 1.0 if v & 1 else -1.0
            te[i, C_CODE + 1] = 1.0 if v & 2 else -1.0
    te[task.stoi["|"], C_BAR] = 1.0

    # --- position embedding
    pe = model.pos_emb.weight
    for t in range(model.max_seq_len):
        pe[t, C_PBIT:C_PBIT + NB] = torch.tensor(_bits(t))
        for k in range(W):
            pe[t, C_QBIT + k * NB:C_QBIT + (k + 1) * NB] = torch.tensor(_bits(max(t - k, 0)))
        pe[t, C_S1] = 1.0 if t < L else 0.0
        pe[t, C_S2] = 1.0 if t >= s2_start else 0.0
        pe[t, C_CHK] = 1.0 if s2_start <= t < s2_start + N_CHECK else 0.0
        pe[t, C_ONE] = 1.0

    # --- layer 1: head k fetches the char code at t-k into window slot k
    a1 = model.blocks[0].attn
    s1 = 10.0
    for k in range(W):
        for b in range(NB):
            a1.W_q.weight[k * NB + b, C_QBIT + k * NB + b] = s1
            a1.W_k.weight[k * NB + b, C_PBIT + b] = 1.0
        for j in range(2):
            a1.W_v.weight[k * 2 + j, C_CODE + j] = 1.0
            a1.W_o.weight[C_WIN + 2 * k + j, k * 2 + j] = 1.0
    # layer-1 MLP: wipe the window stored at the '|' position (clean sentinel key)
    m1 = model.blocks[0].mlp
    M = 10.0
    for i in range(2 * W):
        m1.fc1.weight[2 * i, C_WIN + i] = 1.0
        m1.fc1.weight[2 * i, C_BAR] = M
        m1.fc1.bias[2 * i] = -M
        m1.fc1.weight[2 * i + 1, C_WIN + i] = -1.0
        m1.fc1.weight[2 * i + 1, C_BAR] = M
        m1.fc1.bias[2 * i + 1] = -M
        m1.fc2.weight[C_WIN + i, 2 * i] = -1.0
        m1.fc2.weight[C_WIN + i, 2 * i + 1] = 1.0

    # --- layer 2: window lookup in sentence 1 with sentinel on '|'
    a2 = model.blocks[1].attn
    beta = 20.0
    SENT = 2 * W - 0.5      # just below a full 8-char match (=16)
    BIG = 1000.0
    for i in range(2 * W):
        a2.W_q.weight[i, C_WIN + i] = beta
        a2.W_k.weight[i, C_WIN + i] = 1.0
    a2.W_q.weight[2 * W, C_ONE] = beta
    a2.W_k.weight[2 * W, C_BAR] = SENT
    a2.W_q.weight[2 * W + 1, C_ONE] = beta
    a2.W_k.weight[2 * W + 1, C_S2] = -BIG     # never look up inside sentence 2 / later
    a2.W_v.weight[0, C_S1] = 1.0              # flag: lookup landed in sentence 1
    a2.W_o.weight[C_MATCH, 0] = 1.0

    # --- layer 3: '=' averages match flags over the first 40 sentence-2 positions -> count
    a3 = model.blocks[2].attn
    a3.W_q.weight[0, C_ONE] = 30.0
    a3.W_k.weight[0, C_CHK] = 1.0
    a3.W_v.weight[0, C_MATCH] = 1.0
    a3.W_o.weight[C_COUNT, 0] = float(N_CHECK)

    # --- unembed: logit('1') - logit('0') = 50*(count - 0.5)
    hd = model.head.weight
    hd[task.stoi["1"], C_COUNT] = 25.0
    hd[task.stoi["1"], C_ONE] = -12.5
    hd[task.stoi["0"], C_COUNT] = -25.0
    hd[task.stoi["0"], C_ONE] = 12.5


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "WindowHashMatch8"
model_description = "2-bit char hash; L1 8 heads build 8-char window codes; L2 looks up s2 windows in s1 with sentinel on '|'; L3 counts matches over first 40 s2 positions; '1' if count>=1."


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
