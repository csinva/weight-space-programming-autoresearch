"""
Interpretable transformer for edit-distance-8: "aabbccdd,abcdabcd=" -> "4".

The Levenshtein DP runs as a tied-weight cellular automaton with ONE law at every position.
Each position stores H(i) and V(i), the horizontal and vertical DP differences in {-1,0,+1}:
    z = min(1-m, v+1, h+1)     h' = z - v     v' = z - h      (m = s_i == t_j, v = V(i-1), h = H(i))
Inert positions simply have m=1, so z=0. Comma (H=+1, V=-1) is then a fixed point that holds DP
row 0, and '=' turns the H it copies from row 8 into V=-H, which the head reads.
No training anywhere: all weights are written in closed form.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
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
# Experiment configuration
# ---------------------------------------------------------------------------

N_LAYERS = 16    # one sweep per anti-diagonal (15) plus one to copy row 8 to '='
BETA = 200.0     # match heads: score of the attended s_i position
GAMMA = 100.0    # match heads: score bonus for an equal character
SHARP = 4000.0   # prev/self heads: neighbours differ by SHARP*(1-cos 20deg)/2 nats, about 120
SELFCHECK = 20000


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
    """Pre-norm block with LayerNorm removed: the state is exact integers, nothing to rescale."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer whose single Block is applied N_LAYERS times (tied weights)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 48,
        n_heads: int = 12,
        n_layers: int = N_LAYERS,
        d_ff: int = 49,
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
        self.block = Block(d_model, n_heads, d_ff)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for _ in range(self.n_layers):
            h = self.block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

LEN = 8                      # string length
ROWS = range(9, 17)          # positions of t_1..t_8
ANS = 17                     # position of '='

# Channels of the state vector (column index i is 0-based).
def M(i): return i           # match m(i, this row): >= 1 means z = 0; only ever inhibits
def H(i): return 8 + i       # horizontal difference
def V(i): return 16 + i      # vertical difference
BND = 24                     # boundary V(-1): +1 on rows, -1 on comma, 0 elsewhere
CH = 25                      # two channels: character on a circle (a, b, c, d at 90 degree steps)
POS = 27                     # 18 channels: one-hot position


def write_cell_mlp(mlp: MLP) -> None:
    """Six ReLUs per column. z = z1 - z2 - z3 - z4; Uh, Uv hold h+1 and v+1 (fc2's bias removes the +1s)."""
    fc1, fc2 = mlp.fc1, mlp.fc2
    def unit(u, terms, bias=0.0):
        for ch, w in terms:
            fc1.weight[u, ch] += w
        fc1.bias[u] += bias

    UB = 6 * LEN             # holds BND + 1
    unit(UB, [(BND, 1.0)], 1.0)
    fc2.bias[:] = 0.0
    for i in range(LEN):
        Uh, Uv, z1, z2, z3, z4 = (6 * i + k for k in range(6))
        v, h, m = (V(i - 1) if i else BND), H(i), M(i)
        unit(Uh, [(h, 1.0)], 1.0)
        unit(Uv, [(V(i), 1.0)], 1.0)
        unit(z1, [(v, 1.0), (h, 1.0), (m, -3.0)], 1.0)
        unit(z2, [(v, 1.0), (h, 1.0), (m, -2.0)])
        unit(z3, [(h, 1.0), (v, -1.0), (m, -2.0)], -1.0)
        unit(z4, [(v, 1.0), (h, -1.0), (m, -2.0)], -1.0)
        Uv_in = 6 * (i - 1) + 1 if i else UB
        for out, minus in ((H(i), (Uh, Uv_in)), (V(i), (Uh, Uv))):     # new - old = z - h - v
            fc2.weight[out, z1] = 1.0
            for u in (z2, z3, z4):
                fc2.weight[out, u] = -1.0
            for u in minus:
                fc2.weight[out, u] -= 1.0
            fc2.bias[out] = 2.0


def write_attention(attn: CausalSelfAttention) -> None:
    """Heads 0-7: match per column. Heads 8-9: previous row's H. Heads 10-11: minus own H."""
    dh = attn.d_head
    Q, K, Va, O = attn.W_q.weight, attn.W_k.weight, attn.W_v.weight, attn.W_o.weight
    angle = lambda p: 2 * math.pi * p / 18

    for i in range(LEN):
        o = dh * i
        # score(s_i) = BETA + GAMMA*(same char); score(',') = BETA + GAMMA/2. The comma is the sink.
        for p in ROWS:
            Q[o, POS + p] = BETA
        Q[o + 1, CH] = GAMMA
        Q[o + 2, CH + 1] = GAMMA
        K[o, POS + i] = 1.0
        K[o, POS + LEN] = 1.0 + GAMMA / (2 * BETA)
        K[o + 1, CH] = 1.0
        K[o + 2, CH + 1] = 1.0
        Va[o, POS + i] = 1.0
        O[M(i), o] = 1.0

    for c in range(2):
        prev, self_ = dh * (8 + c), dh * (10 + c)
        for o, sign in ((prev, 1.0), (self_, -1.0)):
            for p in range(ANS + 1):
                target = p - 1 if (o == prev and p > LEN) else p
                Q[o, POS + p] = SHARP * math.cos(angle(target))
                Q[o + 1, POS + p] = SHARP * math.sin(angle(target))
                K[o, POS + p] = math.cos(angle(p))
                K[o + 1, POS + p] = math.sin(angle(p))
            for k in range(4):
                Va[o + k, H(4 * c + k)] = 1.0
                O[H(4 * c + k), o + k] = sign


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        tok, pos = model.token_emb.weight, model.pos_emb.weight

        for c, (cx, cy) in {"a": (1, 0), "b": (0, 1), "c": (-1, 0), "d": (0, -1)}.items():
            tok[task.stoi[c], CH] = cx
            tok[task.stoi[c], CH + 1] = cy
        comma = task.stoi[","]
        tok[comma, BND] = -1.0
        for i in range(LEN):                 # the fixed point that holds DP row 0
            tok[comma, H(i)] = 1.0
            tok[comma, V(i)] = -1.0
        for p in range(ANS + 1):
            pos[p, POS + p] = 1.0
            if p in ROWS:
                pos[p, BND] = 1.0
            else:
                for i in range(LEN):
                    pos[p, M(i)] = 1.0       # inert: "matches everything", so z = 0

        write_attention(model.block.attn)
        write_cell_mlp(model.block.mlp)

        # Readout at '=': S = 8 - sum V(i). logit[d] = 2dS - d^2 is -(S-d)^2 up to a term shared by all d.
        W = model.head.weight
        W[:, POS + ANS] = -1000.0
        for d in range(LEN + 1):
            t = task.stoi[str(d)]
            W[t, POS + ANS] = 2.0 * d * LEN - d * d
            for i in range(LEN):
                W[t, V(i)] = -2.0 * d

        selfcheck(model, task)


def selfcheck(model: SimpleTransformer, task) -> None:
    """Diagnostic only: compare the model with task.distance on far more strings than the eval uses."""
    if not SELFCHECK:
        return
    rng = random.Random(123)
    pairs = []
    for n in range(SELFCHECK):
        s = [rng.choice("abcd") for _ in range(LEN)]
        if n % 2:
            t = [rng.choice("abcd") for _ in range(LEN)]
        else:                                   # near-duplicates exercise low distances
            t = list(s)
            for _ in range(rng.randint(0, 4)):
                t[rng.randrange(LEN)] = rng.choice("abcd")
            if rng.random() < 0.5:
                k = rng.randrange(LEN)
                t = (t[:k] + t[k + 1:] + [rng.choice("abcd")])
        pairs.append(("".join(s), "".join(t)))
    ids = torch.tensor([[task.stoi[c] for c in f"{s},{t}="] for s, t in pairs])
    pred = model(ids)[:, -1].argmax(-1).tolist()
    want = [task.stoi[str(task.distance(s, t))] for s, t in pairs]
    print(f"SELFCHECK: {sum(a == b for a, b in zip(pred, want))}/{len(pairs)}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e06_sharp_uniform_law"
model_description = "e04 with the root cause fixed: prev/self heads sharpened to SHARP=4000 (about 120 nats between neighbours). The scalar state has no thresholds to absorb attention leakage, unlike bit pairs."


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
