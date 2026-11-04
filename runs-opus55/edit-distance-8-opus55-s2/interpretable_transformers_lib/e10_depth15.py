"""
Interpretable transformer for edit-distance-8: "aabbccdd,abcdabcd=" -> "4".

The Levenshtein DP runs as a tied-weight cellular automaton. One Block, applied N_LAYERS times:
  * attention is pure routing:  B(i) := -left.(A(i) + PUNCT)    A(i) := -B(i-1)    M(i) += (s_i == t_j)
  * the MLP is one pure gate:   z = min(1-m, -A+1, -B+1), added to both A and B
Before the MLP, (A, B) hold (-v, -h). After it they hold (z-v, z-h), which are h_out and v_out.
Every non-row position has m=1, hence z=0, and stays at A=B=0. The PUNCT flag at comma supplies DP row 0
(H=+1 everywhere). '=' ends up with B = -H(row 8), and the distance is 8 - sum B.
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

N_LAYERS = 15    # one sweep per anti-diagonal (15) plus one to copy row 8 to '='
GAMMA = 100.0    # match heads: score bonus for an equal character
LOCK = 4000.0    # match heads: score for looking at column i's position
SHARP = 4000.0   # left/own heads: neighbours differ by SHARP*(1-cos 20deg)/2 nats, about 120
SELFCHECK = 20000


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head attention whose head dimension is independent of d_model."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

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
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm block with LayerNorm removed: the state is exact integers, nothing to rescale."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
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
        d_model: int = 30,
        n_heads: int = 14,
        d_head: int = 4,
        n_layers: int = N_LAYERS,
        d_ff: int = 32,
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
        self.block = Block(d_model, n_heads, d_head, d_ff)
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
STEP = 2 * math.pi / 18      # angle between neighbouring positions on the position circle

# Channels of the state vector (column index i is 0-based).
def M(i): return i           # match m(i, this row): >= 1 means z = 0; only ever inhibits
def A(i): return 8 + i       # before the MLP: -v. After it: h_out = z - v
def B(i): return 16 + i      # before the MLP: -h. After it: v_out = z - h
BND = 24                     # 1 on rows: the left boundary V(-1) = +1
PUNCT = 25                   # 1 on ',' (supplies DP row 0) and '=' (the readout constant)
CH = 26                      # two channels: character on a circle (a, b, c, d at 90 degree steps)
PC = 28                      # two channels: position on a circle


def write_cell_mlp(mlp: MLP) -> None:
    """z = z1 - z2 - z3 - z4 is min(1-m, v+1, h+1) on {-1,0,1}^2. Four ReLUs per column."""
    fc1, fc2 = mlp.fc1, mlp.fc2
    units = [   # (weight on A, on B, on M, bias, sign in z), with v = -A and h = -B
        (-1.0, -1.0, -3.0, 1.0, 1.0),     # v + h + 1
        (-1.0, -1.0, -2.0, 0.0, -1.0),    # v + h
        (1.0, -1.0, -2.0, -1.0, -1.0),    # h - v - 1
        (-1.0, 1.0, -2.0, -1.0, -1.0),    # v - h - 1
    ]
    for i in range(LEN):
        for k, (wa, wb, wm, bias, sign) in enumerate(units):
            u = 4 * i + k
            fc1.weight[u, A(i)], fc1.weight[u, B(i)], fc1.weight[u, M(i)], fc1.bias[u] = wa, wb, wm, bias
            fc2.weight[A(i), u] = fc2.weight[B(i), u] = sign


def write_attention(attn: CausalSelfAttention) -> None:
    """Heads 0-7: match per column. Heads 8-9: left A + PUNCT. Heads 10-11: own A + B(i-1). Heads 12-13: own B."""
    dh = attn.d_head
    Q, K, Va, O = attn.W_q.weight, attn.W_k.weight, attn.W_v.weight, attn.W_o.weight

    for i in range(LEN):
        o = dh * i
        # score(s_i) = LOCK + GAMMA*(same char). score(',') = LOCK + GAMMA/2, so the comma is the sink.
        ci, si = math.cos(STEP * i), math.sin(STEP * i)
        Q[o, BND] = 1.0
        K[o, PC], K[o, PC + 1] = LOCK * ci, LOCK * si
        K[o, PUNCT] = LOCK * (1 - math.cos(STEP * (LEN - i))) + GAMMA / 2
        Q[o + 1, CH], Q[o + 2, CH + 1] = GAMMA, GAMMA
        K[o + 1, CH], K[o + 2, CH + 1] = 1.0, 1.0
        Va[o, PC], Va[o, PC + 1] = ci, si       # value: 1 at s_i ...
        Va[o, PUNCT] = -math.cos(STEP * (LEN - i))   # ... and exactly 0 at the comma sink
        O[M(i), o] = 1.0

    cd, sd = math.cos(STEP), math.sin(STEP)
    for head in range(8, 14):
        o = dh * head
        K[o, PC], K[o + 1, PC + 1] = 1.0, 1.0
        if head < 10:      # the left neighbour: rotate the own position code back by one step
            Q[o, PC], Q[o, PC + 1] = SHARP * cd, SHARP * sd
            Q[o + 1, PC], Q[o + 1, PC + 1] = -SHARP * sd, SHARP * cd
        else:              # self
            Q[o, PC], Q[o + 1, PC + 1] = SHARP, SHARP
    for i in range(LEN):
        c, k = i // 4, i % 4
        left, own_a, own_b = dh * (8 + c) + k, dh * (10 + c) + k, dh * (12 + c) + k
        Va[left, A(i)], Va[left, PUNCT] = 1.0, 1.0
        O[B(i), left] = -1.0
        Va[own_b, B(i)] = 1.0
        O[B(i), own_b] = -1.0
        Va[own_a, A(i)] = 1.0
        Va[own_a, B(i - 1) if i else BND] = 1.0
        O[A(i), own_a] = -1.0


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        tok, pos = model.token_emb.weight, model.pos_emb.weight

        for c, (cx, cy) in {"a": (1, 0), "b": (0, 1), "c": (-1, 0), "d": (0, -1)}.items():
            tok[task.stoi[c], CH], tok[task.stoi[c], CH + 1] = cx, cy
        tok[task.stoi[","], PUNCT] = 1.0
        tok[task.stoi["="], PUNCT] = 1.0
        for p in range(ANS + 1):
            pos[p, PC], pos[p, PC + 1] = math.cos(STEP * p), math.sin(STEP * p)
            if p in ROWS:
                pos[p, BND] = 1.0
            else:
                for i in range(LEN):
                    pos[p, M(i)] = N_LAYERS + 1.0    # inert: m >= 1 for all N_LAYERS sweeps, so z = 0

        write_attention(model.block.attn)
        write_cell_mlp(model.block.mlp)

        # Readout at '=': S = 8 - sum B(i). logit[d] = 2dS - d^2 is -(S-d)^2 up to a term shared by all d.
        W = model.head.weight
        W[:, PUNCT] = -1000.0
        for d in range(LEN + 1):
            t = task.stoi[str(d)]
            W[t, PUNCT] = 2.0 * d * LEN - d * d
            for i in range(LEN):
                W[t, B(i)] = -2.0 * d

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
model_shorthand_name = "e10_depth15"
model_description = "Depth sweep on e09: 15 sweeps, one fewer than the 16 the DP's longest dependency chain needs."


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
