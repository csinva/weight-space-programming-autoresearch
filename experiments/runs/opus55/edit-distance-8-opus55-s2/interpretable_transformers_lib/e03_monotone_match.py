"""
Interpretable transformer for edit-distance-8: "aabbccdd,abcdabcd=" -> "4".

The Levenshtein DP is run as a tied-weight cellular automaton. Each t_j position is a DP row
holding, for every column i, the horizontal difference H(i) (D[i,j]-D[i-1,j]) and the vertical
difference V(i) (D[i,j]-D[i,j-1]), both in {-1,0,+1}, stored as a (plus, minus) bit pair.
With m = (s_i == t_j), the DP cell is
    z  = min(1-m, v+1, h+1)      h' = z - v      v' = z - h
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
BETA = 200.0     # attention sharpness for position selection
GAMMA = 100.0    # attention sharpness for character match
GATE = 10.0      # how hard non-row positions switch the MLP off
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
    """Pre-norm block with LayerNorm removed: the state is exact bits, so there is nothing to rescale."""

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
        d_model: int = 64,
        n_heads: int = 16,
        n_layers: int = N_LAYERS,
        d_ff: int = 65,
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

# Channels of the d_model state vector (column index i is 0-based).
def M(i):  return i          # match bit m(i, this row), fresh every layer
def HP(i): return 8 + i      # H(i) == +1
def HN(i): return 16 + i     # H(i) == -1
def VP(i): return 24 + i     # V(i) == +1
def VN(i): return 32 + i     # V(i) == -1
ROW = 40                     # 1 at row positions; doubles as the boundary V(-1) = +1
CH = 41                      # two channels: character on a circle (a, b, c, d at 90 degree steps)
POS = 43                     # 18 channels: one-hot position


def write_cell_mlp(mlp: MLP) -> None:
    """Per column: 4 pass-through units and 4 logic units; m accumulates and only ever inhibits."""
    a, b = mlp.fc1, mlp.fc2
    def unit(u, terms, bias=0.0, gated=True):
        for ch, w in terms:
            a.weight[u, ch] += w
        a.bias[u] += bias
        if gated:
            a.weight[u, ROW] += GATE
            a.bias[u] -= GATE
    PHP = lambda i: 8 * i
    PHN = lambda i: 8 * i + 1
    PVP = lambda i: 8 * i + 2
    PVN = lambda i: 8 * i + 3
    A, B, C, D = (lambda i: 8 * i + 4), (lambda i: 8 * i + 5), (lambda i: 8 * i + 6), (lambda i: 8 * i + 7)
    BOUNDARY = 8 * LEN       # pass-through of V(-1) = +1
    unit(BOUNDARY, [(ROW, 1.0)])

    for i in range(LEN):
        vp = VP(i - 1) if i else ROW          # incoming v, plus bit
        hp, hn, m = HP(i), HN(i), M(i)
        unit(PHP(i), [(hp, 1.0)])
        unit(PHN(i), [(hn, 1.0)])
        unit(PVP(i), [(VP(i), 1.0)])
        unit(PVN(i), [(VN(i), 1.0)])
        vn_terms = [(VN(i - 1), -1.0)] if i else []
        unit(A(i), [(m, -1.0), (hn, -1.0), (vp, -1.0)] + vn_terms, bias=1.0)    # z and v == 0
        unit(B(i), [(vp, 1.0), (m, -1.0), (hn, -1.0)])                          # z and v == +1
        unit(C(i), [(m, -1.0), (hn, -1.0), (hp, -1.0)] + vn_terms, bias=1.0)    # z and h == 0
        unit(D(i), [(hp, 1.0), (m, -1.0)] + vn_terms)                           # z and h == +1

        pass_vp = PVP(i - 1) if i else BOUNDARY
        b.weight[HP(i), A(i)] += 1.0
        b.weight[HP(i), PHP(i)] -= 1.0
        b.weight[HN(i), pass_vp] += 1.0
        b.weight[HN(i), B(i)] -= 1.0
        b.weight[HN(i), PHN(i)] -= 1.0
        if i:
            b.weight[HP(i), PVN(i - 1)] += 1.0
        b.weight[VP(i), PHN(i)] += 1.0
        b.weight[VP(i), C(i)] += 1.0
        b.weight[VP(i), PVP(i)] -= 1.0
        b.weight[VN(i), PHP(i)] += 1.0
        b.weight[VN(i), D(i)] -= 1.0
        b.weight[VN(i), PVN(i)] -= 1.0


def write_attention(attn: CausalSelfAttention) -> None:
    """Heads 0-7: match bit per column. Heads 8-11: previous row's H. Heads 12-15: minus own H."""
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

    for c in range(4):
        prev, self_ = dh * (8 + c), dh * (12 + c)
        for o, sign in ((prev, 1.0), (self_, -1.0)):
            for p in range(ANS + 1):
                target = p - 1 if (o == prev and p > LEN) else p
                Q[o, POS + p] = BETA * math.cos(angle(target))
                Q[o + 1, POS + p] = BETA * math.sin(angle(target))
                K[o, POS + p] = math.cos(angle(p))
                K[o + 1, POS + p] = math.sin(angle(p))
            for k, ch in enumerate((HP(2 * c), HN(2 * c), HP(2 * c + 1), HN(2 * c + 1))):
                Va[o + k, ch] = 1.0
                O[ch, o + k] = sign


def write_weights(model: SimpleTransformer, task) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: characters on a circle, comma holds row 0 (H = +1 everywhere).
        for tok, (cx, cy) in {"a": (1, 0), "b": (0, 1), "c": (-1, 0), "d": (0, -1)}.items():
            model.token_emb.weight[task.stoi[tok], CH] = cx
            model.token_emb.weight[task.stoi[tok], CH + 1] = cy
        for i in range(LEN):
            model.token_emb.weight[task.stoi[","], HP(i)] = 1.0
        for p in range(ANS + 1):
            model.pos_emb.weight[p, POS + p] = 1.0
        for p in ROWS:
            model.pos_emb.weight[p, ROW] = 1.0

        write_attention(model.block.attn)
        write_cell_mlp(model.block.mlp)

        # Readout at '=': S = 8 + sum H(i). logit[d] = 2dS - d^2 is -(S-d)^2 up to a term shared by all d.
        for t in range(task.vocab_size):
            model.head.weight[t, POS + ANS] = -1000.0
        for d in range(LEN + 1):
            t = task.stoi[str(d)]
            model.head.weight[t, POS + ANS] = 2.0 * d * LEN - d * d
            for i in range(LEN):
                model.head.weight[t, HP(i)] = 2.0 * d
                model.head.weight[t, HN(i)] = -2.0 * d

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
model_shorthand_name = "e03_monotone_match"
model_description = "e02 without the match-clearing units: m only ever inhibits, so letting it accumulate over sweeps is harmless. 65 ReLUs."


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
