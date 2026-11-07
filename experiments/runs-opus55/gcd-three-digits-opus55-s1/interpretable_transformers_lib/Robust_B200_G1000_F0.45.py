"""
Hand-written transformer that computes gcd(a, b) for prompts like "392,567=".
Nothing is trained: write_weights() sets every parameter in closed form.

read   : sharp attention heads pull a and b out as scalars.
fold   : one tiny MLP, tied across scales and rounds, folds a onto the nearest
         multiple of c while c shrinks geometrically down to b (long division).
swap   : (a, b) -> (b, a); one fold-chain plus swap is one round of
         least-absolute-remainder Euclid, and about log2(1000) rounds suffice.
write  : emits the digits of gcd one at a time, subtracting those already emitted.
"""

from __future__ import annotations

import argparse
import csv
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
    """Scalar heads: each head has one query/key/value number (d_head = 1)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, n_heads, bias=False)
        self.W_k = nn.Linear(d_model, n_heads, bias=False)
        self.W_v = nn.Linear(d_model, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        q, k, v = (w(x).transpose(1, 2) for w in (self.W_q, self.W_k, self.W_v))  # (B, H, T)
        scores = q.unsqueeze(-1) * k.unsqueeze(-2)               # (B, H, T, T): query i, key j
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)  # (B, H, T, T)
        return self.W_o((attn @ v.unsqueeze(-1)).squeeze(-1).transpose(1, 2))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class AttnBlock(nn.Module):
    """x + attention(x). No LayerNorm: the state holds exact integers."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class MLPBlock(nn.Module):
    """x + MLP(x)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


# Named coordinates of the hidden state.
ONE, VAL = 0, 1                 # constant 1, digit value of the current token
P, P2 = 2, 3                    # position p and p^2: -(p - t)^2 is linear in (p^2, p, 1)
A, B, C, HUND, TENS, OUT = 4, 5, 6, 7, 8, 9
D_MODEL = 10
SEQ = 10                        # positions used: 8 prompt tokens + 2 emitted digits
Q = 1                           # each fold handles a <= Q c, using 2Q knots
RATIO = 2 * Q                   # c shrinks by this factor per fold (power of 2: exact)
LEVELS = next(L for L in range(1, 20) if Q * RATIO ** (L - 1) >= 999)   # folds per round
SCALE = float(RATIO ** (LEVELS - 1))                                     # c starts at SCALE * b
ROUNDS = 9                      # Euclid rounds
BETA = 200.0                    # attention sharpness
GATE = 1000.0                   # write-gate strength
FLOOR = 0.45                    # offset that turns the nearest-digit head into floor


class SimpleTransformer(nn.Module):
    """read (attention) -> [fold x LEVELS -> swap] x ROUNDS -> write (MLP)."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = D_MODEL,
                 n_heads: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ, d_model)
        self.read = AttnBlock(d_model, n_heads)
        self.fold = MLPBlock(d_model, 2 * Q + 1)     # same weights at every level and round
        self.swap = MLPBlock(d_model, 3)             # same weights at every round
        self.write = MLPBlock(d_model, 3)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.read(self.token_emb(ids) + self.pos_emb(pos)[None, :, :])
        for _ in range(ROUNDS):
            for _ in range(LEVELS):
                h = self.fold(h)
            h = self.swap(h)
        return self.head(self.write(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-written weights for Euclid's algorithm. Nothing is trained."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: token -> (ONE=1, VAL=digit); position -> (p, p^2).
        for ch, idx in task.stoi.items():
            model.token_emb.weight[idx, ONE] = 1.0
            model.token_emb.weight[idx, VAL] = float(int(ch)) if ch.isdigit() else 0.0
        for p in range(SEQ):
            model.pos_emb.weight[p, P], model.pos_emb.weight[p, P2] = float(p), float(p * p)

        def bump(weight, row, t):
            """Set row of weight so it computes -(p - t)^2 from (ONE, P, P2)."""
            weight[row, P2], weight[row, P], weight[row, ONE] = -1.0, 2.0 * t, -float(t * t)

        # READ. Query = BETA for every head; key = -(p - t)^2, peaked at the head's
        # target t; value = VAL. Softmax then copies the digit at position t.
        at = model.read.attn
        at.W_q.weight[:, ONE] = BETA
        at.W_v.weight[:, VAL] = 1.0
        places = [(0, A, 100.), (1, A, 10.), (2, A, 1.), (4, B, 100.), (5, B, 10.), (6, B, 1.),
                  (8, HUND, 1.), (9, TENS, 1.)]
        for head, (t, dim, scale) in enumerate(places):
            bump(at.W_k.weight, head, t)
            at.W_o.weight[dim, head] = scale
            if dim == B:
                at.W_o.weight[C, head] = scale * SCALE         # c starts as SCALE * b

        # FOLD: a -> distance to the nearest multiple of c; c -> c / RATIO.
        # Folding only ever adds multiples of b (c is one), so a stays congruent
        # to +-a mod b. Each fold leaves a <= c/2 = Q * (next c), so the next fold
        # is in range; the last has c = b, giving the least absolute remainder.
        #   a' = a - 2 relu(a - c/2) + 2 relu(a - 2c/2) - 2 relu(a - 3c/2) + ...
        f1, f2 = model.fold.mlp.fc1, model.fold.mlp.fc2
        f1.weight[0, C] = 1.0                           # u0 = relu(c) = c
        f2.weight[C, 0] = 1.0 / RATIO - 1.0             # c += (1/RATIO - 1) c
        for k in range(1, 2 * Q + 1):
            f1.weight[k, A], f1.weight[k, C] = 1.0, -k / 2.0            # relu(a - k c/2)
            f2.weight[A, k] = -2.0 if k % 2 else 2.0                    # slope flips

        # SWAP: (a, b, c) -> (b, a, SCALE * a), the next Euclid round.
        s1, s2 = model.swap.mlp.fc1, model.swap.mlp.fc2
        s1.weight[0, A] = s1.weight[1, B] = s1.weight[2, C] = 1.0       # relu(a), relu(b), relu(c)
        s2.weight[A, 0], s2.weight[A, 1] = -1.0, 1.0                    # a += b - a
        s2.weight[B, 0], s2.weight[B, 1] = 1.0, -1.0                    # b += a - b
        s2.weight[C, 0], s2.weight[C, 2] = SCALE, -1.0                  # c += SCALE a - c

        # WRITE: OUT = the remaining value, scaled for the digit due at this
        # position. Each unit gets -GATE (p - t)^2, which is 0 at its position t
        # and at most -GATE elsewhere, so only that unit can fire.
        w1, w2 = model.write.mlp.fc1, model.write.mlp.fc2
        for unit, t in enumerate((7, 8, 9)):
            bump(w1.weight, unit, t)
            w1.weight[unit, [ONE, P, P2]] *= GATE
        w1.weight[:, A] = w1.weight[:, B] = torch.tensor([0.01, 0.1, 1.0])  # g = a + b at the end
        w1.weight[1, HUND] = -10.0                                 # (g - 100h) / 10
        w1.weight[2, HUND], w1.weight[2, TENS] = -100.0, -10.0     # g - 100h - 10t
        w2.weight[OUT, :] = 1.0

        # HEAD: logit_d = -(OUT - FLOOR - d)^2 up to terms independent of d. Linear
        # in (OUT, ONE), and argmax is floor(OUT) with FLOOR just under 1/2.
        model.head.weight[:, ONE] = -1e4                           # never , or =
        for d in range(10):
            model.head.weight[task.stoi[str(d)], OUT] = 2.0 * d
            model.head.weight[task.stoi[str(d)], ONE] = -(2.0 * d * FLOOR + d * d)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = f"Robust_B{BETA:g}_G{GATE:g}_F{FLOOR:g}"
model_description = (f"Robustness of Compact_Q1_L11_R9 to its constants: attention sharpness BETA={BETA:g}, "
                     f"write gate GATE={GATE:g}, floor offset FLOOR={FLOOR:g}.")


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
