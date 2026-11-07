"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
        # LayerNorm removed: the hand-built circuit needs an exactly linear residual stream.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """4-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 320,
        n_heads: int = 5,
        n_layers: int = 4,
        d_ff: int = 6000,
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
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------
# Idea: spf(x) = the smallest prime p <= 313 dividing x, or x itself if no such p (x prime).
#   x mod p == 0  <=>  S_p := sum_j c_{p,j} * d_j is a multiple of p, where d_j is digit j of x
#   (j = 0 most significant) and c_{p,j} = 10^(4-j) mod p. S_p is linear in the digits, and
#   "S_p is a multiple of p" is an exact sum of ReLU hat functions (no multiplication needed).
#
#   layer 1 attn : head j copies the one-hot of prompt digit j to every answer position
#   layer 2 MLP  : F_p = [p | x] for each prime p (hat functions of S_p at k*p)
#   layer 3 MLP  : G_p = [p is the smallest flagged prime]; COPY_{j,c} = [digit j == c and no flag]
#   layer 4 MLP  : OUT_{j,c} = [answer digit j == c], gated by the one-hot of answer position 5+j
#   head         : logit_c = sum_j OUT_{j,c}

PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
]  # all 65 primes p with p*p < 100000
N_PRIMES = 64         # how many of the primes (prefix of PRIMES) the circuit tests
ATTN_SHARPNESS = 30.0  # attention score gap; softmax leakage ~ exp(-30)

# Residual-stream layout (d_model = 320)
CONST = 0


def TOK(v: int) -> int:      # token one-hot, v in 0..10
    return 1 + v


def POS(t: int) -> int:      # position one-hot, t in 0..10
    return 12 + t


def ONEHOT(j: int, c: int) -> int:   # prompt digit j equals c (copied by layer 1)
    return 23 + 10 * j + c


def FLAG(i: int) -> int:     # prime i divides x
    return 73 + i


def GATE(i: int) -> int:     # prime i is the smallest flagged prime
    return 138 + i


def COPY(j: int, c: int) -> int:     # digit j == c and nothing flagged
    return 203 + 10 * j + c


def OUT(j: int, c: int) -> int:      # answer digit j == c
    return 253 + 10 * j + c


def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed."""
    primes = PRIMES[:N_PRIMES]
    n_p = len(primes)
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        # Embeddings: token one-hot, position one-hot, and a constant 1 used by attention queries.
        for v in range(11):
            model.token_emb.weight[v, TOK(v)] = 1.0
        for t in range(model.pos_emb.weight.shape[0]):
            model.pos_emb.weight[t, CONST] = 1.0
            if t < 11:
                model.pos_emb.weight[t, POS(t)] = 1.0

        # Layer 1 attention: head j attends to prompt position j and copies its digit one-hot.
        attn = model.blocks[0].attn
        dh = attn.d_head
        for j in range(5):
            h = j * dh
            attn.W_q.weight[h, CONST] = ATTN_SHARPNESS * math.sqrt(dh)
            attn.W_k.weight[h, POS(j)] = 1.0
            for c in range(10):
                attn.W_v.weight[h + c, TOK(c)] = 1.0
                attn.W_o.weight[ONEHOT(j, c), h + c] = 1.0

        # Layer 2 MLP: F_p = [S_p is a multiple of p] = sum_k hat(S_p - k*p),
        # hat(s) = ReLU(s+1) - 2 ReLU(s) + ReLU(s-1) equals 1 at s=0 and 0 at every other integer.
        S_IDX = [ONEHOT(j, c) for j in range(5) for c in range(10)]
        mlp2 = model.blocks[1].mlp
        n = 0
        for i, p in enumerate(primes):
            coeffs = [pow(10, 4 - j, p) for j in range(5)]          # place value of digit j, mod p
            s_val = torch.tensor([coeffs[j] * c for j in range(5) for c in range(10)], dtype=torch.float32)
            top = 9 * sum(coeffs)                                    # largest possible S_p
            for k in range(top // p + 1):
                m = k * p
                for bias, out_coef in ((-(m - 1), 1.0), (-m, -2.0), (-(m + 1), 1.0)):
                    mlp2.fc1.weight[n, S_IDX] = s_val
                    mlp2.fc1.bias[n] = bias
                    mlp2.fc2.weight[FLAG(i), n] = out_coef
                    n += 1
        assert n <= model.d_ff, f"layer-2 needs {n} neurons, d_ff={model.d_ff}"

        # Layer 3 MLP: GATE_i = ReLU(FLAG_i - sum_{q<i} FLAG_q) (smallest flagged prime);
        # COPY_{j,c} = ReLU(ONEHOT_{j,c} - sum_i FLAG_i) (digit copy when nothing is flagged).
        all_flags = [FLAG(i) for i in range(n_p)]
        mlp3 = model.blocks[2].mlp
        n = 0
        for i in range(n_p):
            mlp3.fc1.weight[n, FLAG(i)] = 1.0
            if i > 0:
                mlp3.fc1.weight[n, [FLAG(q) for q in range(i)]] = -1.0
            mlp3.fc2.weight[GATE(i), n] = 1.0
            n += 1
        for j in range(5):
            for c in range(10):
                mlp3.fc1.weight[n, ONEHOT(j, c)] = 1.0
                if all_flags:
                    mlp3.fc1.weight[n, all_flags] = -1.0
                mlp3.fc2.weight[COPY(j, c), n] = 1.0
                n += 1

        # Layer 4 MLP: OUT_{j,c} = [answer digit j == c] gated by the position one-hot of 5+j.
        # Answer digit j is c if some GATE_i has digit j of p_i equal to c, or COPY_{j,c} is on.
        mlp4 = model.blocks[3].mlp
        n = 0
        for j in range(5):
            for c in range(10):
                gates = [GATE(i) for i, p in enumerate(primes) if int(f"{p:05d}"[j]) == c]
                mlp4.fc1.weight[n, gates + [COPY(j, c), POS(5 + j)]] = 1.0
                mlp4.fc1.bias[n] = -1.0
                mlp4.fc2.weight[OUT(j, c), n] = 1.0
                n += 1

        # Output head: token c logit = number of positions whose answer digit is c.
        for j in range(5):
            for c in range(10):
                model.head.weight[c, OUT(j, c)] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = f"SPF_primes{N_PRIMES}_beta{ATTN_SHARPNESS:g}"
model_description = (
    f"Hand-built 4-layer circuit testing the first {N_PRIMES} primes (<=313) with attention sharpness "
    f"{ATTN_SHARPNESS:g}: linear digit sums, hat-function divisibility flags, smallest-flag gate, digit copy."
)


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
