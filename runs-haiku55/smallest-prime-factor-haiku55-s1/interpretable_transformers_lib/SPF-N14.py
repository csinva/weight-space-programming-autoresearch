"""
Interpretable transformer for the smallest-prime-factor task.

Task (`smallest-prime-factor`):
  prompt "NNNNN=" (5 digits, zero-padded, then '=') -> answer "SSSSS" (smallest prime factor,
  zero-padded; a prime is its own smallest factor). Greedy decoding of 5 tokens.
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
    """Residual block without LayerNorm, so residual-stream sums stay exact (linear)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """4-block causal transformer (no LayerNorm), vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 16,
        d_model: int = 320,
        n_heads: int = 1,
        n_layers: int = 4,
        d_ff: int = 1000,
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
          83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
          179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
          271, 277, 281, 283, 293, 307, 311, 313]

TOK0 = 0
CONST = 11
POS0 = 12
R0 = 28
S0 = R0 + 65
F0 = S0 + 65
G0 = F0 + 65
NPF = G0 + 65
COPY0 = NPF + 1
L0 = COPY0 + 10
A = 40.0


def write_weights(model: SimpleTransformer, task) -> None:
    ps = PRIMES[:N_PRIMES]
    K = len(ps)
    D = model.d_model
    scale = A * math.sqrt(D)
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()

        for t in range(model.token_emb.num_embeddings):
            model.token_emb.weight[t, CONST] = 1.0
            if t <= 10:
                model.token_emb.weight[t, TOK0 + t] = 1.0
        for i in range(model.pos_emb.num_embeddings):
            model.pos_emb.weight[i, POS0 + i] = 1.0

        mlp1 = model.blocks[0].mlp
        for i in range(5):
            for d in range(10):
                u = 10 * i + d
                mlp1.fc1.weight[u, TOK0 + d] = 1.0
                mlp1.fc1.weight[u, POS0 + i] = 1.0
                mlp1.fc1.bias[u] = -1.0
                for k, p in enumerate(ps):
                    mlp1.fc2.weight[R0 + k, u] = float((pow(10, 4 - i, p) * d) % p)

        att2 = model.blocks[1].attn
        att2.W_q.weight[0, CONST] = 1.0
        for i in range(5):
            att2.W_k.weight[0, POS0 + i] = scale
        for k in range(K):
            att2.W_v.weight[R0 + k, R0 + k] = 5.0
            att2.W_o.weight[S0 + k, R0 + k] = 1.0

        mlp2 = model.blocks[1].mlp
        terms = [(+1.0, +1.0), (0.0, -2.0), (-1.0, +1.0)]
        for k, p in enumerate(ps):
            for t in range(5):
                for ci, (shift, coef) in enumerate(terms):
                    u = (5 * k + t) * 3 + ci
                    mlp2.fc1.weight[u, S0 + k] = 1.0
                    mlp2.fc1.bias[u] = -float(t * p) + shift
                    mlp2.fc2.weight[F0 + k, u] = coef

        mlp3 = model.blocks[2].mlp
        for k in range(K):
            mlp3.fc1.weight[k, F0 + k] = 1.0
            for q in range(k):
                mlp3.fc1.weight[k, F0 + q] = -1.0
            mlp3.fc2.weight[G0 + k, k] = 1.0
        u = K
        for k in range(K):
            mlp3.fc1.weight[u, F0 + k] = -1.0
        mlp3.fc1.bias[u] = 1.0
        mlp3.fc2.weight[NPF, u] = 1.0

        att4 = model.blocks[3].attn
        for j in range(5):
            att4.W_q.weight[1 + j, POS0 + 5 + j] = 1.0
            att4.W_k.weight[1 + j, POS0 + j] = scale
        for d in range(10):
            att4.W_v.weight[COPY0 + d, TOK0 + d] = 1.0
            att4.W_o.weight[COPY0 + d, COPY0 + d] = 1.0

        mlp4 = model.blocks[3].mlp
        for k, p in enumerate(ps):
            for j in range(5):
                u = 5 * k + j
                digit = (p // 10 ** (4 - j)) % 10
                mlp4.fc1.weight[u, G0 + k] = 1.0
                mlp4.fc1.weight[u, POS0 + 5 + j] = 1.0
                mlp4.fc1.bias[u] = -1.0
                mlp4.fc2.weight[L0 + digit, u] = 1.0
        base = 5 * K
        for v in range(10):
            u = base + v
            mlp4.fc1.weight[u, NPF] = 1.0
            mlp4.fc1.weight[u, COPY0 + v] = 1.0
            mlp4.fc1.bias[u] = -1.0
            mlp4.fc2.weight[L0 + v, u] = 1.0

        for v in range(11):
            model.head.weight[v, L0 + v] = 1.0


N_PRIMES = 14  # number of moduli used: a prefix of PRIMES (65 = full circuit)
model_shorthand_name = "SPF-N14"
model_description = "Sweep point N=14 primes (divisibility by 2 through 43); fallback copies the input."


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
