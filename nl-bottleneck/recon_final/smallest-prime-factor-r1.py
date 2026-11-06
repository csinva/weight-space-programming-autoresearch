"""
Interpretable transformer for character-level sequence tasks.

Hand-written (no training) 2-block transformer for `smallest-prime-factor`.
prompt "NNNNN=" -> answer "SSSSS".
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
# Residual stream (scalar channels, float64, no LayerNorm):
#   0 digit value d     1 constant 1      2 block-1 key (pos)   3 n
#   4,5,6 factor hundreds/tens/units      7 i   8 i^2   9 s-5 (query)
#   10 copied prompt digit   11,12,13 flags for steps 7,8,9   14 A (answer)
D_MODEL = 15


class Head(nn.Module):
    """Single causal attention head with d_head = dq (no 1/sqrt scaling)."""

    def __init__(self, d_model: int, dq: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, dq, bias=False)
        self.W_k = nn.Linear(d_model, dq, bias=False)
        self.W_v = nn.Linear(d_model, 1, bias=False)
        self.W_o = nn.Linear(1, d_model, bias=False)

    def forward(self, x):
        T = x.shape[1]
        scores = self.W_q(x) @ self.W_k(x).transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o(attn @ self.W_v(x))


def pulse(u):
    # = 1 when u is an integer; 0 once |u - round(u)| > ~1/444 (non-divisors are >= 1/313 away)
    return F.relu(1.0 + 1e4 * (torch.cos(2 * math.pi * u) - 1.0))


class SieveMLP(nn.Module):
    """pulse(n/p) per prime, then lateral inhibition relu(2*pulse_p - sum_{q<=p} pulse_q)."""

    def __init__(self, d_model: int, n_primes: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, n_primes, bias=False)
        self.inhib = nn.Linear(n_primes, n_primes, bias=False)
        self.fc2 = nn.Linear(n_primes, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.inhib(pulse(self.fc1(x)))))


class GateMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


PRIMES = [p for p in range(2, 317) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
assert len(PRIMES) == 65


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int = 16, d_model: int = D_MODEL):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn1 = Head(d_model, 1)
        self.mlp1 = SieveMLP(d_model, len(PRIMES))
        self.attn2 = Head(d_model, 2)
        self.mlp2 = GateMLP(d_model, 4)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.double()

    def forward(self, ids):
        T = ids.shape[1]
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        h = h + self.attn1(h)
        h = h + self.mlp1(h)
        h = h + self.attn2(h)
        h = h + self.mlp2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    stoi = task.stoi
    # token embedding: scalar digit value; '=' -> 0
    for d in range(10):
        model.token_emb.weight[stoi[str(d)], 0] = d
    # positional embedding
    P = model.pos_emb.weight
    for i in range(P.shape[0]):
        P[i, 1] = 1.0
        P[i, 2] = (4 - i) * math.log(10) if i < 5 else -60.0
        P[i, 7] = i
        P[i, 8] = i * i
        P[i, 9] = i - 5
        if i in (7, 8, 9):
            P[i, 11 + (i - 7)] = 1.0
    # Block 1 attention: constant query, weights 10^(4-i)/11111 -> n
    a = model.attn1
    a.W_q.weight[0, 1] = 1.0
    a.W_k.weight[0, 2] = 1.0
    a.W_v.weight[0, 0] = 1.0
    a.W_o.weight[3, 0] = 11111.0
    # Block 1 MLP: sieve with lateral inhibition, smallest firing prime writes its digits
    m = model.mlp1
    for j, p in enumerate(PRIMES):
        m.fc1.weight[j, 3] = 1.0 / p
        for k in range(j + 1):
            m.inhib.weight[j, k] = -1.0
        m.inhib.weight[j, j] += 2.0
        m.fc2.weight[4, j] = p // 100
        m.fc2.weight[5, j] = (p // 10) % 10
        m.fc2.weight[6, j] = p % 10
    # Block 2 attention: Gaussian copy, score -40(i-(s-5))^2 (s-only term dropped by softmax)
    a = model.attn2
    a.W_q.weight[0, 9] = 80.0
    a.W_q.weight[1, 1] = -40.0
    a.W_k.weight[0, 7] = 1.0
    a.W_k.weight[1, 8] = 1.0
    a.W_v.weight[0, 0] = 1.0
    a.W_o.weight[10, 0] = 1.0
    # Block 2 MLP: ReLU gate
    g = model.mlp2
    g.fc1.weight[0, 10] = 1.0          # copy ...
    g.fc1.weight[0, 4:7] = -9.0        # ... suppressed by any nonzero factor digit
    for k in range(3):                 # release factor digit k on its flagged step
        g.fc1.weight[1 + k, 4 + k] = 1.0
        g.fc1.weight[1 + k, 11 + k] = 9.0
        g.fc1.weight[1 + k, 1] = -9.0
    g.fc2.weight[14, :] = 1.0
    # Readout: logit_d = 2d*A - d^2 ; '=' blocked
    H = model.head.weight
    for d in range(10):
        H[stoi[str(d)], 14] = 2.0 * d
        H[stoi[str(d)], 1] = -float(d * d)
    H[stoi["="], 1] = -1e6


model_shorthand_name = "spf_sieve_pulse_v1"
model_description = ("2-block no-LN float64 scalar transformer: attn1 builds n via 10^(4-i)/11111 softmax; "
                     "MLP1 cos-pulse sieve over 65 primes<=313 with lateral inhibition writes smallest factor digits; "
                     "attn2 Gaussian copy head; MLP2 ReLU gate copies n if no factor else releases factor digits.")


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
