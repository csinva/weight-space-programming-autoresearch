"""
Interpretable transformer for smallest-prime-factor (hand-written weights, no training).
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

torch.set_default_dtype(torch.float64)

# Residual stream layout (all scalar features, no one-hot)
DIG, ONE, KEY1, POS, POS2, T, F7, F8, F9, N, H, TE, U, CP, A = range(15)
D_MODEL = 15
PRIMES = [p for p in range(2, 317) if all(p % q for q in range(2, int(p ** 0.5) + 1))]  # 65 primes


class Head(nn.Module):
    """Single causal attention head with unscaled dot-product scores."""

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, 1, bias=False)
        self.W_o = nn.Linear(1, d_model, bias=False)

    def forward(self, x):
        Tn = x.shape[1]
        scores = self.W_q(x) @ self.W_k(x).transpose(-2, -1)
        mask = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o(attn @ self.W_v(x))


class SieveMLP(nn.Module):
    """pulse(n/p) per prime -> lateral inhibition (ReLU) -> write factor digits."""

    def __init__(self, d_model: int, n_primes: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, n_primes)
        self.inhib = nn.Linear(n_primes, n_primes, bias=False)
        self.fc2 = nn.Linear(n_primes, d_model, bias=False)

    def forward(self, x):
        u = self.fc1(x)
        pulse = F.relu(1.0 + 1e4 * (torch.cos(2 * math.pi * u) - 1.0))
        win = F.relu(self.inhib(pulse))
        return self.fc2(win)


class GateMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    """Two blocks, no LayerNorm, float64."""

    def __init__(self, vocab_size: int, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.attn1 = Head(D_MODEL, 1)
        self.mlp1 = SieveMLP(D_MODEL, len(PRIMES))
        self.attn2 = Head(D_MODEL, 2)
        self.mlp2 = GateMLP(D_MODEL, 4)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids):
        Tn = ids.shape[1]
        h = self.token_emb(ids) + self.pos_emb(torch.arange(Tn, device=ids.device))[None]
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
    # Embeddings: token -> digit value; position -> constant features
    for d in range(10):
        model.token_emb.weight[stoi[str(d)], DIG] = d
    L = model.pos_emb.weight.shape[0]
    for i in range(L):
        pe = model.pos_emb.weight[i]
        pe[ONE] = 1.0
        pe[KEY1] = (4 - i) * math.log(10) if i < 5 else -60.0
        pe[POS] = i
        pe[POS2] = i * i
        pe[T] = i - 5
    model.pos_emb.weight[7, F7] = 1.0   # last three output steps
    model.pos_emb.weight[8, F8] = 1.0
    model.pos_emb.weight[9, F9] = 1.0

    # Block 1 attention: constant query, key = (4-i) ln10 -> weights 10^(4-i)/11111; x11111 -> n
    a = model.attn1
    a.W_q.weight[0, ONE] = 1.0
    a.W_k.weight[0, KEY1] = 1.0
    a.W_v.weight[0, DIG] = 11111.0
    a.W_o.weight[N, 0] = 1.0

    # Block 1 MLP: sieve over 65 primes, lateral inhibition 2*pulse_p - sum_{q<=p} pulse_q
    m = model.mlp1
    for j, p in enumerate(PRIMES):
        m.fc1.weight[j, N] = 1.0 / p
        m.inhib.weight[j, :j + 1] = -1.0
        m.inhib.weight[j, j] = 1.0
        m.fc2.weight[H, j] = p // 100
        m.fc2.weight[TE, j] = (p // 10) % 10
        m.fc2.weight[U, j] = p % 10

    # Block 2 attention: Gaussian copy, score -40(i-(s-5))^2 (query-only term dropped)
    a = model.attn2
    a.W_q.weight[0, T] = 80.0
    a.W_k.weight[0, POS] = 1.0
    a.W_q.weight[1, ONE] = -40.0
    a.W_k.weight[1, POS2] = 1.0
    a.W_v.weight[0, DIG] = 1.0
    a.W_o.weight[CP, 0] = 1.0

    # Block 2 MLP: ReLU gate
    g = model.mlp2
    g.fc1.weight[0, CP] = 1.0
    g.fc1.weight[0, H] = -9.0
    g.fc1.weight[0, TE] = -9.0
    g.fc1.weight[0, U] = -9.0
    for k, (src_f, flag) in enumerate([(H, F7), (TE, F8), (U, F9)], start=1):
        g.fc1.weight[k, src_f] = 1.0
        g.fc1.weight[k, flag] = 9.0
        g.fc1.bias[k] = -9.0
    g.fc2.weight[A, :] = 1.0

    # Readout: logit_d = 2d*A - d^2 (nearest digit); '=' blocked
    for d in range(10):
        model.head.weight[stoi[str(d)], A] = 2.0 * d
        model.head.weight[stoi[str(d)], ONE] = -float(d * d)
    model.head.weight[stoi["="], ONE] = -1e9


model_shorthand_name = "spf_sieve_v1"
model_description = ("2-block float64 scalar transformer: attn1 assembles n via softmax weights 10^(4-i)/11111; "
                     "MLP1 cosine-pulse sieve over 65 primes with lateral inhibition writes factor digits; "
                     "attn2 Gaussian copy head; MLP2 ReLU gate; quadratic nearest-digit readout.")


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
