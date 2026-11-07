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
        # LayerNorms removed (identity) so the hand-written circuits stay exactly linear.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 160,
        n_heads: int = 5,
        n_layers: int = 2,
        d_ff: int = 1024,
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

# Residual stream layout
TOK = 0          # 11 dims: token one-hot
POS = 11         # 16 dims: position one-hot
SLOT = 27        # 50 dims: gathered prompt digits, slot i*10+c  (i = digit index 0..4)
DFLAG = 77       # 65 dims: D_p = [p divides N] for each prime p <= 313
LOGIT = 142      # 11 dims: output logits


def small_primes(limit: int = 316) -> list[int]:
    return [n for n in range(2, limit + 1) if all(n % q for q in range(2, int(n ** 0.5) + 1))]


def write_weights(model: SimpleTransformer, task) -> None:
    """Closed-form circuit:
    L1 attn : 5 heads, head i attends to prompt position i, copies digit one-hot into slot i.
    L1 MLP  : for each prime p<=313, T_p = sum_i (d_i*10^(4-i) mod p) is linear in the slots,
              and p | N  <=>  T_p in {0,p,2p,3p,4p}.  Integer "bump" ReLU triples give D_p.
    L2 MLP  : first-divisor selection F_p = relu(D_p - sum_{q<p} D_q), gated by answer position j,
              writes digit j of p; if no D_p fires (N prime), copy digit j of N.
    """
    torch.manual_seed(0)
    primes = small_primes()
    assert len(primes) == 65 and primes[-1] == 313
    D = model.d_model
    dh = model.blocks[0].attn.d_head
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # embeddings
        for t in range(model.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1.0
        for q in range(model.max_seq_len):
            model.pos_emb.weight[q, POS + q] = 1.0

        # ---- layer 1 attention: gather digits ----
        a = model.blocks[0].attn
        BIG = 100.0 * math.sqrt(dh)
        for i in range(5):
            h0 = i * dh
            for q in range(model.max_seq_len):
                a.W_q.weight[h0, POS + q] = 1.0          # constant-1 query
            a.W_k.weight[h0, POS + i] = BIG               # key fires on position i
            for c in range(10):
                a.W_v.weight[h0 + c, TOK + c] = 1.0
                a.W_o.weight[SLOT + i * 10 + c, h0 + c] = 1.0

        # ---- layer 1 MLP: divisibility flags ----
        m1 = model.blocks[0].mlp
        n = 0
        for k, p in enumerate(primes):
            coef = torch.zeros(D)
            for i in range(5):
                for c in range(10):
                    coef[SLOT + i * 10 + c] = (c * 10 ** (4 - i)) % p
            for m in range(5):
                for off, w in ((1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)):
                    m1.fc1.weight[n] = coef
                    m1.fc1.bias[n] = -m * p + off
                    m1.fc2.weight[DFLAG + k, n] = w
                    n += 1
        assert n <= model.d_ff

        # ---- layer 2 MLP: select smallest divisor / copy N ----
        m2 = model.blocks[1].mlp
        OUT = 10.0
        n = 0
        for k, p in enumerate(primes):
            ps = f"{p:05d}"
            for j in range(5):
                m2.fc1.weight[n, DFLAG + k] = 1.0
                for kk in range(k):
                    m2.fc1.weight[n, DFLAG + kk] = -1.0
                m2.fc1.weight[n, POS + 5 + j] = 1.0
                m2.fc1.bias[n] = -1.0
                m2.fc2.weight[LOGIT + int(ps[j]), n] = OUT
                n += 1
        for j in range(5):
            for c in range(10):
                m2.fc1.weight[n, SLOT + j * 10 + c] = 1.0
                m2.fc1.weight[n, POS + 5 + j] = 1.0
                for kk in range(len(primes)):
                    m2.fc1.weight[n, DFLAG + kk] = -1.0
                m2.fc1.bias[n] = -1.0
                m2.fc2.weight[LOGIT + c, n] = OUT
                n += 1
        assert n <= model.d_ff

        # ---- unembedding ----
        for t in range(model.vocab_size):
            model.head.weight[t, LOGIT + t] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "SPF_ModSumBump_v1"
model_description = ("2-layer no-LN transformer: attention gathers 5 digits; MLP1 tests p|N for 65 primes<=313 via "
                     "integer bump on sum of digit residues; MLP2 picks first divisor (or copies N if prime).")


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
