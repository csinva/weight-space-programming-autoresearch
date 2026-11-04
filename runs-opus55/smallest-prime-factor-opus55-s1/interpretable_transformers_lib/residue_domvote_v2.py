"""
Hand-written transformer computing the smallest prime factor by trial division in residue arithmetic.

  "03127=" -> "00053"
  Block 1 attention: 5 heads fetch the 5 digits; W_o writes S_p = sum_i signed(d_i 10^i mod p).
  Block 1 MLP:       bumps at multiples of p give f_p = (p | n); each f_p votes 1/p^2 for p's digits.
  Block 2 attention: copies input digit t with a tiny weight (the answer if no prime divides n).
  Block 2 MLP:       gates the vote slot belonging to output position t onto the logits.

No training. LayerNorm is removed because it would destroy the exact integer arithmetic.
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
# Circuit layout
# ---------------------------------------------------------------------------

MAX_P = 313            # largest trial divisor; 317^2 > 99999
COPY_EPS = 0.25        # weight of the "n is prime, copy the input digit" fallback (below any vote)
GATE = 2.0 ** 22       # large enough to shut a vote slot off at the wrong position
BETA = 10.0            # attention sharpness


def primes_upto(m):
    return [p for p in range(2, m + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def signed(x, p):
    x %= p
    return x - p if x > p // 2 else x


PRIMES = primes_upto(MAX_P)
P = len(PRIMES)


def residue(p, place, digit):
    """Signed residue of digit * 10^place mod p."""
    return signed(digit * 10 ** place, p)


def multiples(p):
    """Multiples of p that S_p can actually reach."""
    lo = sum(min(residue(p, i, v) for v in range(10)) for i in range(5))
    hi = sum(max(residue(p, i, v) for v in range(10)) for i in range(5))
    return [c for c in range(lo - lo % p, hi + 1, p) if c >= lo]


BUMPS = [(i, c) for i, p in enumerate(PRIMES) for c in multiples(p)]


def domination_votes(limit=100000):
    """vote[p] = 1 + largest total vote of larger primes that can divide n together with p."""
    vote = {}
    def best(start, budget):
        total = 0
        for i in range(start, P):
            q = PRIMES[i]
            if q >= budget:
                break
            total = max(total, vote[q] + best(i + 1, budget // q))
        return total
    for i in reversed(range(P)):
        p = PRIMES[i]
        vote[p] = 1 + best(i + 1, (limit - 1) // p + 1)
    return vote


VOTE = domination_votes()
print("max vote", max(VOTE.values()), "sum", sum(VOTE.values()))

# Dimensions of the hidden state.
_n = 0
def _alloc(k):
    global _n
    s = slice(_n, _n + k)
    _n += k
    return s

TOK = _alloc(10)       # one-hot digit
POS_IN = _alloc(5)     # one-hot input position
POS_OUT = _alloc(5)    # one-hot output position
BIAS = _alloc(1)       # constant 1
RES = _alloc(P)        # S_p for each prime
ZERO = _alloc(1)       # vote "n is composite": the two leading answer digits are 0
HUND = _alloc(4)       # vote for the hundreds digit of p
TENS = _alloc(10)      # vote for the tens digit of p
ONES = _alloc(10)      # vote for the ones digit of p
LOGIT = _alloc(10)     # answer digit logits
D_MODEL = _n

# Output position -> vote slot. The two leading digits are "0" if any prime divides n.
SLOTS = [(0, ZERO), (1, ZERO), (2, HUND), (3, TENS), (4, ONES)]
GATE_UNITS = [(t, s, v) for t, s in SLOTS for v in range(s.stop - s.start)]

# (n_heads, d_head, d_ff) of each block.
BLOCK_SPECS = [(5, 11, 3 * len(BUMPS)), (1, 10, len(GATE_UNITS))]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_attn = n_heads * d_head
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)
        self.W_o = nn.Linear(d_attn, d_model, bias=False)

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
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Causal transformer without LayerNorm; blocks are configured by BLOCK_SPECS."""

    def __init__(self, vocab_size: int, max_seq_len: int = 11):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len = 11
        self.d_model = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, *spec) for spec in BLOCK_SPECS])
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate every parameter in closed form. No training."""
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Embeddings: one-hot digit, one-hot position, constant bias.
        for v in range(10):
            model.token_emb.weight[v, TOK.start + v] = 1
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, BIAS.start] = 1
            if t < 5:
                model.pos_emb.weight[t, POS_IN.start + t] = 1
            elif t < 10:
                model.pos_emb.weight[t, POS_OUT.start + t - 5] = 1

        # Block 1 attention. Head j always attends to input position j and reads its digit;
        # W_o turns "digit v at head j" into its residue contribution for every prime.
        b1, b2 = model.blocks
        a = b1.attn
        dh = a.d_head
        for j in range(5):
            a.W_q.weight[j * dh, BIAS.start] = BETA
            a.W_k.weight[j * dh, POS_IN.start + j] = BETA
            for v in range(10):
                a.W_v.weight[j * dh + 1 + v, TOK.start + v] = 1
                for i, p in enumerate(PRIMES):
                    a.W_o.weight[RES.start + i, j * dh + 1 + v] = residue(p, 4 - j, v)

        # Block 1 MLP. relu(S-c+1) - 2 relu(S-c) + relu(S-c-1) is 1 at S=c, 0 at other integers.
        m = b1.mlp
        for u, (i, c) in enumerate(BUMPS):
            p = PRIMES[i]
            vote = VOTE[p]
            slots = [ZERO.start, HUND.start + p // 100, TENS.start + p // 10 % 10, ONES.start + p % 10]
            for k, (off, w) in enumerate([(1, 1.0), (0, -2.0), (-1, 1.0)]):
                m.fc1.weight[3 * u + k, RES.start + i] = 1
                m.fc1.bias[3 * u + k] = off - c
                for s in slots:
                    m.fc2.weight[s, 3 * u + k] = w * vote

        # Block 2 attention. Output position t reads input digit t; a whisper on the logits.
        a = b2.attn
        for t in range(5):
            a.W_q.weight[t, POS_OUT.start + t] = BETA
            a.W_k.weight[t, POS_IN.start + t] = BETA
        for v in range(10):
            a.W_v.weight[v, TOK.start + v] = 1
            a.W_o.weight[LOGIT.start + v, v] = COPY_EPS

        # Block 2 MLP. relu(vote + GATE*pos_t - GATE) passes the vote only at position t.
        m = b2.mlp
        for u, (t, s, v) in enumerate(GATE_UNITS):
            m.fc1.weight[u, s.start + v] = 1
            m.fc1.weight[u, POS_OUT.start + t] = GATE
            m.fc1.bias[u] = -GATE
            m.fc2.weight[LOGIT.start + v, u] = 1

        # Read the logits out.
        for v in range(10):
            model.head.weight[v, LOGIT.start + v] = 1

    exhaustive_check(model, task)


def exhaustive_check(model, task, top=100000):
    """Diagnostic only: compares the circuit with a sieve on every n in 2..99999."""
    spf = list(range(top))
    for i in range(2, int(top ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, top, i):
                if spf[j] == j:
                    spf[j] = i
    ns = torch.arange(2, top)
    answers = torch.tensor(spf[2:])
    places = [10000, 1000, 100, 10, 1]
    ids = torch.stack([ns // q % 10 for q in places]
                      + [torch.full_like(ns, 10)]
                      + [answers // q % 10 for q in places[:-1]], dim=1)
    target = torch.stack([answers // q % 10 for q in places], dim=1)
    with torch.no_grad():
        pred = torch.cat([model(ids[i:i + 4096])[:, 5:10].argmax(-1) for i in range(0, len(ids), 4096)])
    ok = (pred == target).all(1)
    is_prime = answers == ns
    print(f"EXHAUSTIVE all={ok.float().mean():.5f} primes={ok[is_prime].float().mean():.5f} "
          f"composites={ok[~is_prime].float().mean():.5f} errors={(~ok).sum().item()} first={ns[~ok][:8].tolist()}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "residue_domvote_v2"
model_description = "As v1 but integer domination votes: vote(p) = 1 + max total vote of larger primes that can co-divide n, so the smallest divisor always wins."


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
