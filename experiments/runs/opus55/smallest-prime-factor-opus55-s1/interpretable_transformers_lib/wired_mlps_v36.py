"""
Hand-written transformer computing the smallest prime factor by trial division in residue arithmetic.

  "03127=" -> "00053"
  First attention:   5 heads fetch the 5 digits; W_o writes S_p = sum_i d_i signed(10^i mod p).
  Fold MLPs:         repeated reflections |x - c|, one ReLU each, collapse the multiples of p.
  Last attention:    copies input digit t with a tiny weight (the answer if n is prime).
  Last MLP:          fires if x_p reached its target at position t, casting p's vote for its digit.

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

MAX_P = 313            # largest trial divisor; 317^2 > 99999, so 313 is needed for exactness
COPY_EPS = 0.25        # weight of the "n is prime, copy the input digit" fallback (below any margin)
BETA = 4.0             # attention sharpness; the attention score is BETA^2 / sqrt(d_head)
TOKEN = 2.0 ** -10     # embedding size: small enough that a token barely touches its own logit


def primes_upto(m):
    return [p for p in range(2, m + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def signed(x, p):
    x %= p
    return x - p if x > p // 2 else x


PRIMES = primes_upto(MAX_P)
P = len(PRIMES)


def residue(p, place, digit):
    """Contribution of a digit to S_p: the digit times the signed residue of 10^place mod p."""
    return digit * signed(10 ** place, p)


def fold_plan(p):
    """Reflection points that collapse all multiples of p onto one value at an end of the range.

    Reflecting x -> |x - c| about the middle of the multiples still in play merges them in pairs,
    and keeps them apart from every non-multiple. Returns (centres, target, side), where side is
    +1 if the target ends up at the top of the range and -1 if at the bottom.
    """
    lo = sum(min(residue(p, i, v) for v in range(10)) for i in range(5))
    hi = sum(max(residue(p, i, v) for v in range(10)) for i in range(5))
    targets = [c for c in range(lo - lo % p, hi + 1, p) if c >= lo]
    centres = []
    while not (len(targets) == 1 and targets[0] in (lo, hi)):
        n = len(targets)
        c = (targets[(n - 1) // 2] + targets[n // 2]) / 2
        targets = sorted({abs(t - c) for t in targets})
        lo, hi = 0, max(hi - c, c - lo)
        centres.append(c)
    return centres, targets[0], (-1 if targets[0] == lo else 1)


PLANS = [fold_plan(p) for p in PRIMES]
DEPTH = max(len(centres) for centres, _, _ in PLANS)
# For each fold block: (prime index, centre) of every prime that still has a fold to make.
FOLDS = [[(i, plan[0][b]) for i, plan in enumerate(PLANS) if b < len(plan[0])] for b in range(DEPTH)]


# Binary priority: each prime's vote is twice the next one's, so it beats all larger primes combined.
VOTE = {p: 2.0 ** (P - 1 - i) for i, p in enumerate(PRIMES)}

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
RES = _alloc(P)        # S_p for each prime, folded in place until multiples of p coincide
LOGIT = TOK            # tied: a digit's logit lives in the dimension of its embedding
D_MODEL = _n

# (output positions, prime index, digit). Each prime has one unit per distinct digit of its
# answer, open at every position where that digit appears.
GATE_UNITS = [(tuple(t for t in range(5) if f"{p:05d}"[t] == d), i, int(d))
              for i, p in enumerate(PRIMES) for d in sorted(set(f"{p:05d}"))]

# The hidden dimensions that the fold MLPs and the last MLP read and write.
RES_DIMS = list(range(RES.start, RES.stop))
GATE_READS = list(range(POS_OUT.start, POS_OUT.stop)) + RES_DIMS
GATE_WRITES = list(range(LOGIT.start, LOGIT.stop))

# (n_heads, d_head, d_ff, MLP reads, MLP writes) of each block.
BLOCK_SPECS = ([(5, 1, len(FOLDS[0]), RES_DIMS, RES_DIMS)]
               + [(0, 0, len(units), RES_DIMS, RES_DIMS) for units in FOLDS[1:]]
               + [(1, 10, len(GATE_UNITS), GATE_READS, GATE_WRITES)])


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
    """An MLP wired to a declared subset of the hidden dimensions."""

    def __init__(self, d_ff: int, reads: list[int], writes: list[int]):
        super().__init__()
        self.reads, self.writes = reads, writes
        self.fc1 = nn.Linear(len(reads), d_ff)
        self.fc2 = nn.Linear(d_ff, len(writes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[..., self.writes] = self.fc2(F.relu(self.fc1(x[..., self.reads])))
        return out


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, reads: list[int], writes: list[int]):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if n_heads else None
        self.mlp = MLP(d_ff, reads, writes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
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
            model.token_emb.weight[v, TOK.start + v] = TOKEN
        for t in range(model.max_seq_len):
            model.pos_emb.weight[t, BIAS.start] = 1
            if t < 5:
                model.pos_emb.weight[t, POS_IN.start + t] = 1
            elif t < 10:
                model.pos_emb.weight[t, POS_OUT.start + t - 5] = 1

        # Block 1 attention. Head j always attends to input position j and reads its digit as a
        # number; W_o multiplies it by the power of ten of that place, mod each prime.
        a = model.blocks[0].attn
        dh = a.d_head
        for j in range(5):
            a.W_q.weight[j * dh, BIAS.start] = BETA
            a.W_k.weight[j * dh, POS_IN.start + j] = BETA
            for v in range(10):
                a.W_v.weight[j * dh, TOK.start + v] = v / TOKEN
            for i, p in enumerate(PRIMES):
                a.W_o.weight[RES.start + i, j * dh] = residue(p, 4 - j, 1)

        # Fold MLPs. One ReLU per prime: with the residual, x + (2 relu(c - x) - c) = |x - c|.
        for block, units in zip(model.blocks, FOLDS):
            m = block.mlp
            for u, (i, c) in enumerate(units):
                m.fc1.weight[u, i] = -1
                m.fc1.bias[u] = c
                m.fc2.weight[i, u] = 2
                m.fc2.bias[i] = -c

        # Last attention. Output position t reads input digit t; a whisper on the logits, which is
        # the answer if n is prime. The score is sharpened to match the first attention's.
        last = model.blocks[-1]
        a = last.attn
        sharp = BETA * a.d_head ** 0.25
        for t in range(5):
            a.W_q.weight[t, POS_OUT.start + t] = sharp
            a.W_k.weight[t, POS_IN.start + t] = sharp
        for v in range(10):
            a.W_v.weight[v, TOK.start + v] = 1 / TOKEN
            a.W_o.weight[LOGIT.start + v, v] = COPY_EPS

        # Last MLP. relu(pos_t - 2 |x_p - target|) fires only at position t if p divides n, since
        # the target sits at an end of the range. Non-multiples are at least 1 away; the gain of 2
        # is a dead zone that keeps tiny attention leaks away from the huge votes in fc2.
        m = last.mlp
        for u, (ts, i, v) in enumerate(GATE_UNITS):
            _, target, side = PLANS[i]
            m.fc1.weight[u, 5 + i] = 2 * side
            for t in ts:
                m.fc1.weight[u, t] = 1
            m.fc1.bias[u] = -2 * side * target
            m.fc2.weight[v, u] = VOTE[PRIMES[i]]

        # Read the logits out.
        for v in range(10):
            model.head.weight[v, LOGIT.start + v] = 1

    nnz = sum((p != 0).sum().item() for p in model.parameters())
    print(f"nonzero={nnz} folds={[len(u) for u in FOLDS]} d_model={D_MODEL}")
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
model_shorthand_name = "wired_mlps_v36"
model_description = "Signed residues restored (unsigned cost one more block). Each MLP is wired to the dimensions it uses: fold MLPs touch only RES; the last reads POS_OUT and RES and writes logits."


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
