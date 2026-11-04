"""
Interpretable transformer for smallest-prime-factor: "64507=" -> "00251".
Weights are hand-written, never trained.

Circuit (see write_weights):
  block 1 attention : 5 heads read the 5 prompt digits; W_o adds up signed residues y_p of n mod p
  block 1 MLP       : ReLU bumps at the multiples of p turn y_p into a flag F_p = (p divides n)
  block 2 MLP       : priority gate G_p = F_p AND no smaller prime flag
  block 3 attention : copy head: answer step k reads prompt digit k (default answer: n is prime)
  block 3 MLP       : (digit j of the winning prime) AND (step j) overrides the copied digit
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
# Hand-design plan
# ---------------------------------------------------------------------------

PRIMES = [p for p in range(2, 317) if all(p % q for q in range(2, p))]
M = len(PRIMES)
T_, P_ = 0, 11                        # token one-hot, position one-hot (16)
Y_, F_, G_, O_ = 27, 27 + M, 27 + 2 * M, 27 + 3 * M   # residue, flag, first-factor, output logits
D_MODEL = 235                         # 27 + 3 * 65 + 11 = 233, rounded up to a multiple of 5 heads
N_HEADS = 5
BETA = 40.0                           # attention sharpness


def signed_residue(a: int, p: int) -> int:
    r = a % p
    return r - p if r > p // 2 else r


def teeth_for(p: int) -> list[int]:
    """Multiples of p reachable by the sum of the five signed digit residues."""
    reach = {0}
    for i in range(5):
        reach = {y + signed_residue(d * 10 ** (4 - i), p) for y in reach for d in range(10)}
    return sorted(y for y in reach if y % p == 0)


TEETH = {p: teeth_for(p) for p in PRIMES}
N_BUMPS = sum(len(t) for t in TEETH.values())
D_FF = 3 * N_BUMPS


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
    """Pre-LN removed: the hand-written circuit needs exact integer signals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = 3,
        d_ff: int = D_FF,
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
        self.final_ln = nn.LayerNorm(d_model)
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in closed form. No training."""
    torch.manual_seed(0)
    for prm in model.parameters():
        prm.data.zero_()
    dh = model.blocks[0].attn.d_head
    sharp = BETA * math.sqrt(dh)

    # Embeddings: token one-hot, position one-hot.
    for t in range(11):
        model.token_emb.weight.data[t, T_ + t] = 1.0
    for s in range(16):
        model.pos_emb.weight.data[s, P_ + s] = 1.0

    # --- Block 1 attention: head i reads prompt digit i, W_o sums digit residues into y_p.
    b1 = model.blocks[0]
    for i in range(5):
        off = i * dh
        for s in range(5, 16):
            b1.attn.W_q.weight.data[off, P_ + s] = sharp      # answer positions look at ...
        b1.attn.W_k.weight.data[off, P_ + i] = 1.0            # ... prompt position i
        for d in range(10):
            b1.attn.W_v.weight.data[off + d, T_ + d] = 1.0    # the digit, as a one-hot
            for j, p in enumerate(PRIMES):
                b1.attn.W_o.weight.data[Y_ + j, off + d] = signed_residue(d * 10 ** (4 - i), p)

    # --- Block 1 MLP: a triangular bump of height 1 at every reachable multiple of p.
    u = 0
    for j, p in enumerate(PRIMES):
        for c in TEETH[p]:
            for off, gain in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                b1.mlp.fc1.weight.data[u, Y_ + j] = 1.0
                b1.mlp.fc1.bias.data[u] = off - c
                b1.mlp.fc2.weight.data[F_ + j, u] = gain
                u += 1

    # --- Block 2 MLP: priority gate, G_p = F_p AND NOT (any smaller prime flag).
    b2 = model.blocks[1]
    for j in range(M):
        b2.mlp.fc1.weight.data[j, F_ + j] = 1.0
        b2.mlp.fc1.weight.data[j, F_:F_ + j] = -1.0
        b2.mlp.fc2.weight.data[G_ + j, j] = 1.0

    # --- Block 3 attention: copy head, answer step k reads prompt digit k into the logits.
    b3 = model.blocks[2]
    for k in range(5):
        b3.attn.W_q.weight.data[k, P_ + 5 + k] = sharp
        b3.attn.W_k.weight.data[k, P_ + k] = 1.0
    for d in range(10):
        b3.attn.W_v.weight.data[d, T_ + d] = 1.0
        b3.attn.W_o.weight.data[O_ + d, d] = 1.0

    # --- Block 3 MLP: (digit j of the winning prime is d) AND (step j), overrides the copy.
    u = 0
    for jd in range(3):
        for d in range(10):
            for j, p in enumerate(PRIMES):
                if f"{p:03d}"[jd] == str(d):
                    b3.mlp.fc1.weight.data[u, G_ + j] = 1.0
            b3.mlp.fc1.weight.data[u, P_ + 7 + jd] = 1.0
            b3.mlp.fc1.bias.data[u] = -1.0
            b3.mlp.fc2.weight.data[O_ + d, u] = 3.0
            u += 1
    b3.mlp.fc1.weight.data[u, G_:G_ + M] = 1.0                # any factor at step 0 or 1: emit 0
    b3.mlp.fc1.weight.data[u, P_ + 5:P_ + 7] = 1.0
    b3.mlp.fc1.bias.data[u] = -1.0
    b3.mlp.fc2.weight.data[O_, u] = 3.0

    # Read the logits, final LayerNorm is a uniform shift + scale so argmax is unaffected.
    model.final_ln.weight.data.fill_(1.0)
    for v in range(11):
        model.head.weight.data[v, O_ + v] = 1.0
    exhaustive_check(model, task)


def exhaustive_check(model, task) -> None:
    """Verification only (not part of the model): test every n the prompt can hold."""
    def spf(n):
        for q in PRIMES:
            if q * q > n:
                break
            if n % q == 0:
                return q
        return n
    ns = list(range(2, 100000))
    prompts = [f"{n:05d}={spf(n):05d}" for n in ns]
    ids = torch.tensor([task.encode(s) for s in prompts])
    ok, wrong = 0, []
    with torch.no_grad():
        for b in range(0, len(ns), 5000):
            x = ids[b:b + 5000]
            hit = (model(x[:, :10])[:, 5:10].argmax(-1) == x[:, 6:11]).all(1)
            ok += int(hit.sum())
            wrong += [ns[b + int(i)] for i in (~hit).nonzero()[:, 0]]
    print(f"EXHAUSTIVE {ok}/{len(ns)} wrong[:10]={wrong[:10]}")
    for n in wrong[:3]:
        x = ids[n - 2:n - 1, :10]
        h = model.token_emb(x) + model.pos_emb(torch.arange(10))[None]
        print("n", n, "pred", model(x)[0, 5:10].argmax(-1).tolist())
        for bi, b in enumerate(model.blocks):
            h = b(h)
            print(" block", bi, "Y", h[0, 5, Y_:Y_ + 8].tolist(), "F", h[0, 5, F_:F_ + 8].tolist(), "G", h[0, 5, G_:G_ + 8].tolist())
    ex = task.generate_examples(200, seed=0)
    vals = [int(e.prompt[:5]) for e in ex]
    answers = [int(e.answer) for e in ex]
    print("sample n range", min(vals), max(vals), "prime", sum(a == v for a, v in zip(answers, vals)),
          "spf<30", sum(a < 30 for a in answers), "spf>=100 composite", sum(100 <= a != v for a, v in zip(answers, vals)))


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "relu_comb_v1"
model_description = "3 blocks without inner LN: attention sums signed residues for all 65 primes <=316, ReLU bumps at multiples flag divisibility, priority gate picks smallest, per-digit mux plus copy head for primes."


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
