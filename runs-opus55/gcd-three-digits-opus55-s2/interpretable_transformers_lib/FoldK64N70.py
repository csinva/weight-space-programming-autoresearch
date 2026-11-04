"""
Interpretable transformer for character-level sequence tasks.

Task: gcd-three-digits, prompt "865,395=" -> answer "005".
The model is never trained; write_weights() sets every parameter in closed form.

Circuit (channels are named below):
  parser  attention whose softmax weights are place values 100:10:1 -> integers A, B
  euclid  one tied ReLU block repeated N_STEPS times: (A,B) -> (B, D(A,B)), where
          D is the distance from A to the nearest multiple of B (a triangular wave)
  readout attention scales g by position and subtracts already-emitted digits -> X
  head    parabola over digits, logit_d = -(X - c_d)^2, which floors X
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
# Experiment knobs
# ---------------------------------------------------------------------------
N_STEPS = 70          # tied Euclid iterations
FOLD_K = 64           # even number of knots in the triangular fold
D_MODEL = 20
N_HEADS = 2
DTYPE = torch.float64
SELFTEST_N = 2000     # random end-to-end prompts in the self-test (0 disables it)


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


class SimpleTransformer(nn.Module):
    """Attention parser, one tied residual MLP block looped, attention readout."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.parser = CausalSelfAttention(D_MODEL, N_HEADS)
        self.euclid = MLP(D_MODEL, FOLD_K + 2)
        self.readout = CausalSelfAttention(D_MODEL, N_HEADS)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        h = h + self.parser(h)
        for _ in range(N_STEPS):
            h = h + self.euclid(h)
        h = h + self.readout(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

VAL, A, B, X, P0 = 0, 1, 2, 3, 4   # value channels, then one-hot position channels
BIG = 40.0                         # attention logit that silences a position


def set_attention(attn, score_tables, value_channels, outputs):
    """Program attention. score_tables[h][p][j] is the logit of position p reading j.

    Head h reads the sum of channels value_channels[h]; outputs is (head, channel,
    gain) triples. The query is the one-hot position, so each key column holds a row
    of the table.
    """
    dh, d = attn.d_head, attn.d_head * attn.n_heads
    Wq, Wk, Wv, Wo = (torch.zeros(d, d, dtype=DTYPE) for _ in range(4))
    for h, table in enumerate(score_tables):
        for p, row in enumerate(table):
            Wq[h * dh + p, P0 + p] = math.sqrt(dh)
            for j, s in enumerate(row):
                Wk[h * dh + p, P0 + j] = s
        for ch in value_channels[h]:
            Wv[h * dh, ch] = 1.0
    for h, ch, gain in outputs:
        Wo[ch, h * dh] = gain
    attn.W_q.weight.copy_(Wq)
    attn.W_k.weight.copy_(Wk)
    attn.W_v.weight.copy_(Wv)
    attn.W_o.weight.copy_(Wo)


def place_row(weights):
    """Logits giving softmax weights proportional to {position: weight}; rest silenced."""
    return [math.log(weights[j]) if j in weights else -BIG for j in range(10)]


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in closed form. No training."""
    torch.manual_seed(0)
    model.to(DTYPE)
    for p in model.parameters():
        p.zero_()

    # Embeddings: digit value on VAL (',' and '=' are zero), one-hot position.
    for d in range(10):
        model.token_emb.weight[task.stoi[str(d)], VAL] = float(d)
    for p in range(10):
        model.pos_emb.weight[p, P0 + p] = 1.0

    # Parser: positions >= 7 read the two numbers; position 3 (the comma) reads
    # itself, so its A, B stay 0 and it can serve as the "zero" slot for readout.
    silent = [[0.0] * 10 for _ in range(10)]
    table_a, table_b = [list(r) for r in silent], [list(r) for r in silent]
    for p in (7, 8, 9):
        table_a[p] = place_row({0: 100.0, 1: 10.0, 2: 1.0})
        table_b[p] = place_row({4: 100.0, 5: 10.0, 6: 1.0})
    table_a[3] = table_b[3] = place_row({3: 1.0})
    set_attention(model.parser, [table_a, table_b], [[VAL], [VAL]], [(0, A, 111.0), (1, B, 111.0)])

    # Euclid: (A,B) -> (B, D(A,B)), where D is the triangular wave of distance to the
    # nearest multiple of B: D = A + sum_{m=1..K} (-1)^m 2 relu(A - m B/2).
    # Unit 0 carries A, unit 1 carries B, units 1+m are the knots of the wave.
    fc1, fc2 = model.euclid.fc1.weight, model.euclid.fc2.weight
    fc1[0, A], fc1[1, B] = 1.0, 1.0
    fc2[A, 0], fc2[A, 1] = -1.0, 1.0        # dA = B - A
    fc2[B, 0], fc2[B, 1] = 1.0, -1.0        # dB = (A - B) + knots
    for m in range(1, FOLD_K + 1):
        fc1[1 + m, A], fc1[1 + m, B] = 1.0, -m / 2
        fc2[B, 1 + m] = 2.0 * (-1) ** m

    # Readout: X = g * scale(p) - (digits already emitted), scaled to this place.
    # The fixed point is the 2-cycle (g,0) <-> (0,g), so g = A + B on either side.
    table_g = [list(r) for r in silent]
    table_d = [list(r) for r in silent]
    table_g[7] = place_row({7: 1.0, 3: 99.0})     # g/100
    table_g[8] = place_row({8: 1.0, 3: 9.0})      # g/10
    table_g[9] = place_row({9: 1.0})              # g
    table_d[7] = place_row({3: 1.0})              # nothing emitted yet
    table_d[8] = place_row({8: 10.0, 3: 100.0})   # 10*t1 / 110
    table_d[9] = place_row({8: 100.0, 9: 10.0})   # (100*t1 + 10*t2) / 110
    set_attention(model.readout, [table_g, table_d], [[A, B], [VAL]], [(0, X, 1.0), (1, X, -110.0)])

    # Head: logit_d = -(X - c_d)^2 = 2 c_d X - c_d^2 (dropping X^2), floors X.
    for d in range(10):
        c = d + 0.495
        model.head.weight[task.stoi[str(d)], X] = 2 * c
        model.head.bias[task.stoi[str(d)]] = -c * c
    model.head.bias[task.stoi[","]] = -1e9
    model.head.bias[task.stoi["="]] = -1e9

    if SELFTEST_N:
        steps_needed(model)
        selftest(model, task)


@torch.no_grad()
def selftest(model, task):
    """Independent check of the finished model on hard and random prompts.

    math.gcd is used only to grade the model's output here, never to produce it.
    """
    rng = random.Random(1)
    pairs = [(g, 0) for g in range(1000)] + [(0, g) for g in range(1000)]
    pairs += [(g, g) for g in range(1000)]
    pairs += [(rng.randrange(1000), rng.randrange(1000)) for _ in range(SELFTEST_N)]
    ids = torch.tensor([[task.stoi[c] for c in f"{a:03d},{b:03d}="] for a, b in pairs])
    for _ in range(3):
        next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    wrong = []
    for (a, b), row in zip(pairs, ids[:, 8:].tolist()):
        got = "".join(task.itos[i] for i in row)
        if got != f"{math.gcd(a, b):03d}":
            wrong.append((a, b, got))
    print(f"SELFTEST wrong={len(wrong)}/{len(pairs)} first={wrong[:8]}")


@torch.no_grad()
def steps_needed(model):
    """Grade the Euclid block alone on all 10^6 pairs (torch.gcd only grades it)."""
    idx = [A, B]
    w1, b1 = model.euclid.fc1.weight[:, idx], model.euclid.fc1.bias
    w2, b2 = model.euclid.fc2.weight[idx, :], model.euclid.fc2.bias[idx]
    ints = torch.cartesian_prod(torch.arange(1000), torch.arange(1000))
    good_budget, good_end, max_steps = 0, 0, 0
    for chunk in ints.split(100000):
        truth = torch.gcd(chunk[:, 0], chunk[:, 1]).to(DTYPE)
        h = chunk.to(DTYPE)
        is_good = lambda s: ((s.sum(1) == truth) & (s.min(1).values == 0)).sum().item()
        step, at_budget = 0, None
        while step < 3000:
            if step == N_STEPS:
                at_budget = is_good(h)
            if h.min(1).values.max() == 0 and step >= N_STEPS:
                break
            if h.min(1).values.max() == 0 and at_budget is None:
                at_budget = is_good(h)
            if h.min(1).values.max() > 0:
                max_steps = max(max_steps, step + 1)
            h = h + (F.relu(h @ w1.T + b1) @ w2.T + b2)
            step += 1
        good_budget += at_budget if at_budget is not None else is_good(h)
        good_end += is_good(h)
    print(f"ALLPAIRS exact_at_N_STEPS={good_budget / 1e6:.6f} steps_to_converge={max_steps} "
          f"exact_at_end={good_end / 1e6:.6f}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "FoldK64N70"
model_description = "Fold Euclid with 64 knots and 70 tied steps, testing the prediction that steps fall as roughly 4000/K."


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
