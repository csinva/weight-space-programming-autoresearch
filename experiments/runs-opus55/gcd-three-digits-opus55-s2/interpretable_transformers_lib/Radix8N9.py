"""
Interpretable transformer for gcd-three-digits: "865,395=" -> "005".
Never trained; write_weights() sets every parameter in closed form.

Each token carries a 4-channel state (VAL, A, B, X) and a one-hot position.
  parser  six heads each attend to one digit; W_o holds place values, so A and B are
          exact integers in any precision
  euclid  a tied round of composed ReLU folds repeated N_STEPS times. Each fold moves
          A to its distance from the nearest multiple of s*B (a triangular wave),
          which keeps gcd(A,B); the last fold (s=1) also swaps A and B
  readout attention scales g by position and subtracts already-emitted digits -> X
  head    parabola over digits, logit_d = -(X - c_d)^2, which floors X
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
# Experiment knobs
# ---------------------------------------------------------------------------
N_STEPS = 9           # tied Euclid rounds
FOLDS = [(64, 8), (8, 8), (1, 8)]   # (scale, even knots); the last fold also swaps
DTYPE = torch.float32
BIG = 40.0            # attention logit that silences a position
SELFTEST_N = 1        # nonzero runs the all-pairs checks before evaluation

VAL, A, B, X = 0, 1, 2, 3   # state channels
N_STATE, N_POS = 4, 10


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class PositionAttention(nn.Module):
    """Causal multi-head attention: positions address, state channels are carried."""

    def __init__(self, n_heads: int, d_key: int):
        super().__init__()
        self.n_heads, self.d_key = n_heads, d_key
        self.W_q = nn.Linear(N_POS, n_heads * d_key, bias=False)
        self.W_k = nn.Linear(N_POS, n_heads * d_key, bias=False)
        self.W_v = nn.Linear(N_STATE, n_heads, bias=False)
        self.W_o = nn.Linear(n_heads, N_STATE, bias=False)

    def forward(self, state: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        T = pos.shape[1]
        H, dk = self.n_heads, self.d_key
        q = self.W_q(pos).view(1, T, H, dk).transpose(1, 2)
        k = self.W_k(pos).view(1, T, H, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=pos.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        v = self.W_v(state).transpose(1, 2).unsqueeze(-1)
        return self.W_o((attn @ v).squeeze(-1).transpose(1, 2))


class Fold(nn.Module):
    """Residual MLP acting on the (A, B) registers only."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(2, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, 2, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(F.relu(self.fc1(state[..., A:B + 1])))
        return state + F.pad(delta, (A, N_STATE - B - 1))


class SimpleTransformer(nn.Module):
    """Attention parser, tied residual MLP folds, attention readout."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, 1)
        self.pos_emb = nn.Embedding(N_POS, N_POS)
        self.parser = PositionAttention(n_heads=6, d_key=1)
        last = len(FOLDS) - 1
        self.folds = nn.ModuleList(
            [Fold(k + (2 if i == last else 0)) for i, (_, k) in enumerate(FOLDS)]
        )
        self.readout = PositionAttention(n_heads=2, d_key=3)
        self.head = nn.Linear(1, vocab_size, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        val = self.token_emb(ids)
        state = F.pad(val, (0, N_STATE - 1))
        pos = self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        state = state + self.parser(state, pos)
        for _ in range(N_STEPS):
            for fold in self.folds:
                state = fold(state)
        state = state + self.readout(state, pos)
        return self.head(state[..., X:X + 1])


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def place_row(weights):
    """Logits giving softmax weights proportional to {position: weight}. Chosen
    positions sit at +BIG and the rest stay at 0, so ignored positions need no weight."""
    return [BIG + math.log(weights[j]) if j in weights else 0.0 for j in range(N_POS)]


def set_attention(attn, slots, tables, value_channels, outputs):
    """Program attention. slots maps a position to its query slot; tables[h][slot] is
    that slot's row of logits over positions. Head h carries the sum of the channels
    value_channels[h]; outputs is (head, channel, gain) triples."""
    dk = attn.d_key
    for h, table in enumerate(tables):
        for p, slot in slots.items():
            attn.W_q.weight[h * dk + slot, p] = math.sqrt(dk)
        for slot, row in enumerate(table):
            for j, s in enumerate(row):
                attn.W_k.weight[h * dk + slot, j] = s
        for ch in value_channels[h]:
            attn.W_v.weight[h, ch] = 1.0
    for h, ch, gain in outputs:
        attn.W_o.weight[ch, h] = gain


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in closed form. No training."""
    torch.manual_seed(0)
    model.to(DTYPE)
    for p in model.parameters():
        p.zero_()

    # Embeddings: the digit's value (',' and '=' are zero) and a one-hot position.
    for d in range(10):
        model.token_emb.weight[task.stoi[str(d)], 0] = float(d)
    model.pos_emb.weight.copy_(torch.eye(N_POS, dtype=DTYPE))

    # Parser: each head puts all its attention on one digit, and W_o multiplies it by
    # its place value. Query +1 at output positions selects the digit (key +BIG);
    # query -1 at the comma flips the signs, so the comma (key -BIG) reads itself, its
    # A, B stay 0, and readout can use it as a "zero" slot.
    digits = [(0, A, 100.0), (1, A, 10.0), (2, A, 1.0), (4, B, 100.0), (5, B, 10.0), (6, B, 1.0)]
    for h, (j, ch, gain) in enumerate(digits):
        model.parser.W_q.weight[h, 3] = -1.0
        model.parser.W_q.weight[h, 7:10] = 1.0
        model.parser.W_k.weight[h, 3] = -BIG
        model.parser.W_k.weight[h, j] = BIG
        model.parser.W_v.weight[h, VAL] = 1.0
        model.parser.W_o.weight[ch, h] = gain

    # Euclid: each fold replaces A by its distance to the nearest multiple of s*B,
    # D_s = A + sum_{m=1..K} (-1)^m 2 relu(A - m s B/2). Any such move keeps
    # gcd(A,B). The final fold also swaps, (A,B) -> (B, D_1(A,B)), using units 0 and 1
    # to carry A and B. Register indices here are 0 for A and 1 for B.
    for i, (s, k) in enumerate(FOLDS):
        fc1, fc2 = model.folds[i].fc1.weight, model.folds[i].fc2.weight
        swap = i == len(FOLDS) - 1
        off = 2 if swap else 0
        if swap:
            fc1[0, 0], fc1[1, 1] = 1.0, 1.0
            fc2[0, 0], fc2[0, 1] = -1.0, 1.0   # dA = B - A
            fc2[1, 0], fc2[1, 1] = 1.0, -1.0   # dB = (A - B) + knots
        for m in range(1, k + 1):
            fc1[off + m - 1, 0], fc1[off + m - 1, 1] = 1.0, -s * m / 2
            fc2[1 if swap else 0, off + m - 1] = 2.0 * (-1) ** m

    # Readout: X = g * scale(p) - (digits already emitted), scaled to this place.
    # The fixed point is the 2-cycle (g,0) <-> (0,g), so g = A + B on either side.
    set_attention(
        model.readout, {7: 0, 8: 1, 9: 2},
        [[place_row({7: 1.0, 3: 99.0}), place_row({8: 1.0, 3: 9.0}), place_row({9: 1.0})],
         [place_row({3: 1.0}), place_row({8: 10.0, 3: 100.0}), place_row({8: 100.0, 9: 10.0})]],
        [[A, B], [VAL]], [(0, X, 1.0), (1, X, -110.0)],
    )

    # Head: logit_d = -(X - c_d)^2 = 2 c_d X - c_d^2 (dropping X^2), which floors X.
    for d in range(10):
        c = d + 0.495
        model.head.weight[task.stoi[str(d)], 0] = 2 * c
        model.head.bias[task.stoi[str(d)]] = -c * c
    model.head.bias[task.stoi[","]] = -1e9
    model.head.bias[task.stoi["="]] = -1e9

    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"WEIGHTS nonzero={nonzero}/{total}")
    if SELFTEST_N:
        steps_needed(model)
        selftest(model, task)


@torch.no_grad()
def selftest(model, task):
    """Run every prompt through the finished model. torch.gcd only grades its output."""
    ints = torch.cartesian_prod(torch.arange(1000), torch.arange(1000))
    digit = torch.tensor([task.stoi[str(d)] for d in range(10)])
    value = torch.zeros(task.vocab_size, dtype=torch.long)
    value[digit] = torch.arange(10)
    comma, equals = task.stoi[","], task.stoi["="]
    wrong, examples = 0, []
    for chunk in ints.split(100000):
        a, b = chunk[:, 0], chunk[:, 1]
        cols = [digit[a // 100], digit[a // 10 % 10], digit[a % 10], torch.full_like(a, comma),
                digit[b // 100], digit[b // 10 % 10], digit[b % 10], torch.full_like(a, equals)]
        ids = torch.stack(cols, dim=1)
        for _ in range(3):
            next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        got = 100 * value[ids[:, 8]] + 10 * value[ids[:, 9]] + value[ids[:, 10]]
        bad = got != torch.gcd(a, b)
        wrong += bad.sum().item()
        examples += chunk[bad][:4].tolist()
    print(f"END_TO_END wrong={wrong}/{len(ints)} first={examples[:8]}")


@torch.no_grad()
def steps_needed(model):
    """Grade the folds alone on all 10^6 pairs. torch.gcd only grades them."""
    layers = [(f.fc1.weight, f.fc2.weight) for f in model.folds]
    ints = torch.cartesian_prod(torch.arange(1000), torch.arange(1000))
    good_budget, good_end, max_steps = 0, 0, 0
    for chunk in ints.split(100000):
        truth = torch.gcd(chunk[:, 0], chunk[:, 1]).to(DTYPE)
        h = chunk.to(DTYPE)
        is_good = lambda s: ((s.sum(1) == truth) & (s.min(1).values == 0)).sum().item()
        step, at_budget = 0, None
        while step < 3000:
            live = h.min(1).values.max() > 0
            if step == N_STEPS or (not live and at_budget is None):
                at_budget = is_good(h)
            if not live and step >= N_STEPS:
                break
            if live:
                max_steps = max(max_steps, step + 1)
            for w1, w2 in layers:
                h = h + F.relu(h @ w1.T) @ w2.T
            step += 1
        good_budget += at_budget if at_budget is not None else is_good(h)
        good_end += is_good(h)
    print(f"ALLPAIRS exact_at_N_STEPS={good_budget / 1e6:.6f} steps_to_converge={max_steps} "
          f"exact_at_end={good_end / 1e6:.6f}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Radix8N9"
model_description = "Radix-8 division: three 8-knot folds at scales 64b, 8b, b (26 hidden units), the remaining point on the depth versus sparsity curve."


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
