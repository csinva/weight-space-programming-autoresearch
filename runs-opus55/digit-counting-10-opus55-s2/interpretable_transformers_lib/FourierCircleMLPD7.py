"""Hand-written transformer for digit-counting-10. No training: see write_weights()."""

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
    """No LayerNorm: every signal keeps its literal value. The MLP is optional."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer: count at the target digit, then read and decode."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 7,
        n_heads: int = 1,
        n_layers: int = 2,
        d_ff: int = 11,
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
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff if i == n_layers - 1 else 0) for i in range(n_layers)]
        )
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-written circuit. Task: DDDDDDDDDDT= -> two-digit count of T among the D's.

    Dims: 0,1 digit on the unit circle | 2 'I am the target' | 3 'I emit tens' | 4 'I emit units'
          5 register c | 6 register r (the count as read at the answer positions)

    Layer 1, at the target: q=k=circle(digit), so score ~ cos(2pi(d-t)/10), and with a big
    scale attention is uniform over the k matches + the target itself. Only the target
    has a value (dim 2 -> 5), so c = 1/(k+1): softmax normalisation is the counter.
    Layer 2: answer positions attend to the target and copy c into r.
    MLP: piecewise-linear interpolation of circle(k % 10) through the 11 values of r.
    Head: logit_d = circle(d) . (dims 0,1), tied to the token embedding.
    """
    D = model.d_model
    for p in model.parameters():
        nn.init.zeros_(p)
    A = 100.0  # logit scale: must beat the '0'/'1' token echoed at pos 12
    GAP = 1 - math.cos(2 * math.pi / 10)        # cosine gap between neighbouring digits
    S = math.sqrt(40 * math.sqrt(D) / GAP)      # score margin 40 after the sqrt(dh) scaling
    L = math.sqrt(40 * math.sqrt(D))
    circ = lambda d: torch.tensor([math.cos(2 * math.pi * d / 10), math.sin(2 * math.pi * d / 10)])

    # Embeddings
    for d in range(10):
        model.token_emb.weight.data[d, :2] = circ(d)
    model.pos_emb.weight.data[10, 2] = 1.0
    model.pos_emb.weight.data[11, 3] = 1.0
    model.pos_emb.weight.data[12, 4] = 1.0

    # Layer 1: same-digit attention, value = target flag, into register c
    a = model.blocks[0].attn
    a.W_q.weight.data[0, 0] = a.W_q.weight.data[1, 1] = S
    a.W_k.weight.data[0, 0] = a.W_k.weight.data[1, 1] = S
    a.W_v.weight.data[5, 2] = 1.0
    a.W_o.weight.data[5, 5] = 1.0

    # Layer 2: answer positions look up the target, copy c -> r
    b = model.blocks[1].attn
    b.W_q.weight.data[0, 3] = L
    b.W_q.weight.data[0, 4] = L
    b.W_k.weight.data[0, 2] = L
    b.W_v.weight.data[6, 5] = 1.0
    b.W_o.weight.data[6, 6] = 1.0

    # MLP: y(r) passes through A*circle(k % 10) at r = 1/(k+1); knots run in ascending r.
    m = model.blocks[1].mlp
    G = 100.0  # gate strength
    knots = [1.0 / (k + 1) for k in range(10, -1, -1)]  # r for k = 10, 9, ..., 0
    ys = [circ(k % 10) * A for k in range(10, -1, -1)]
    slopes = [(ys[i + 1] - ys[i]) / (knots[i + 1] - knots[i]) for i in range(10)]
    for i in range(10):
        m.fc1.weight.data[i, 6] = 1.0
        m.fc1.weight.data[i, 3] = -G       # silenced at the tens position
        m.fc1.bias.data[i] = -knots[i]
        m.fc2.weight.data[:2, i] = slopes[i] - (slopes[i - 1] if i else 0)
    m.fc2.bias.data[:2] = circ(0) * A      # the k = 10 / default answer is '0'

    # Tens unit (index 10): fires only at tens position when r is below the k=9/10 midpoint
    mid = (knots[0] + knots[1]) / 2
    G2 = 1000.0
    m.fc1.weight.data[10, 6] = -G2
    m.fc1.weight.data[10, 4] = -G * G2
    m.fc1.bias.data[10] = G2 * mid
    m.fc2.weight.data[:2, 10] = A * (circ(1) - circ(0)) / (G2 * (mid - knots[0]))

    # Head: tied to the digit circle
    for d in range(10):
        model.head.weight.data[d, :2] = circ(d)


@torch.no_grad()
def self_check(model: SimpleTransformer, task) -> None:
    """Verification only, it does not influence the weights: exhaustive (target, k) grid."""
    def gen(prompts):
        ids = torch.tensor([task.encode(p) for p in prompts])
        for _ in range(task.answer_len):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], 1)
        return ["".join(task.itos[i] for i in row[-task.answer_len:].tolist()) for row in ids]

    g = torch.Generator().manual_seed(1)
    prompts, targets = [], []
    for t in range(10):
        for k in range(11):
            for _ in range(20):
                other = [d for d in range(10) if d != t]
                pick = torch.randint(0, 9, (10,), generator=g).tolist()
                digits = [other[i] for i in pick]
                for pos in torch.randperm(10, generator=g)[:k].tolist():
                    digits[pos] = t
                prompts.append("".join(map(str, digits)) + str(t) + "=")
                targets.append(f"{k:02d}")
    for ex in task.generate_examples(5000, seed=1):
        prompts.append(ex.prompt)
        targets.append(ex.answer)
    out = gen(prompts)
    bad = [(p, o, t) for p, o, t in zip(prompts, out, targets) if o != t]
    print(f"SELF-CHECK: {len(prompts) - len(bad)}/{len(prompts)} correct; first failures: {bad[:5]}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "FourierCircleMLPD7"
model_description = "Digits on a 2D Fourier circle: same-digit attention is a cosine; softmax counter gives 1/(k+1); MLP interpolates a circle vector read by a weight-tied head. d_model=7."


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
    self_check(model, task)
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
