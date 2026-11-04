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
        return out


class SimpleTransformer(nn.Module):
    """2-layer causal attention stack with residuals. No MLP, no LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 4, n_heads: int = 1, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(13, d_model)  # a 14-token sequence feeds at most 13 inputs
        self.attns = nn.ModuleList([CausalSelfAttention(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for attn in self.attns:
            h = h + attn(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-written circuit. Task: DDDDDDDDDDT= -> two-digit count of T among the D's.

    Dims: 0,1 digit on the unit circle ('=' sits off it at E) | 2 signed flag f: +1 target,
          -1 answer positions, 0 digits | 3 register r

    Layer 1, at the target: q=k=circle(digit), so score ~ cos(2pi(d-t)/10), and with a big
    scale attention is uniform over the k matches + the target itself. Only the target
    has a positive value, so f becomes 1 + 1/(k+1): softmax is the counter.
    Layer 2: q=-f, k=+f, so f<0 positions see the target at a large positive score and
    themselves at a negative one; they copy its f into r.
    Head (no MLP): argmax over lines tangent to rho^2 is a nearest-prototype classifier
    on rho = r - 1, so logit_u = 2 p_u rho - p_u^2 with prototype p_u = 1/(u+1). The
    circle dims supply the constants: (1,0) after '0', circle(1) after '1', E at '='.
    """
    D = model.d_model
    for p in model.parameters():
        nn.init.zeros_(p)
    GAP = 1 - math.cos(2 * math.pi / 10)        # cosine gap between neighbouring digits
    S = math.sqrt(40 * math.sqrt(D) / GAP)      # score margin 40 after the sqrt(dh) scaling
    L = math.sqrt(60 * math.sqrt(D))
    circ = lambda d: torch.tensor([math.cos(2 * math.pi * d / 10), math.sin(2 * math.pi * d / 10)])

    # Embeddings
    for d in range(10):
        model.token_emb.weight.data[d, :2] = circ(d)
    # '=' is placed so that the head's tens decision falls at the k=9/10 midpoint.
    mid = (1 / 10 + 1 / 11) / 2
    model.token_emb.weight.data[10, :2] = torch.tensor([(2 + mid) / 1.75, 1.0])
    model.pos_emb.weight.data[10, 2] = 1.0
    model.pos_emb.weight.data[11, 2] = -1.0
    model.pos_emb.weight.data[12, 2] = -1.0

    # Layer 1: same-digit attention; the target adds its own share to its own flag
    a = model.attns[0]
    a.W_q.weight.data[0, 0] = a.W_q.weight.data[1, 1] = S
    a.W_k.weight.data[0, 0] = a.W_k.weight.data[1, 1] = S
    a.W_v.weight.data[2, 2] = 1.0

    # Layer 2: answer positions (f<0) look up the target (f>0), copy f -> r
    b = model.attns[1]
    b.W_q.weight.data[0, 2] = -L
    b.W_k.weight.data[0, 2] = L
    b.W_v.weight.data[3, 2] = 1.0

    # Head: tangent lines of rho^2 at the prototypes, rho = r - 1, constant from dim 0
    h = model.head.weight.data
    for u in range(10):
        p = 1.0 / (u + 1)
        h[u, 3] = 2 * p
        h[u, 0] = -2 * p - p * p
    # Dim 1 is 0 after a '0', and positive after '1' or at '=': it silences digits >= 2,
    # and favours '0' over '1', which gives units '0' for the count 10.
    h[0, 1] = 1.0
    h[2:10, 1] = -100.0
    h[10, 0] = -10.0           # '=' is never emitted


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
model_shorthand_name = "SignedFlagD4"
model_description = "One signed flag replaces the position flags (+1 target, -1 answer), so layer 2 finds the target by sign; '=' is placed off the circle to give the head its tens bias. d_model=4."


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
