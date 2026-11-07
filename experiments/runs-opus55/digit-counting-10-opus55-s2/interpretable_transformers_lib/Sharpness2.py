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
    """Single-head attention: score = x diag(m) x'; the value is always the flag channel."""

    def __init__(self, d_model: int):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        scores = (x * self.m) @ x.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        out = torch.zeros_like(x)
        out[..., 2:] = scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ x[..., 2:]
        return out


class SimpleTransformer(nn.Module):
    """2-layer causal attention stack with residuals. No MLP, no LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 3, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.target_mark = nn.Parameter(torch.zeros(d_model))  # added at position 10
        self.echo_mark = nn.Parameter(torch.zeros(d_model))    # added at position 12
        self.attns = nn.ModuleList([CausalSelfAttention(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 10, bias=False)  # only digits are ever emitted

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        marks = torch.zeros(ids.shape[1], self.d_model, device=ids.device)
        marks[10] = self.target_mark
        if ids.shape[1] > 12:
            marks[12] = self.echo_mark
        h = self.token_emb(ids) + marks
        for attn in self.attns:
            h = h + attn(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Hand-written circuit. Task: DDDDDDDDDDT= -> two-digit count of T among the D's.

    Dims: 0,1 digit on the unit circle ('=' at E; position 12 is shifted by (1,0))
          2 signed flag f: +1 target, -1 answer positions, 0 digits

    Layer 1, at the target: q=k=circle(digit), so score ~ cos(2pi(d-t)/10), and with a big
    scale attention is uniform over the k matches + the target itself, so f becomes
    1 + 1/(k+1): softmax is the counter. The answer positions lie outside the unit circle,
    so they attend to themselves only, which takes their f to exactly -2.
    Layer 2: q=-f, k=+f, so f<0 positions attend to the target and add its f to their
    own: f = -2 + 1 + rho = rho - 1, with rho = 1/(k+1).
    Head (no MLP): argmax over lines tangent to rho^2 is a nearest-prototype classifier,
    so logit_u = 2 p_u rho - p_u^2 with prototype p_u = 1/(u+1). The circle dims supply
    the constants: (2,0) after '0', (1.81,0.59) after '1', E at '='.
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    MARGIN = 2.0                                # score gap between a match and the nearest non-match
    GAP = 1 - math.cos(2 * math.pi / 10)        # cosine gap between neighbouring digits
    circ = lambda d: torch.tensor([math.cos(2 * math.pi * d / 10), math.sin(2 * math.pi * d / 10)])

    # Embeddings
    for d in range(10):
        model.token_emb.weight.data[d, :2] = circ(d)
    # '=' is placed so that the head's tens decision falls at the k=9/10 midpoint:
    # with beta(0)-beta(1) = 2 and alpha(0)-alpha(1) = 1/8, solve for E[1] given E[0] = 1.5.
    mid = (1 / 10 + 1 / 11) / 2
    model.token_emb.weight.data[10] = torch.tensor([1.5, (1 - mid - 0.125 * 1.5) / 2, -1.0])
    model.target_mark.data[:] = torch.tensor([0.0, 0.0, 1.0])
    model.echo_mark.data[:] = torch.tensor([1.0, 0.0, -1.0])  # off the circle: self-attends

    # Layer 1: same-digit attention; the target adds its own share to its own flag
    model.attns[0].m.data[:] = torch.tensor([MARGIN / GAP, MARGIN / GAP, 0.0])

    # Layer 2: answer positions (f<0) look up the target (f>0), add its f to their own
    model.attns[1].m.data[:] = torch.tensor([0.0, 0.0, -1.5 * MARGIN])

    # Head: tangent lines of rho^2 at the prototypes; dim 2 = rho - 1, constant from dim 0 = 2
    h = model.head.weight.data
    for u in range(10):
        p = 1.0 / (u + 1)
        h[u, 2] = 2 * p
        h[u, 0] = (2 * p - p * p) / 2
    # Dim 1 is 0 after a '0', and positive after '1' or at '=': it silences digits >= 2,
    # and favours '0' over '1', which gives units '0' for the count 10.
    h[0, 1] = 2.0
    h[2:10, 1] = -100.0


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
    ids = torch.tensor([task.encode(p + o) for p, o in zip(prompts, out)])
    top2 = model(ids[:, :-1])[:, -task.answer_len:].topk(2, dim=-1).values
    print(f"MIN LOGIT MARGIN: {(top2[..., 0] - top2[..., 1]).min().item():.3e}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Sharpness2"
model_description = "Sharpness sweep: MetricAttention with the attention score margin lowered to 2, so non-matches leak about e^-2 of attention each. Expected to break the count."


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
