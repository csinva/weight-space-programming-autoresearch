"""
Interpretable transformer for `linear-interpolation-two-points`:
prompt "x1,y1;x2,y2;x=" -> two-digit answer y, e.g. "4,16;5,20;3=" -> "12".

Hand-written circuit (never trained). Every d_model channel has a name:
  EMBED  A digit token is -10d in `out`: printing tens digit t subtracts 10t, and
         READ recovers d from it. Positions carry flags and a compass address.
  READ   Block 1 attention: one head per number, pointing along that number's
         compass direction. Attention weights 10:1 on the tens:units characters
         implement place value. At the tens step each head also attends to '='
         (value 0), so numbers read at 1/10 volume: the same circuit then yields
         y/10. Prompt positions attend to ',' so their registers stay 0. W_o
         stores dy, dx, dq = x - x1 and writes the null-slope answer y1 + 3 dq.
  FIT    Block 2 attention (no MLP anywhere): one correction head per other
         slope k. With e_k = dy - k dx, q = e_k and key = -beta e_k give a self
         score of -beta e_k^2, so the head stays on itself only if slope k fits
         (adding (k-3) dq to out) and otherwise falls back to the zero-valued ','.
  PRINT  Linear head rounds out: logit_d = 2d(out - 0.45 phase) - d^2, the
         tangents of -(out-d)^2. The tens step's shift turns round into floor.

Usage:
    uv run interpretable_transformer.py --task linear-interpolation-two-points
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

NULL_SLOPE = 3            # assumed unless a correction head finds a better fit
SLOPES = [1, 2, 4, 5]
READS = ["x1", "y1", "x2", "y2", "x"]
NAMES = ["dig", "live", "phase", "stay", "a_cos", "a_sin", "dy", "dx", "dq", "out",
         "anchor_hi", "anchor_lo", "zero"]
C = {n: i for i, n in enumerate(NAMES)}


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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in closed form. No training."""
    for p in model.parameters():
        p.zero_()

    # Embeddings: a digit token is its value in dig (what READ sees, and never
    # overwritten) and -10d in out (its worth as a printed tens digit).
    for d in range(10):
        model.token_emb.weight[task.stoi[str(d)], C["dig"]] = d
        model.token_emb.weight[task.stoi[str(d)], C["out"]] = -10 * d
    pos = model.pos_emb.weight
    pos[11:, C["live"]] = 1      # the two answer steps are the only ones that compute
    pos[:11, C["phase"]] = -1    # prompt: attend to the fallback
    pos[11, C["phase"]] = 1      # tens step: 1/10 volume; units step stays 0
    L = 20.0  # attention sharpness: selected characters beat the rest by e^L
    place_value = {"x1": {0: L}, "y1": {2: L + math.log(10), 3: L}, "x2": {5: L},
                   "y2": {7: L + math.log(10), 8: L}, "x": {10: L}}
    # Each number lies in its own compass direction; radius is its place value.
    angle = {r: 2 * math.pi * h / len(READS) for h, r in enumerate(READS)}
    for r, slots in place_value.items():
        for p, radius in slots.items():
            pos[p, C["a_cos"]] = radius * math.cos(angle[r])
            pos[p, C["a_sin"]] = radius * math.sin(angle[r])
    pos[1, C["stay"]] = 100.0    # the ',' at position 1 is the zero-valued fallback,
    pos[11, C["stay"]] = 110.0   # a fitting head outranks the fallback,
    pos[12, C["stay"]] = 120.0   # with the newest step outranking the previous one.

    # READ: head h asks "where is number h?" and returns the digit value there.
    # A two-digit number reads (10a + b) / 11, so W_o rescales by 11.
    # Key dim 0 selects the number. Key dim 1 gives the zero-valued '=' 9x the
    # number's total weight at the '=' step, for 1/10 volume. Muted positions
    # (query -1 on dim 1) go to ','. Position 0 can only see itself, but it reads
    # the same digit for every number, so its e_k is still 0.
    # W_o keeps what the formula needs: dy, dx, dq = x - x1, and already writes
    # the null-slope answer y1 + 3 dq into out.
    # Default model: 4 heads per block, 16 dims per head, so each head's row in
    # q/k/v starts at 16*h, and scores are divided by sqrt(16): keys are scaled by 4.
    dh, root = model.blocks[0].attn.d_head, math.sqrt(model.blocks[0].attn.d_head)
    wiring = {"x1": [("dx", -1), ("dq", -1), ("out", -NULL_SLOPE)],
              "y1": [("out", 11), ("dy", -11)],
              "x2": [("dx", 1)], "y2": [("dy", 11)],
              "x": [("dq", 1), ("out", NULL_SLOPE)]}
    for b, h, r in [(0, 0, "x1"), (0, 1, "y1"), (0, 2, "x2"), (0, 3, "y2"), (1, 0, "x")]:
        read = model.blocks[b].attn
        two_digit = r.startswith("y")
        read.W_q.weight[dh * h, C["live"]] = 1
        read.W_k.weight[dh * h, C["a_cos"]] = root * math.cos(angle[r])
        read.W_k.weight[dh * h, C["a_sin"]] = root * math.sin(angle[r])
        read.W_q.weight[dh * h + 1, C["phase"]] = 1
        read.W_k.weight[dh * h + 1, C["live"]] = root * (L + math.log(99 if two_digit else 9) + 110.0)
        read.W_k.weight[dh * h + 1, C["stay"]] = -root
        read.W_v.weight[dh * h, C["dig"]] = 1
        for reg, gain in wiring[r]:
            read.W_o.weight[C[reg], dh * h] = gain

    # FIT: head k scores positions by e_k(t) * -beta e_k(j) + stay(j). A wrong
    # slope has e_k^2 >= 1 (>= 0.01 at 1/10 volume), which outweighs `stay`.
    fit = model.blocks[2].attn
    beta = 2000.0
    for h, k in enumerate(SLOPES):
        for w, c in [(1, "dy"), (-k, "dx")]:
            fit.W_q.weight[dh * h, C[c]] = w
            fit.W_k.weight[dh * h, C[c]] = -root * beta * w
        fit.W_q.weight[dh * h + 1, C["live"]] = 1
        fit.W_k.weight[dh * h + 1, C["stay"]] = root
        fit.W_v.weight[dh * h, C["dq"]] = k - NULL_SLOPE  # correct the null-slope answer
        fit.W_o.weight[C["out"], dh * h] = 1

    # Head: logit_d = 2d*(out - 0.45 phase) - d^2 rounds `out` at the units step
    # and floors it at the tens step; punctuation is never printed.
    for tok, i in task.stoi.items():
        if tok.isdigit():
            d = int(tok)
            model.head.weight[i, C["out"]] = 2 * d
            model.head.weight[i, C["phase"]] = -0.9 * d
            model.head.weight[i, C["live"]] = -d * d
        else:
            model.head.weight[i, C["live"]] = -1e3

    # LayerNorm, tamed. Huge opposite anchors pin the std near sigma, and matching
    # LN gains undo that scale. The mean mu is still subtracted from every channel,
    # but `zero` reads as exactly -mu, so a row with weights summing to s cancels
    # mu by putting -s on `zero`. LN is then a near-identity for everything we read.
    A = 1e4
    pos[:, C["anchor_hi"]], pos[:, C["anchor_lo"]] = A, -A
    sigma = A * math.sqrt(2 / model.d_model)
    for ln in [model.final_ln] + [b.ln1 for b in model.blocks] + [b.ln2 for b in model.blocks]:
        ln.weight.fill_(sigma)
    for W in [model.head.weight] + [getattr(b.attn, n).weight for b in model.blocks
                                    for n in ("W_q", "W_k", "W_v")]:
        W[:, C["zero"]] = -W.sum(dim=1)
    diagnostics(model, task)


def diagnostics(model: SimpleTransformer, task) -> None:
    """Sanity checks only (not used by the weights)."""
    acc, _ = evaluate(model, task, n_samples=5000, seed=7, device="cpu", verbose=False)
    print(f"diagnostic accuracy on 5000 seed-7 examples: {acc:.4f}")
    # Out-of-distribution lines: slopes 0-9, which the task itself never generates
    # beyond 1-5. Autoregressive generation, same as the evaluator's protocol.
    import random
    rng, hits, n = random.Random(3), 0, 2000
    by_slope: dict[int, list[int]] = {}
    for _ in range(n):
        m, b = rng.randint(0, 9), rng.randint(0, 9)
        x1, x2 = rng.sample(range(10), 2)
        x = rng.randint(0, 9)
        prompt = f"{x1},{m * x1 + b:02d};{x2},{m * x2 + b:02d};{x}="
        ids = torch.tensor([task.encode(prompt)])
        for _ in range(2):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
        ok = task.decode(ids[0, -2:].tolist()) == f"{m * x + b:02d}"
        hits += ok
        by_slope.setdefault(m, []).append(ok)
    print(f"OOD lines (m 0-9): {hits / n:.4f}",
          {m: round(sum(v) / len(v), 3) for m, v in sorted(by_slope.items())})


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "DefaultArch24CleanDig"
model_description = "DefaultArch23 with a dedicated dig channel for read values, because block 1 writes into out at '=' and broke block 2's zero-valued '=' volume knob."


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
