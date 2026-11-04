"""
Interpretable transformer for MRPC paraphrase detection, weights written by hand.

Prompt: "<s1, 60 chars>|<s2, 60 chars>=" -> answer "1" (paraphrase) or "0".

Circuit (every Linear is a sparse, readable wiring matrix; nothing is trained):
  * Registers: named slices of the d_model state (see class R).
  * L1 "window": heads look k characters back using sinusoidal position keys and
    copy that character's 6-bit code, so each position holds its last K chars.
  * L2 "match": each position queries earlier positions with its shingle. A
    null attention slot sits just below an exact match, so attention leaves the
    null slot only for an exact n-gram match located in s1.
  * L3 "pool": the '=' position averages match bits over non-pad s2 positions.
  * Head: linear threshold over the pooled overlap fractions.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
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
# Hand-set experiment configuration
# ---------------------------------------------------------------------------

NGRAMS = (3, 4, 6)              # shingle length of each L2 match head
FEATURE_WEIGHTS = (0.0, 1.0, 0.0)  # weight of each pooled overlap fraction
THRESHOLD = 0.35                # answer "1" iff weighted overlap > THRESHOLD


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class R:
    """Register map: which slice of the d_model state holds what."""
    BITS = 6                    # bits in a character code
    K = 6                       # characters of history held by each position
    WIN = 0                     # [0, 36): codes of chars at offsets 0..K-1
    POS = 36                    # [36, 38): (cos, sin) of position
    IS_S1 = 38                  # position lies in sentence 1
    IS_S2 = 39                  # position lies in sentence 2
    NOTPAD = 40                 # character is not the '_' pad
    BIAS = 41                   # constant 1
    MATCH = 42                  # [42, 45): per-position match bit, one per n-gram
    POOLED = 45                 # [45, 48): overlap fraction, one per n-gram
    D = 120


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, null_slot: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # Null slot: one extra attention target per head with a fixed score and
        # zero value, so "attend to nothing" is an explicit option.
        self.null_score = nn.Parameter(torch.zeros(n_heads)) if null_slot else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        if self.null_score is not None:
            null = self.null_score.view(1, H, 1, 1).expand(B, H, T, 1)
            attn = torch.cat([scores, null], dim=-1).softmax(dim=-1)[..., :-1]
        else:
            attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class Block(nn.Module):
    """Pre-norm-free block: attention writes into empty registers via the residual."""

    def __init__(self, d_model: int, n_heads: int, null_slot: bool = False):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, null_slot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class SimpleTransformer(nn.Module):
    """Three hand-wired attention layers: window, match, pool."""

    def __init__(self, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        D = R.D
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([
            Block(D, 20),                  # L1 window: 6-dim heads
            Block(D, 3, null_slot=True),   # L2 match: 40-dim heads
            Block(D, 12),                  # L3 pool: 10-dim heads
        ])
        self.head = nn.Linear(D, vocab_size, bias=False)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(ids))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def char_codes(task) -> torch.Tensor:
    """+-1 binary code of each vocabulary index; the pad is the zero vector."""
    codes = torch.zeros(task.vocab_size, R.BITS)
    for i in range(task.vocab_size):
        for b in range(R.BITS):
            codes[i, b] = 1.0 if (i >> b) & 1 else -1.0
    codes[task.stoi["_"]] = 0.0
    return codes


@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place from constants. No training."""
    for p in model.parameters():
        p.zero_()

    # Embeddings: char code + notpad; position sinusoid + sentence flags.
    period = 256.0
    theta = 2 * math.pi / period
    te, pe = model.token_emb.weight, model.pos_emb.weight
    te[:, R.WIN:R.WIN + R.BITS] = char_codes(task)
    te[:, R.NOTPAD] = 1.0
    te[task.stoi["_"], R.NOTPAD] = 0.0
    sent = task.SENT_LEN
    for p in range(model.max_seq_len):
        pe[p, R.POS] = math.cos(theta * p)
        pe[p, R.POS + 1] = math.sin(theta * p)
        pe[p, R.IS_S1] = float(p < sent)
        pe[p, R.IS_S2] = float(sent < p < 2 * sent + 1)
        pe[p, R.BIAS] = 1.0

    # L1 window: head k-1 finds position t-k by rotating its position query.
    a = model.blocks[0].attn
    dh = a.d_head
    g = math.sqrt(1e5 * math.sqrt(dh))
    for k in range(1, R.K):
        h = (k - 1) * dh
        c, s = math.cos(k * theta), math.sin(k * theta)
        a.W_q.weight[h, R.POS], a.W_q.weight[h, R.POS + 1] = g * c, g * s
        a.W_q.weight[h + 1, R.POS], a.W_q.weight[h + 1, R.POS + 1] = -g * s, g * c
        a.W_k.weight[h, R.POS] = g
        a.W_k.weight[h + 1, R.POS + 1] = g
        for b in range(R.BITS):
            a.W_v.weight[h + b, R.WIN + b] = 1.0
            a.W_o.weight[R.WIN + k * R.BITS + b, h + b] = 1.0

    # L2 match: dot product of shingles counts agreeing bits; only an exact
    # n-gram (score = n * BITS) beats the null slot. s2 positions are vetoed.
    a = model.blocks[1].attn
    dh = a.d_head
    sharp = 10.0
    g = math.sqrt(sharp * math.sqrt(dh))
    for i, n in enumerate(NGRAMS):
        h = i * dh
        for j in range(n * R.BITS):
            a.W_q.weight[h + j, R.WIN + j] = g
            a.W_k.weight[h + j, R.WIN + j] = g
        a.W_q.weight[h + 36, R.BIAS] = g
        a.W_k.weight[h + 36, R.IS_S2] = -100.0 * g
        a.W_v.weight[h, R.IS_S1] = 1.0
        a.W_o.weight[R.MATCH + i, h] = 1.0
        a.null_score[i] = sharp * (n * R.BITS - 1)

    # L3 pool: uniform attention over non-pad s2 positions, reading match bits.
    a = model.blocks[2].attn
    dh = a.d_head
    g = math.sqrt(30.0 * math.sqrt(dh))
    for i in range(len(NGRAMS)):
        h = i * dh
        a.W_q.weight[h, R.BIAS] = g
        a.W_k.weight[h, R.IS_S2] = g
        a.W_k.weight[h, R.NOTPAD] = g
        a.W_v.weight[h, R.MATCH + i] = 1.0
        a.W_o.weight[R.POOLED + i, h] = 1.0

    # Head: answer "1" iff the weighted overlap clears the threshold.
    w = model.head.weight
    w[:, R.BIAS] = -100.0
    w[task.stoi["0"], R.BIAS] = 0.0
    w[task.stoi["1"], R.BIAS] = -20.0 * THRESHOLD
    for i, fw in enumerate(FEATURE_WEIGHTS):
        w[task.stoi["1"], R.POOLED + i] = 20.0 * fw

    dev_report(model, task)


@torch.no_grad()
def dev_report(model: SimpleTransformer, task) -> None:
    """Diagnostics only: prints numbers for the human-style reading of results.

    Nothing here feeds back into the weights. Prompts of the evaluation draw
    (seed 0) are removed unprinted, so tuning decisions only ever see dev data.
    """
    held = {e.prompt for e in task.generate_examples(200, seed=0)}
    seen, dev = set(), []
    for e in task.generate_examples(2000, seed=12345):
        if e.prompt not in held and e.prompt not in seen:
            seen.add(e.prompt)
            dev.append(e)
    ids = torch.tensor([task.encode(e.prompt) for e in dev])
    y = torch.tensor([int(e.answer) for e in dev])
    h = torch.cat([model.features(b)[:, -1] for b in ids.split(256)])
    pred = model.head(h).argmax(-1)
    one = task.stoi["1"]
    acc = ((pred == one).long() == y).float().mean().item()
    print(f"DEV n={len(dev)} pos_rate={y.float().mean():.4f} acc={acc:.4f} "
          f"pred_pos={(pred == one).float().mean():.4f}")
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    for i, n in enumerate(NGRAMS):
        f = h[:, R.POOLED + i]
        print(f"DEV n={n} q|y=0 {torch.quantile(f[y == 0], qs).numpy().round(3)} "
              f"q|y=1 {torch.quantile(f[y == 1], qs).numpy().round(3)}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "E04_FourGramOverlap"
model_description = "Fraction of s2 4-grams found exactly in s1 via attention, thresholded at 0.35."


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
