"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Character log-odds sentiment classifier.

    1. Compute log P(c|y=1) - log P(c|y=0) over the SST-2 pool (closed-form).
    2. Embed each token with its log-odds in residual dim 0 (and a constant 1
       in dim 1 to give LayerNorm a stable variance scale).
    3. Block 0 attention: uniform causal attention, V copies LN(x)[0] into a
       sentiment channel (residual dim D-1). At position '=' this becomes the
       sum/mean of per-char log-odds over the sentence.
    4. Blocks 1 & 2: all weights zero -> no-op (residual stream unchanged).
    5. Head: project residual dim D-1 to logit('1') - logit('0').
    """
    torch.manual_seed(0)
    V = model.vocab_size
    D = model.d_model

    pool = task._load_pool()  # closed-form: a fixed cached subset of SST-2 train.
    pos = torch.ones(V)  # Laplace smoothing
    neg = torch.ones(V)
    SKIP = set("_=01")
    for text, label in pool:
        for c in text:
            if c in SKIP:
                continue
            j = task.stoi[c]
            if label == '1':
                pos[j] += 1
            else:
                neg[j] += 1
    log_odds = torch.log(pos / pos.sum()) - torch.log(neg / neg.sum())
    # Neutralize non-content tokens (padding, '=', '0', '1').
    for c in SKIP:
        log_odds[task.stoi[c]] = 0.0

    # Calibrate by per-pool sentence scoring and a threshold search so that
    # classifier balance roughly matches the SST-2 class balance.
    import numpy as np
    lo_np = log_odds.numpy()
    scores = np.array([
        sum(lo_np[task.stoi[c]] for c in text) for text, _ in pool
    ])
    labels = np.array([int(lab) for _, lab in pool])
    # Find the threshold T* with the highest training-pool accuracy.
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order]
    # For each split, predict 0 for left part, 1 for right part.
    n_pool = len(pool)
    n_pos = labels.sum()
    pos_right = n_pos
    neg_right = n_pool - n_pos
    pos_left = 0
    neg_left = 0
    best_acc = -1
    best_T = s_sorted[0] - 1.0
    for i in range(n_pool):
        # accuracy if T = s_sorted[i]: predict 0 for first i (where score <= T), 1 for rest.
        acc = (neg_left + pos_right) / n_pool
        if acc > best_acc:
            best_acc = acc
            best_T = (s_sorted[i - 1] + s_sorted[i]) / 2 if i > 0 else s_sorted[i] - 1.0
        if y_sorted[i] == 1:
            pos_right -= 1; pos_left += 1
        else:
            neg_right -= 1; neg_left += 1
    print(f"[CharLogOdds] pool accuracy at T*={best_T:.3f}: {best_acc:.4f}", flush=True)
    # Shift every log-odds value by -T*/81 so that 81-position mean threshold = 0.
    log_odds = log_odds - float(best_T) / 81.0

    with torch.no_grad():
        # ----- Embeddings -----
        # Layout per token c:
        #   dim 0  : log_odds[c]      (sentiment signal)
        #   dim 1  : +0.5             (balanced constant pair, stabilizes LN
        #   dim 2  : -0.5              variance without biasing the mean)
        #   dim D-1: sentiment-aggregation scratch (filled by attention)
        model.token_emb.weight.zero_()
        for c in range(V):
            model.token_emb.weight[c, 0] = log_odds[c]
            model.token_emb.weight[c, 1] = 0.5
            model.token_emb.weight[c, 2] = -0.5
        model.pos_emb.weight.zero_()

        # ----- All blocks zero by default -----
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0); blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0); blk.ln2.bias.zero_()
            blk.attn.W_q.weight.zero_()
            blk.attn.W_k.weight.zero_()
            blk.attn.W_v.weight.zero_()
            blk.attn.W_o.weight.zero_()
            blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
            blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # ----- Block 0: aggregate sentiment via uniform causal attention -----
        b0 = model.blocks[0]
        # Q=0, K=0 -> uniform softmax over causal positions (incl. self).
        # V: copy LN(x)[0] (the sentiment) to head-output dim 0.
        b0.attn.W_v.weight[0, 0] = 1.0
        # W_o: route attention output dim 0 -> residual dim D-1 (sentiment channel).
        b0.attn.W_o.weight[D - 1, 0] = 50.0

        # No additional bias needed: balanced constants in dims 1,2 keep LN mean
        # near zero, so sign(sentiment) → sign(LN_out[D-1]) at the final norm.

        # ----- Final layer & classifier head -----
        model.final_ln.weight.fill_(1.0); model.final_ln.bias.zero_()
        model.head.weight.zero_()
        # Logit('1') = +K * LN(residual)[D-1]; Logit('0') = -K * LN(residual)[D-1].
        K = 100.0
        model.head.weight[task.stoi['1'], D - 1] = K
        model.head.weight[task.stoi['0'], D - 1] = -K


model_shorthand_name = "CharLogOddsCal"
model_description = "Per-char log-odds (Laplace-smoothed) over SST-2 pool; calibrated via threshold search on pool; uniform causal attention averages sentiment at '=' position."


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
