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
    """Block without LayerNorm so hand-set features stay exact."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# Residual stream layout
TOK = 0          # 0..10  token one-hot ('0'..'9', '=')
POS = 11         # 11..26 position one-hot (16 positions)
QD = 27          # 27..36 query digit one-hot (copied from position 10)
HIST = 37        # 37..46 histogram of digits over positions 0..9 (fraction)
LOG = 47         # 47..57 output logits (one per vocab entry)
D_MODEL = 64
KS = list(range(-1, 12))   # ramp offsets for the count
D_FF = 2 * 10 * len(KS)    # 260


class SimpleTransformer(nn.Module):
    """2-layer causal transformer without LayerNorm."""

    def __init__(self, vocab_size: int, max_seq_len: int = 16, d_model: int = D_MODEL,
                 n_heads: int = 1, n_layers: int = 2, d_ff: int = D_FF):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

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

@torch.no_grad()
def write_weights(model: SimpleTransformer, task) -> None:
    for p in model.parameters():
        p.zero_()
    D = model.d_model
    V = task.vocab_size                      # 11: '0'..'9', '='
    S = 100.0 * math.sqrt(D)                 # attention sharpness (score gap 100 after 1/sqrt(dh))
    M = 100.0                                # gating constant in MLP

    # Embeddings: token one-hot and position one-hot
    for t in range(V):
        model.token_emb.weight[t, TOK + t] = 1.0
    for p in range(model.max_seq_len):
        model.pos_emb.weight[p, POS + p] = 1.0

    # Layer 1 attention: positions 11, 12 attend to position 10, copy its digit into QD
    a1 = model.blocks[0].attn
    for p in (11, 12):
        a1.W_q.weight[0, POS + p] = S
    a1.W_k.weight[0, POS + 10] = 1.0
    for d in range(10):
        a1.W_v.weight[QD + d, TOK + d] = 1.0
    a1.W_o.weight.copy_(torch.eye(D))
    # layer 1 MLP stays zero

    # Layer 2 attention: positions 11, 12 attend uniformly over digit positions 0..9,
    # averaging token one-hots -> HIST_d = count_d / 10
    a2 = model.blocks[1].attn
    for p in (11, 12):
        a2.W_q.weight[0, POS + p] = S
    for p in range(10):
        a2.W_k.weight[0, POS + p] = 1.0
    for d in range(10):
        a2.W_v.weight[HIST + d, TOK + d] = 1.0
    a2.W_o.weight.copy_(torch.eye(D))

    # Layer 2 MLP: unit (g, d, k) = relu(10*HIST_d - k + M(QD_d - 1) + M(P_g - 1))
    #   = relu(c - k) if d is the query digit and we are at position g, else 0.
    # Indicator[c == k] = r(k-1) - 2 r(k) + r(k+1)  (exact for integer c).
    # Position 11 emits the tens digit ('1' iff c == 10); position 12 emits c % 10.
    m = model.blocks[1].mlp
    idx = {}
    u = 0
    for g in (11, 12):
        for d in range(10):
            for k in KS:
                m.fc1.weight[u, HIST + d] = 10.0
                m.fc1.weight[u, QD + d] = M
                m.fc1.weight[u, POS + g] = M
                m.fc1.bias[u] = -k - 2 * M
                idx[(g, d, k)] = u
                u += 1
    assert u == m.fc1.out_features

    def target(g, c):
        if g == 11:
            return 1 if c == 10 else 0
        return c % 10

    for g in (11, 12):
        for d in range(10):
            for c in range(0, 11):
                out = LOG + target(g, c)
                m.fc2.weight[out, idx[(g, d, c - 1)]] += 1.0
                m.fc2.weight[out, idx[(g, d, c)]] += -2.0
                m.fc2.weight[out, idx[(g, d, c + 1)]] += 1.0

    # Head: read logit dims, scaled for a clear margin
    for t in range(V):
        model.head.weight[t, LOG + t] = 10.0


model_shorthand_name = "QueryCopyHistRamp"
model_description = ("2-layer no-LN: L1 attn copies query digit to answer positions; L2 attn averages digit "
                     "one-hots over positions 0-9; L2 MLP selects query bin and decodes count via relu ramps gated by position.")


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
