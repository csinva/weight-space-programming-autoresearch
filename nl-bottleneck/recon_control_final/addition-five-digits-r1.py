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
        # LayerNorm removed: the hand-written circuit keeps exact integer values in the residual.
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
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
        d_model: int = 40,
        n_heads: int = 5,
        n_layers: int = 2,
        d_ff: int = 32,
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
    """Closed-form carry-lookahead adder.

    Residual layout:
      0      : constant 1
      1      : digit value of current token (0 for '+', '=')
      2..19  : one-hot position
      20..24 : t_i = a_i + b_i (place i, 0 = least significant), written by layer-1 attention
      25..29 : c_j carry into place j (j=1..5), written by layer-1 MLP:
               c_j = [sum_{i<j} t_i 10^i >= 10^j]  (exact integer threshold, 2 ReLUs)
      30     : s = output digit at this position, written by layer-2 MLP:
               u_j = t_j + c_j selected by position, s = u_j - 10*[u_j >= 10]
    Readout: logit_d = 2 d s - d^2 (= -(s-d)^2 + const) -> argmax = s.
    Position p in 11..16 predicts output digit S_j with j = 16 - p.
    """
    with torch.no_grad():
        for p_ in model.parameters():
            p_.zero_()
        stoi = task.stoi
        CONST, DIG, POS, T, C, S = 0, 1, 2, 20, 25, 30
        for ch, idx in stoi.items():
            model.token_emb.weight[idx, CONST] = 1.0
            if ch.isdigit():
                model.token_emb.weight[idx, DIG] = float(int(ch))
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS + p] = 1.0

        # ---- layer 1 attention: head i averages digits at a_i (pos 4-i) and b_i (pos 10-i)
        blk = model.blocks[0]
        att = blk.attn
        dh = att.d_head
        for i in range(5):
            r = dh * i
            att.W_q.weight[r, CONST] = 1.0
            for p in range(model.max_seq_len):
                score = 0.0 if p in (4 - i, 10 - i) else -1e4
                att.W_k.weight[r, POS + p] = score * math.sqrt(dh)
            att.W_v.weight[r, DIG] = 1.0
            att.W_o.weight[T + i, r] = 2.0  # 2 * mean = a_i + b_i

        # ---- layer 1 MLP: carries c_j = ReLU(D_j - 10^j + 1) - ReLU(D_j - 10^j)
        fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
        h = 0
        for j in range(1, 6):
            for off, sign in ((1.0, 1.0), (0.0, -1.0)):
                for i in range(j):
                    fc1.weight[h, T + i] = float(10 ** i)
                fc1.bias[h] = off - float(10 ** j)
                fc2.weight[C + j - 1, h] = sign
                h += 1

        # ---- layer 2 MLP: gate u_j by position, then mod 10
        blk = model.blocks[1]
        fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
        M = 100.0
        h = 0
        for j in range(6):
            p = 16 - j
            for off, coef in ((0.0, 1.0), (-9.0, -10.0), (-10.0, 10.0)):
                if j <= 4:
                    fc1.weight[h, T + j] = 1.0
                if j >= 1:
                    fc1.weight[h, C + j - 1] = 1.0
                fc1.weight[h, POS + p] = M
                fc1.bias[h] = off - M
                fc2.weight[S, h] = coef
                h += 1

        # ---- readout
        for ch, idx in stoi.items():
            if ch.isdigit():
                d = float(int(ch))
                model.head.weight[idx, S] = 2 * d
                model.head.weight[idx, CONST] = -d * d
            else:
                model.head.weight[idx, CONST] = -1e4


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CarryLookaheadExact"
model_description = "2-layer no-LN transformer: heads gather a_i+b_i per place, MLP computes carries via exact integer threshold sum t_i 10^i >= 10^j, layer-2 MLP selects place by position and applies mod 10, quadratic readout."


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
