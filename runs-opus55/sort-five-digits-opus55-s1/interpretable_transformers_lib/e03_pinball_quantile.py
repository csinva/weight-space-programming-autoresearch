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
        d_model: int = 11,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 1,
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
    """Sorting as quantile estimation: the k-th smallest of x_1..x_5 maximises
        score_k(d) = -1/2 sum_i |x_i - d| + (k - 2) * d      (negative pinball loss)
    which is concave in d with half-integer slopes, so the maximiser is unique.

    Channels: 0-9 = one per digit d, S = 10 = signed position channel.
      token_emb  eps * e_digit ('=' is the zero vector)
      pos_emb    S = +1 on the prompt, S = t - 10 on output step t (always negative)
      ln1        constant mean/std on the prompt, so its affine terms undo the
                 normalisation: prompt tokens come out as exactly e_digit + e_S
      W_q, W_k   S -> S with opposite signs: output positions attend only to the
                 prompt (uniformly, because all prompt positions have S = 1)
      W_v        digit x -> the column -|x - d| / 2 (centred across d)
      W_o        5 * digits (mean -> sum); 3 * S (so h_S = t - 10 + 3 = k - 2)
      head       digit channels pass through; S multiplies the tilt d - 4.5.
                 Rows sum to zero, which makes final_ln's mean subtraction harmless;
                 logits are centred, so the zero '=' logit can never win.
    The residual leaks eps * e_previous_digit into the logits, so eps < 1/2 keeps it
    below the smallest score gap. The MLP is zeroed: no non-linearity is needed.
    """
    torch.manual_seed(0)
    D, S, eps, sharp = model.d_model, 10, 0.25, 10.0
    d = torch.arange(10.0)
    block = model.blocks[0]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.token_emb.weight[:10, :10] = eps * torch.eye(10)
        model.pos_emb.weight[:5, S] = 1.0
        model.pos_emb.weight[5:, S] = torch.arange(5, model.max_seq_len) - 10.0

        m = (1 + eps) / D
        s = ((1 + eps ** 2) / D - m ** 2) ** 0.5
        block.ln1.weight[:10], block.ln1.bias[:10] = s / eps, m / eps
        block.ln1.weight[S], block.ln1.bias[S] = s, m
        block.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        attn = block.attn
        attn.W_q.weight[S, S] = -sharp
        attn.W_k.weight[S, S] = sharp
        V = -0.5 * (d[:, None] - d[None, :]).abs()     # V[j, x]: channel j, token x
        attn.W_v.weight[:10, :10] = V - V.mean(dim=0, keepdim=True)
        attn.W_v.weight[S, S] = 1.0
        attn.W_o.weight[:10, :10] = 5 * torch.eye(10)
        attn.W_o.weight[S, S] = 3.0

        u = d - 4.5
        model.head.weight[:10, :10] = torch.eye(10) - (1 + u)[:, None] / 10
        model.head.weight[:10, S] = u
    check_all(model, task)


def check_all(model, task) -> None:
    """Diagnostic only: run generation on all 100000 inputs and report accuracy and margin."""
    n = task.prompt_len - 1
    digits = torch.cartesian_prod(*[torch.arange(10)] * n)
    ids = torch.cat([digits, torch.full((len(digits), 1), task.stoi["="])], dim=1)
    margin = float("inf")
    with torch.no_grad():
        for _ in range(task.answer_len):
            logits = model(ids)[:, -1]
            top = logits.topk(2, dim=-1).values
            margin = min(margin, (top[:, 0] - top[:, 1]).min().item())
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
    acc = (ids[:, -task.answer_len:] == digits.sort(dim=1).values).all(dim=1).float().mean().item()
    print(f"EXHAUSTIVE accuracy={acc:.6f} min_logit_margin={margin:.4f}")


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "e03_pinball_quantile"
model_description = "1 block, d_model=11, no MLP use: uniform attention over the prompt plus a position tilt gives the negative pinball loss; its argmax is the k-th smallest digit."


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
