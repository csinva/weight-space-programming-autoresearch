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
        d_model: int = 27,
        n_heads: int = 1,
        n_layers: int = 2,
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
    """Populate `model`'s parameters in-place. No training allowed.

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Task: 10 digits + query digit q + '=' -> two-digit count of q among the 10 digits.
    # (The attention also counts the query itself, so it sees c+1 matches.)
    # Residual layout (d=27): 0/1 = +M/-M constant pair (keeps LayerNorm scale ~constant),
    #   2..11 token one-hot, 12 'is data position (0..10)', 13/14 'is pos 10' / 'is pos 11',
    #   15..24 query digit (copied by L1), 25 F = match fraction (L2 attention),
    #   26 balance dim (keeps every residual vector zero-mean so LayerNorm is a pure rescale).
    d = model.d_model
    for prm in model.parameters():
        prm.data.zero_()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
    M = 30.0
    sig = math.sqrt(d) / math.sqrt(2 * M * M + 8)  # LayerNorm scale (constant pair dominates norm)
    T, ISD, P10, P11, QD, F, BAL = 2, 12, 13, 14, 15, 25, 26
    stoi = task.stoi
    dig = [stoi[str(j)] for j in range(10)]
    # --- embeddings
    for j in range(10):
        model.token_emb.weight.data[dig[j], T + j] = 1.0
        model.token_emb.weight.data[dig[j], BAL] = -1.0
    model.token_emb.weight.data[stoi["="], P11] = 1.0  # '=' token only occurs at position 11
    model.token_emb.weight.data[stoi["="], BAL] = -1.0
    pe = model.pos_emb.weight.data
    for p in range(pe.shape[0]):
        pe[p, 0], pe[p, 1] = M, -M
        if p <= 12:
            bal = 0.0
            if p == 10:
                pe[p, P10] = 1.0
                bal += 1.0
            if p <= 10:
                pe[p, ISD] = 1.0
                bal += 1.0
            if p == 12:  # first answer token is always '0': cancel it so position 12 is a pure sink
                pe[p, T + 0] = -1.0
                bal -= 1.0
            pe[p, BAL] = -bal
    # --- layer 1: every position copies the query digit (position 10) into QD
    a1 = model.blocks[0].attn
    a1.W_q.weight.data[0, 0] = 15 * math.sqrt(d) / (M * sig * sig)
    a1.W_k.weight.data[0, P10] = 1.0
    for j in range(10):
        a1.W_v.weight.data[QD + j, T + j] = 1 / sig
        a1.W_o.weight.data[QD + j, QD + j] = 1.0
        a1.W_o.weight.data[BAL, QD + j] = -1.0
    # --- layer 2: attention; nonmatching digits score -30, matches ln(1/3), sinks (pos 11,12) 0
    a2 = model.blocks[1].attn
    Q, w = 10.0, 1 / 12
    P = Q + math.log(w)
    amp = math.sqrt(P * math.sqrt(d)) / sig
    for j in range(10):
        a2.W_q.weight.data[j, QD + j] = amp
        a2.W_k.weight.data[j, T + j] = amp
    a2.W_q.weight.data[10, 0] = 1.0
    a2.W_k.weight.data[10, ISD] = -Q * math.sqrt(d) / (M * sig * sig)
    a2.W_v.weight.data[F, ISD] = 1 / sig
    a2.W_o.weight.data[F, F] = 1.0
    a2.W_o.weight.data[BAL, F] = -1.0
    # F = c'*w/(c'*w+2), c' = count+1 (two sinks of weight 1: positions 11 and 12).
    # No MLP needed: head logit_d = gam*(2 th_d F - th_d^2) = gam*(F^2 - (F-th_d)^2): lines tangent
    # to a parabola, so argmax_d picks the digit whose theta_d is nearest to F.
    gam = 1000.0
    hw = model.head.weight.data
    for dd in range(10):
        th = (dd + 1) * w / ((dd + 1) * w + 2)
        hw[dig[dd], F] = gam * 2 * th / sig
        hw[dig[dd], 0] = (100 - gam * th * th) / (M * sig)
    # position 11 is forced to output '0'
    hw[dig[0], P11] = 1000 / sig


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "FinalCountingAttn"
model_description = "d=27, 2 layers, 1 head, no MLP: L1 copies query digit (pos 10); L2 attention softmax mass F on matching digits (vs 2 sink positions) encodes count; head logits are parabola-tangent lines of F; '=' token forces first digit 0."


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
