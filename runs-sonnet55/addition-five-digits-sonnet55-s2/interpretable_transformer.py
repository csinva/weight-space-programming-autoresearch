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
    """Populate `model`'s parameters in-place. No training allowed.

    Hand-written 2-layer adder (answer is written most-significant digit first).

    Sequence: a0..a4 '+' b0..b4 '=' r0..r5 (r0 = carry-out digit, r1..r5 = digits).
    Residual stream is split into named subspaces (balanced one-hot codes so
    LayerNorm only rescales):
      TOK token one-hot, POS position one-hot, AIN a-digit fetched to b_j,
      S one-hot of s_j=a_j+b_j (0..18), G flag s_j>=10, NINE flag s_j==9,
      SSEL one-hot of s_i for the output digit, CAR carry flag into that digit,
      OUT one-hot of the output digit.
    Layer 1: attention at b_j fetches a_j (position p-6); MLP = 10x10 lookup of a_j+b_j.
    Layer 2: head A fetches own s_i (key p-6; '+' gives s=0 for the carry digit);
             head B fetches carry = G flag of the nearest non-9 lower position
             (sink '=' -> 0); MLP = lookup (s, carry) -> (s+carry) mod 10.
    """
    D = model.d_model
    dh = D // model.n_heads
    NP = model.max_seq_len
    TOK, POS = 0, 12
    AIN, S, G, NINE = 30, 40, 59, 60
    SSEL, CAR, OUT = 62, 81, 82
    PLUS = task.stoi["+"]
    SHARP, NINE_PEN, TH1 = 4.0, 40.0, 1.3
    STEP = 4.0  # score drop per digit of distance in the carry head
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(12):
            model.token_emb.weight[t, TOK + t] = 1.0
            model.token_emb.weight[t, TOK:TOK + 12] -= 1 / 12
        for p in range(NP):
            model.pos_emb.weight[p, POS + p] = 1.0
            model.pos_emb.weight[p, POS:POS + NP] -= 1 / NP
        for blk in list(model.blocks) + []:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        def nrm2(*blocks):  # squared norm of a sum of balanced one-hots
            return sum((1 - 1 / n) ** 2 + (n - 1) / n ** 2 for n in blocks)
        s0 = math.sqrt(D / nrm2(12, NP))
        s1 = math.sqrt(D / nrm2(12, NP, 10))
        s2 = math.sqrt(D / (nrm2(12, NP, 10, 19) + 1))
        s3 = math.sqrt(D / (nrm2(12, NP, 10, 19, 19) + 2))

        def set_pos_query(attn, head, E):
            # E: dict (p,q)->score; column sums fixed to zero via row 0 (LayerNorm centering)
            R = torch.zeros(NP, NP)
            for (p, q), v in E.items():
                R[p, q] = v
            R[0] -= R.sum(0)
            for q in range(NP):
                attn.W_q.weight[head * dh + q, POS:POS + NP] = R[:, q]
                attn.W_k.weight[head * dh + q, POS + q] = 1.0

        # ---------------- layer 1 ----------------
        blk = model.blocks[0]
        set_pos_query(blk.attn, 0, {(p, p - 6): SHARP for p in range(6, 11)})
        for d in range(10):
            blk.attn.W_v.weight[d, TOK + d] = 1 / s0
            blk.attn.W_o.weight[AIN + d, d] = 1.0
        u = 0
        for a in range(10):
            for b in range(10):
                blk.mlp.fc1.weight[u, AIN + a] = 1.0
                blk.mlp.fc1.weight[u, TOK + b] = 1.0
                blk.mlp.fc1.bias[u] = -TH1 * s1
                g = 1 / (0.5 * s1)
                v = a + b
                blk.mlp.fc2.weight[S:S + 19, u] = -g / 19
                blk.mlp.fc2.weight[S + v, u] += g
                if v >= 10:
                    blk.mlp.fc2.weight[G, u] = g
                if v == 9:
                    blk.mlp.fc2.weight[NINE, u] = g
                u += 1
        # '+' token carries s=0 (the "no digit" source for the carry-out step)
        blk.mlp.fc1.weight[u, TOK + PLUS] = 1.0
        blk.mlp.fc1.bias[u] = -0.5 * s0
        g = 1 / (0.4 * s0)
        blk.mlp.fc2.weight[S:S + 19, u] = -g / 19
        blk.mlp.fc2.weight[S, u] += g

        # ---------------- layer 2 ----------------
        blk = model.blocks[1]
        # head A: own digit sum
        set_pos_query(blk.attn, 0, {(p, p - 6): SHARP for p in range(11, 17)})
        for d in range(19):
            blk.attn.W_v.weight[d, S + d] = 1 / s2
            blk.attn.W_o.weight[SSEL + d, d] = 1.0
        # head B: carry from nearest non-9 position j>=k (keys 6+j), sink at '=' (pos 11)
        E = {}
        for p in range(11, 17):
            k = p - 11
            for q in range(NP):
                E[(p, q)] = -12.0
            E[(p, 11)] = 2.0
            for j in range(k, 5):
                E[(p, 6 + j)] = 20.0 - STEP * (j - k)
        set_pos_query(blk.attn, 1, E)
        c = 1 * dh + 18
        for p in range(11, 17):
            blk.attn.W_q.weight[c, POS + p] = -NINE_PEN
        blk.attn.W_k.weight[c, NINE] = 1.0
        blk.attn.W_v.weight[dh, G] = 1 / s2
        blk.attn.W_o.weight[CAR, dh] = 1.0
        # MLP: (s, carry) -> (s + carry) mod 10
        u = 0
        for sv in range(19):
            for cy in range(2):
                blk.mlp.fc1.weight[u, SSEL + sv] = 1.0
                blk.mlp.fc1.weight[u, CAR] = 1.0 if cy else -1.0
                blk.mlp.fc1.bias[u] = -(1.5 if cy else 0.45) * s3
                g = 0.5
                blk.mlp.fc2.weight[OUT:OUT + 10, u] = -g / 10
                blk.mlp.fc2.weight[OUT + (sv + cy) % 10, u] += g
                u += 1
        for d in range(10):
            model.head.weight[d, OUT + d] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Lookahead_canonical_v2"
model_description = "Canonical d92/h2/ff101 2-layer lookahead-carry adder: SHARP=4, NINE_PEN=40, TH1=1.3, STEP=4 (cleaned constants)."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len,
                              d_model=92, n_heads=2, n_layers=2, d_ff=101)
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
