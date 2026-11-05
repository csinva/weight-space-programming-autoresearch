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
        d_model: int = 33,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 9,
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
    """Sort = k-th smallest digit at answer step k.

    Residual layout: tok one-hot [0:11], K (k+1 at answer positions) [11], const [12],
    bag-of-digit counts [13:23], logit scores [23:33].
    Attention (1 head): answer positions average the 5 prompt digits -> counts (K masks answer keys).
    MLP: output k is the smallest digit d with cum_le(d) > k (cum_le = #digits <= d);
    unit d = relu(cum_le(d) - k), weighted 6^(9-d) into logit d. Digit 9 is the default (const baseline).
    """
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        D = model.d_model
        # layout: tok one-hot [0:11], K (= k+1 at answer positions, 0 in prompt), const, bag [13:23], logits [23:33]
        TOK, K, C, BAG, LOG = 0, 11, 12, 13, 23
        T = P = Cv = 10.0
        m1 = (T + Cv) / D  # LN stats at prompt positions (K = 0)
        s1 = math.sqrt((T * T + Cv * Cv) / D - m1 * m1 + 1e-5)
        c0 = T + Cv + 5  # LN mean numerator at answer positions, excluding the K term (counts sum to 5)

        for t in range(11):
            model.token_emb.weight[t, TOK + t] = T
        for p in range(model.pos_emb.num_embeddings):
            model.pos_emb.weight[p, C] = Cv
            if p >= 5:  # answer step k = p-5, stored as k+1
                model.pos_emb.weight[p, K] = P * (p - 4)

        blk = model.blocks[0]
        at = blk.attn
        # query reads const; key penalises K>0 (answer positions) -> uniform over prompt digits
        at.W_q.weight[0, C] = 1.0
        at.W_k.weight[0, K] = -30.0
        g1 = s1 / T
        h = m1 * s1 / (T * (Cv - m1))  # cancels the LN mean offset so value = exact one-hot
        for e in range(10):
            at.W_v.weight[BAG + e, TOK + e] = g1
            at.W_v.weight[BAG + e, C] = h
            at.W_o.weight[BAG + e, BAG + e] = 5.0  # mean over 5 digits -> count

        # MLP unit per digit d (type B): relu(cum_le(d) - k) > 0 for every d >= answer;
        # geometric output weights make the smallest such d win the argmax.
        specs = (("B", 0.0, 1.0),)  # ramp only: relu(cum_le(d)-k), no step
        mlp = blk.mlp
        u = 0
        for d in range(9):
            for typ, delta, out_sign in specs:
                out_sign = out_sign * 6.0 ** (9 - d)  # geometric weights: smallest qualifying d dominates
                sign = 1.0 if typ == "A" else -1.0  # A: k - cum_lt(d); B: cum_le(d) - k - 1
                w = torch.zeros(D)
                for e in range(10):
                    if typ == "A" and e < d:
                        w[BAG + e] = -1.0
                    if typ == "B" and e <= d:
                        w[BAG + e] = 1.0
                # LN mean m = (c0 + P*k')/D depends on k' = k+1, so solve for w[K], w[C] jointly:
                #   w_K - S/D = sign/P ;  w_C*Cv - c0*S/D = delta - sign   (S = sum of all weights)
                d_eff = delta - sign
                S = (w.sum() + sign / P + d_eff / Cv) / (1 - 1 / D - c0 / (D * Cv))
                w[K] = sign / P + S / D
                w[C] = (d_eff + c0 * S / D) / Cv
                mlp.fc1.weight[u] = w
                mlp.fc2.weight[LOG + d, u] = out_sign
                u += 1
        for d in range(10):
            model.head.weight[d, LOG + d] = 1.0
        model.head.weight[9, C] = 0.05  # digit 9 has no units: constant baseline above all "false" digits
        model.head.weight[10, C] = -1.0  # '=' is never an answer: keep its logit very low


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "final_ff9_geom6"
model_description = "1 layer/1 head sort: attention averages prompt digits into counts; 9 MLP ramp units relu(cum_le(d)-k) weighted 6^(9-d) so the smallest digit with cum_le(d)>k wins (9 is a const-baseline default); d_model=33."


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
