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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attn:
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
        d_ff=256,
        use_attn=True,
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
        ffs = d_ff if isinstance(d_ff, (list, tuple)) else [d_ff] * n_layers
        attn_flags = use_attn if isinstance(use_attn, (list, tuple)) else [use_attn] * n_layers
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffs[i], attn_flags[i]) for i in range(n_layers)])
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
    """Long-division circuit.  Position 3+j predicts bit j (MSB first).

    Attention (10 heads, head p fixed on source position p) sums
        R = 100a + 10b + c - sum_{i<j} 2^(7-i) * bit_i      (scaled by eps)
    i.e. the remainder r_j of N after removing the already emitted bits.  Heads whose
    source bit has not been emitted yet attend to the '=' sink (position 3, value 0).
    MLP: unit j (gated to position 3+j) fires iff r_j >= 2^(7-j), writing the '1' logit;
    '0' wins by a constant default logit (exactly one of tokens '0','1','=' is active at each
    answer position).

    Residual (d_model=22): 0-10 token one-hot | 11-21 position one-hot.
    R reuses token dim 9 and the '1' logit scratch reuses token dim 3 (both unused after the
    prompt digits).  Weight scales were calibrated by hand-evaluating prompts (no training).
    """
    TOK, POS, R, LOGIT = 0, 11, 9, 2  # R reuses token dim '9' (zero after the prompt)
    D = model.d_model
    dh = D // model.n_heads
    stoi = task.stoi
    # eps: scale of R (small so LayerNorm's scale barely depends on R); K: attention sharpness; G: position gate
    eps, K, G = 0.31 / 100, 30.0, 4.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        for t in range(task.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p_ in range(11):
            model.pos_emb.weight[p_, POS + p_] = 1.0
        NONDIGIT = [stoi["0"], stoi["1"], stoi["="]]  # exactly one is active at every position >= 3

        att = model.blocks[0].attn
        src_pos = [0, 1, 2] + [4 + i for i in range(7)]
        coef = [100, 10, 1] + [-(2 ** (7 - i)) for i in range(7)]
        for h, (sp, c) in enumerate(zip(src_pos, coef)):
            for t in NONDIGIT:
                att.W_q.weight[h * dh, TOK + t] = 1.0
            att.W_k.weight[h * dh, POS + sp] = K
            if sp > 3:
                att.W_k.weight[h * dh, POS + 3] = K / 4  # sink while source not yet emitted
            if sp < 3:   # digit head: value = digit * coef (read off the token one-hot)
                for d in range(10):
                    att.W_v.weight[h * dh + 1, TOK + stoi[str(d)]] = eps * c * d
            else:        # bit head: only token '1' contributes (sink '=' acts like bit 0)
                att.W_v.weight[h * dh + 1, TOK + stoi["1"]] = eps * c
            att.W_o.weight[R, h * dh + 1] = 1.0

        mlp = model.blocks[0].mlp
        for j in range(8):
            mlp.fc1.weight[j, R] = 1.0     # unit j: fires iff position==3+j and r >= 2^(7-j)
            mlp.fc1.weight[j, POS + 3 + j] = G
            mlp.fc2.weight[LOGIT + 1, j] = 1.0

        def pre_at(n, j):
            s = "%03d=" % n + "0" * j
            ids = torch.tensor([[stoi[ch] for ch in s]])
            h = model.token_emb(ids) + model.pos_emb(torch.arange(ids.shape[1]))[None]
            h = h + att(model.blocks[0].ln1(h))
            return mlp.fc1.weight @ F.layer_norm(h, (D,))[0, -1]

        # calibrate: remainder exactly 2^(7-j) (bit=1) vs 2^(7-j)-1 (bit=0) at position 3+j
        for j in range(8):
            hi, lo = pre_at(2 ** (7 - j), j), pre_at(2 ** (7 - j) - 1, j)
            mlp.fc1.bias[j] = -(hi[j] + lo[j]) / 2
            mlp.fc2.weight[:, j] /= (hi[j] - lo[j]) / 2

        model.head.weight[stoi["1"], LOGIT + 1] = 1.0
        for t in NONDIGIT:
            model.head.weight[stoi["0"], TOK + t] = 0.5   # constant default logit for '0'


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "remainder_d22_clean_final"
model_description = "Long division, 1 layer, d_model 22, 11 heads (10 used), no constant dims: heads sum 100a+10b+c minus emitted bits*2^k into R; 8 position-gated threshold units test r>=2^(7-j); '0' is a constant default logit."


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
                              d_model=22, n_heads=11, n_layers=1, d_ff=8)
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
