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

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Prompt "x1,y1;x2,y2;q=" (y's two digits), answer = y1 + a*(q-x1), a=(y2-y1)/(x2-x1) in 1..5.
    d = model.d_model
    H, dh = model.n_heads, model.d_model // model.n_heads
    B = 1e4         # huge constant (+B,-B) pair: makes LayerNorm a fixed rescale
    std = B * math.sqrt(2.0 / d)

    # ---- residual stream layout -------------------------------------------
    names = {}
    def alloc(name, n=1):
        i = len(names_list)
        for k in range(n):
            names_list.append(f"{name}{k}" if n > 1 else name)
        return i
    names_list = []
    CP = alloc("constP"); CM = alloc("constM")
    NUM = alloc("num")                    # numeric value of a digit token
    POS = alloc("pos", 13)               # one-hot position
    BAL = alloc("balance")                # keeps the sum of the stream at 0 (so LN mean = 0)
    G = alloc("gathered", 7)              # x1, y1tens, y1ones, x2, y2tens, y2ones, q
    IA = alloc("slope_is", 6)             # IA+a = [slope == a]
    VS = alloc("vsum")                    # sum_a a*I_a*(q-x1)
    TF = alloc("tens_full", 6)            # one-hot of tens digit of answer
    L = alloc("logit", 10)                # digit logits
    assert len(names_list) <= d, len(names_list)
    X1, Y1T, Y1O, X2, Y2T, Y2O, Q = [G + i for i in range(7)]
    src_pos = [0, 2, 3, 5, 7, 8, 10]      # prompt positions of the 7 gathered tokens

    for p in model.parameters():
        p.data.zero_()
    def bal_vec(w):                       # w: tensor [..., d]; set BAL entry so the sum is 0
        w[..., BAL] = 0
        w[..., BAL] = -w.sum(-1)

    # ---- embeddings ---------------------------------------------------------
    te = model.token_emb.weight.data
    for tok, i in task.stoi.items():
        te[i, CP] = B; te[i, CM] = -B
        if tok.isdigit():
            te[i, NUM] = int(tok)
    pe = model.pos_emb.weight.data
    for p in range(13):
        pe[p, POS + p] = 1.0
    bal_vec(te); bal_vec(pe)

    # ---- LayerNorms: with the huge constant, LN(x) ~= x if gain = std --------
    lns = [model.final_ln] + [m for b in model.blocks for m in (b.ln1, b.ln2)]
    for ln in lns:
        ln.weight.data.fill_(std); ln.bias.data.zero_()

    # ---- Block 0 attention: 7 heads, each fetches one prompt token by absolute position
    at = model.blocks[0].attn
    S = 20.0
    for h, sp in enumerate(src_pos):
        at.W_q.weight.data[h * dh, CP] = S * math.sqrt(dh) / B     # constant query
        at.W_k.weight.data[h * dh, POS + sp] = 1.0                 # key = "I am position sp"
        at.W_v.weight.data[h * dh, NUM] = 1.0                      # value = digit value
        at.W_o.weight.data[G + h, h * dh] = 1.0
    bal_vec(at.W_o.weight.data.T)

    def mlp_set(layer, units):
        """units: list of (dict dim->coef for input, bias, dict outdim->coef)"""
        m = model.blocks[layer].mlp
        assert len(units) <= m.fc1.weight.shape[0]
        for u, (win, b, wout) in enumerate(units):
            for k, v in win.items():
                m.fc1.weight.data[u, k] = v
            m.fc1.bias.data[u] = b
            for k, v in wout.items():
                m.fc2.weight.data[k, u] = v
        bal_vec(m.fc2.weight.data.T)

    tri = [(1, 1.0), (0, -2.0), (-1, 1.0)]   # relu(z+1)-2relu(z)+relu(z-1) = [z == 0] for integer z

    # ---- Block 0 MLP: slope detector  I_a = [a*(x2-x1) == y2-y1]
    units = []
    for a in range(1, 6):
        win = {X2: a, X1: -a, Y2T: -10, Y1T: 10, Y2O: -1, Y1O: 1}
        for s, c in tri:
            units.append((win, s, {IA + a: c}))
    mlp_set(0, units)

    # ---- Block 1 MLP: vsum = sum_a a*I_a*t  (t = q-x1), gating t by I_a
    M = 9.0
    units = []
    for a in range(1, 6):
        units.append(({Q: 1, X1: -1, IA + a: M}, -M, {VS: a}))
        units.append(({Q: -1, X1: 1, IA + a: M}, -M, {VS: -a}))
    mlp_set(1, units)

    # ---- Block 2 MLP: tens digit of v = y1 + vsum  as one-hot (trapezoid on [10i, 10i+9])
    vin = {Y1T: 10, Y1O: 1, VS: 1}
    units = []
    for i in range(6):
        for s, c in [(1, 1.0), (0, -1.0), (-9, -1.0), (-10, 1.0)]:
            units.append((vin, s - 10 * i, {TF + i: c}))
    mlp_set(2, units)

    # ---- Block 3 MLP: output logits.  pos 11 -> tens digit, pos 12 -> ones digit
    units = []
    for i in range(6):                                   # tens, gated by "pos==11"
        units.append(({TF + i: 1, POS + 11: 1}, -1.0, {L + i: 1.0}))
    big = 20.0
    ones_in = dict(vin)
    for i in range(6):
        ones_in[TF + i] = -10 * i                        # ones = v - 10*tens
    for j in range(10):                                  # ones digit, gated by "pos==12"
        for s, c in tri:
            w = dict(ones_in); w[POS + 12] = big
            units.append((w, s - j - big, {L + j: c}))
    mlp_set(3, units)

    # ---- head: logit of digit token d = L_d
    for dgt in range(10):
        model.head.weight.data[task.stoi[str(dgt)], L + dgt] = 1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "Final_Compact_d48"
model_description = ("Compact 4-layer hand-wired (d=48, 8 heads, d_ff=36): L0 attn gathers 7 prompt digits by absolute position, "
                     "MLP detects slope a=dy/dx via equality tests; L1 gates (q-x1) by slope -> y=y1+a(q-x1); L2/L3 turn y into "
                     "tens/ones digit one-hots gated by position. +-B constant dims + a balance dim make LayerNorm an identity.")


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
    max_seq_len = task.seq_len
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len,
                              d_model=48, n_heads=8, n_layers=4, d_ff=36)
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
